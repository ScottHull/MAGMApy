import pandas as pd
from copy import copy
from math import sqrt, log10
import sys

from src.composition import *
from src.k_constants import get_K


def get_base_oxide_reactants(species, all_oxides):
    """
    Get the base oxide reactants from a given product.
    i.e. in the reaction MgO(liq) + SiO2(liq) = MgSiO3(liq), we would want to find the base oxide reactants and stoich
    from MgSiO3.
    The stoich would have to satisfy the following:
        K = MgSiO3 / [ SiO2 * MgO ]
    However, we can have more complex stoich.  For example:
        K = Mg2SiO4 / [ MgO^2 * SiO2 ]
    We will assume that we can balance these equations with cation stoich.
    :param all_oxides:
    :param species:
    :return:
    """
    reactants = {}
    product_stoich = get_molecule_stoichiometry(molecule=species)
    for i in all_oxides:
        stoich = get_molecule_stoichiometry(molecule=i)
        for j in stoich.keys():
            if j != "O":
                if j in product_stoich.keys():
                    reactants.update(
                        {i: product_stoich[j] / stoich[j]})  # i.e. 1 Mg2SiO4 requires 2 MgO, so we need 2/1 MgO
    # TODO: fix hardcoding below
    if "Fe2O3" in reactants.keys() and species != "Fe3O4":  # only want to describe Fe-containing species in terms of FeO
        del reactants["Fe2O3"]
    if species == "Fe3O4":
        reactants["Fe2O3"] = 1.0
        reactants["FeO"] = 1.0
    return reactants


def get_base_oxide_in_complex_species_stoich(species, all_species):
    """
    Given a base species, returns the stoichiometry of that species in a different element.
    For example, given K2O, it will return a stoichiometry for K2SiO3 of 1 because K2O appears in K2SiO3 1 time.
    For KAlO2, this would be 0.5, since K2O appears in KAlO2 0.5 times.
    The returned keys are the complex species, not the base species.
    :param species:
    :param all_species:
    :return:
    """
    if species == "Fe2O3":
        return {"Fe3O4": 1}  # the appearance of Fe2O3 is already accounted for in the activity coefficient function
    d = {}
    species_stoich = get_molecule_stoichiometry(molecule=species, return_oxygen=False)
    for i in species_stoich.keys():
        for j in all_species:
            stoich = get_molecule_stoichiometry(molecule=j, return_oxygen=False)
            if i in stoich.keys():
                d.update({j: stoich[i] / species_stoich[i]})
    return d


class LiquidActivity:
    """
    Handles activity calculates for all species in the liquid phase.
    """

    def __init__(self, composition, complex_species, gas_system):
        self.liquid_oxide_mass_fraction = None  # wt% of the liquid oxide in the liquid phase
        self.liquid_oxide_mole_fraction = None  # mol% of the liquid oxide in the liquid phase
        self.liquid_oxide_masses = None  # absolute mass of the liquid oxide composition
        self.liquid_oxide_moles = None  # absolute molar abundances of liquid cations in the liquid
        self.complex_species_data = pd.read_excel("data/MAGMA_Thermodynamic_Data.xlsx", sheet_name="Table 3",
                                                  index_col="Product")
        self.initial_oxide_melt_masses = {}  # initial oxide melt masses
        self.initial_element_melt_masses = {}  # initial element melt masses
        self.complex_species = complex_species  # a list of complex species to consider in the model
        if complex_species == "__all__":  # include all complex species in the spreadsheet
            self.complex_species = self.complex_species_data.index.tolist()
        self.composition = composition  # composition object
        self.activity_coefficients = self.__get_initial_activty_coefficients()
        self.activities = self.__initial_activity_setup()  # melt activities
        self.counter = 0  # for tracking Fe2O3
        self.iteration = 0  # for tracking the number of iterations in the activity convergence loop
        self.initial_melt_mass = self.__get_initial_melt_mass()  # calculate the initial mass of the melt in order to calculate vapor%
        self.melt_mass = 1e-99  # absolute mass of the melt
        self.cation_mass = {}  # mass of cations in the liquid
        self.cation_mass_fraction = self.get_cation_fraction_from_moles()  # mass fraction of cations in liquid
        self.gas_system = gas_system  # gas system object
        self.temperature = -100000  # temperature of the liquid

    def get_cation_fraction_from_moles(self, include_O=True):
        """
        Gets cation mass fraction of liquid from moles of cations in liquid.
        :return:
        """
        cations = self.composition.liquid_abundances
        self.cation_mass = {i: self.composition.liquid_abundances[i] * self.composition.get_molecule_mass(molecule=i)
                            for i in cations.keys()}
        if include_O:  # assume all missing mass between liquid mass and cation mass is due to O
            if self.melt_mass < 1e-90:  # likely initial iteration where we dont yet know melt mass
                self.cation_mass.update({"O": 1e-99})  # make it very small
            else:
                self.cation_mass.update({"O": self.melt_mass - sum(self.cation_mass.values())})
        self.cation_mass_fraction = {i: self.cation_mass[i] / sum(self.cation_mass.values()) for i in
                                     self.cation_mass.keys()}
        return self.cation_mass_fraction

    def get_liquid_oxide_mass_fraction(self):
        """
        Returns a diction of the liquid cation moles, represented as oxide mass fraction.
        :return:
        """
        liquid_moles = self.composition.liquid_abundances  # the absolute liquid molar abundances
        # get the absolute oxide molar abundances
        self.liquid_oxide_moles = self.composition.cations_to_oxides(liquid_moles,
                                                                     list(self.composition.oxide_mole_fraction.keys()))
        # calculate the mole fraction of the liquid oxide
        self.liquid_oxide_mole_fraction = {i: self.liquid_oxide_moles[i] / sum(self.liquid_oxide_moles.values()) for
                                           i in self.liquid_oxide_moles.keys()}
        # convert absolute moles to absolute mass
        self.liquid_oxide_masses = self.composition.moles_to_mass(self.liquid_oxide_moles)
        # make sure mass has been conserved
        if abs(sum(self.liquid_oxide_masses.values()) - self.melt_mass) > 10 ** -4:
            if "O" not in self.liquid_oxide_moles.keys():  # extra mass will be due to missing O in cation dict
                pass  # I don't feel like dealing with this right now
            else:
                raise Exception("Mass of liquid was conserved in liquid oxide calculation.")
        # convert absolute mass to mass fraction
        self.liquid_oxide_mass_fraction = self.composition.mass_to_mass_percent(self.liquid_oxide_masses)
        return self.liquid_oxide_mass_fraction

    def __get_initial_melt_mass(self):
        """
        Returns the initial mass of the melt for vaporization
        :return:
        """
        self.initial_oxide_melt_masses = {}
        self.initial_element_melt_masses = {}
        initial_melt_mass = 0.0
        for i in self.composition.cation_fraction.keys():
            # get the base oxide for the elment (i.e. SiO2 for Si)
            base_oxide = get_element_in_base_oxide(element=i, oxides=self.composition.mole_pct_composition)
            oxide_mw = self.composition.get_molecule_mass(molecule=base_oxide)  # get molecular weight of oxide
            oxide_stoich = get_molecule_stoichiometry(molecule=base_oxide)
            oxide_mass = self.composition.liquid_abundances[i] * oxide_mw * (1.0 / oxide_stoich[i])
            for element in oxide_stoich.keys():
                if element not in self.initial_element_melt_masses.keys():
                    self.initial_element_melt_masses.update({element: 0.0})
                if base_oxide not in self.initial_oxide_melt_masses.keys():
                    self.initial_oxide_melt_masses.update({base_oxide: 0.0})
                self.initial_element_melt_masses[element] += oxide_mass / oxide_mw * oxide_stoich[element] * self.composition.get_molecule_mass(element)
            initial_melt_mass += oxide_mass
            self.initial_oxide_melt_masses[base_oxide] += oxide_mass
        # check conservation of mass between all three values
        if abs(sum(self.initial_oxide_melt_masses.values()) - sum(self.initial_element_melt_masses.values())) > \
                10 ** -3 and abs(initial_melt_mass - sum(self.initial_element_melt_masses.values())) > 10 ** -3:
            raise Exception(
                f"Mass of melt was not conserved in initial melt mass calculation.\n"
                f"Inital oxide mass: {sum(self.initial_oxide_melt_masses.values())}\n"
                f"Initial element mass: {sum(self.initial_element_melt_masses.values())}\n"
                f"Initial melt mass: {initial_melt_mass}"
            )
        return initial_melt_mass

    def __get_initial_activty_coefficients(self):
        """
        Assume initial activity coefficients are 1.
        :return:
        """
        coeffs = {}
        for i in self.composition.oxide_mole_fraction.keys():  # loop through all cations in composition
            coeffs.update({i: 1.0})  # assume initial ideal behavior (gamma = 1)
        # TODO: handle Fe2O3 better...
        coeffs.update({"Fe2O3": 1.0})
        return coeffs

    def __initial_activity_setup(self):
        """
        Placeholder values for the activities dictionary.
        :return:
        """
        activities = {}
        for i in self.composition.moles_composition:  # iterate through oxides
            activities.update({i: 0.0})  # set placeholder activities for oxides in melt
        for i in self.complex_species:
            activities.update({i: 0.0})  # set placeholder activities for complex species in melt
        return activities

    def __calculate_oxide_activities(self):
        """
        Calculates the activities of the base oxides in the melt (i.e. a_SiO2).
        :return:
        """
        for i in self.composition.moles_composition.keys():  # iterate through oxides
            if i == "Fe2O3":
                if self.counter == 1:
                    self.activities[i] = 0.0
                else:
                    # use gas chemistry to estimate Fe2O3
                    self.activities[i] = self.activity_coefficients[i] * \
                                         self.gas_system.partial_pressures_minor_species[i + "_l"]
            else:
                # Henrian Behavior... a_i = gamma_i * x_i
                self.activities[i] = self.activity_coefficients[i] * self.composition.oxide_mole_fraction[i]

    def __calculate_complex_species_activities(self, temperature):
        """
        Calculate the activities of complex species in the melt (i.e. MgSiO3).
        :param temperature:
        :return:
        """
        for i in self.complex_species:
            # TODO: fix this Fe2O3 hardcoding
            if i != "Fe2O3":
                # for example, if we have MgSiO3, then we want MgO and SiO2
                reactants = get_base_oxide_reactants(species=i, all_oxides=self.composition.moles_composition.keys())
                tmp_activity = get_K(df=self.complex_species_data, species=i, temperature=temperature)
                for j in reactants.keys():
                    # i.e. for K_Mg2SiO4, we need to multiply it by MgO^2 and SiO2^1
                    tmp_activity *= self.activities[j] ** reactants[j]
                self.activities[i] = tmp_activity
        return self.activities

    def __calculate_activities(self, temperature):
        """
        Calculate activities of base oxides and complex species in the melt.
        :param temperature:
        :return:
        """
        self.__calculate_oxide_activities()  # calculate activities of base oxides (i.e. SiO2)
        # calculate activities of complex species from base oxide activities (i.e. to calculate a_MgSiO3,
        # you need to know a_SiO2)
        self.__calculate_complex_species_activities(temperature=temperature)
        return self.activities

    def __check_activity_coefficient_convergence(self, criterion=1e-5):
        """
        If the solution for an activity has converged, return True.  Else, return False.
        :param criterion:
        :return:
        """
        self.ratios = {}
        passed_ratios = {}
        for i in self.activity_coefficients:
            r = None
            if self.activities[i] != 0:  # some activities can be 0 and we want to zero division error
                r = abs(log10(self.activity_coefficients[i] / self.previous_activity_coefficients[i]))
                self.ratios.update({i: self.activity_coefficients[i] / self.previous_activity_coefficients[i]})
            else:
                r = 0.0
                self.ratios.update({i: 0})
            passed = False
            if r <= criterion:  # has the solution converged?
                passed = True
            passed_ratios.update({i: passed})
        if False in passed_ratios.values():  # if the solution has NOT converged
            return False
        return True  # if the solution has converged

    def __calculate_activity_coefficients(self):
        """
        Returns activity coefficients of base elements in the system (i.e. Si, Mg, Fe, ...).
        gamma_i = a_ox / sum(a_ci)
        where `ox` is the pure oxide containing that element (i.e. SiO2 for Si) 'a_ci' is all of the complex species
        activities containing that element.
        :return:
        """
        self.previous_activity_coefficients = copy(
            self.activity_coefficients)  # make a copy of old activities so that we can reference it for solution convergence later
        for i in self.activity_coefficients.keys():
            if self.activities[i] != 0.0:  # don't do anything if activity = 0 to avoid divide by 0 errors
                sum_activities_complex = self.activities[
                    i]  # the sum of activities of all complex species containing element i, including the base oxide
                # get the appearances of the element in all complex species
                base_oxide_appearances = get_base_oxide_in_complex_species_stoich(species=i,
                                                                                  all_species=self.complex_species)
                for j in base_oxide_appearances.keys():
                    # the element stoich times the activity of the containing complex species
                    # i.e. for Si, you would need 2 * CaMgSi2O6 since Si has a stoich of 2
                    sum_activities_complex += base_oxide_appearances[j] * self.activities[j]
                self.activity_coefficients[i] = self.activities[i] / sum_activities_complex
            else:
                self.activity_coefficients[i] = 0.0
            if i == "Fe2O3" and self.counter == 1:
                self.activity_coefficients[i] = 1.0
        return self.activity_coefficients

    def __adjust_activity_coefficients(self):
        """
        Adjust activity coefficients by their geometric means in order to achieve convergence.
        The activity coefficient for Fe2O3 is technically an adjustment factor and not a true activity coefficient
        because the mole fraction of Fe2O3 in the melt is not known.
        :return:
        """
        # TODO: check Fe2O3 as it should be included here.  In MAGMA it ends with 1 with no defined Fe2O3.
        # adjust activity coefficients by geometric means.
        for i in self.activity_coefficients:
            if self.iteration < 30:
                self.activity_coefficients[i] = (self.activity_coefficients[i] *
                                                 self.previous_activity_coefficients[i]) ** (1 / 2)
            elif 30 <= self.iteration < 500:
                self.activity_coefficients[i] = (self.activity_coefficients[i] * (
                        self.previous_activity_coefficients[i] ** 2)) ** (1 / 3)
            else:
                self.activity_coefficients[i] = (self.activity_coefficients[i] * (
                        self.previous_activity_coefficients[i] ** 4)) ** (1 / 5)
        return self.activity_coefficients

    def calculate_activities(self, temperature):
        """
        Master function that runs convergence solution on activity calculations.
        :return:
        """
        # print("[*] Solving for melt activities...")
        # run the initial activity calculation
        # initially assume that the activity coefficients = 1
        self.temperature = temperature
        self.counter += 1  # for handing Fe2O3
        has_converged = False
        while not has_converged:
            self.__calculate_activities(temperature=temperature)  # calculate base oxide and complex species activitiesƒ
            self.__calculate_activity_coefficients()  # calculate activity coefficients
            has_converged = self.__check_activity_coefficient_convergence()  # has the solution converged?
            self.__adjust_activity_coefficients()  # bump the activity coefficients
            self.iteration += 1  # increment the counter if it has not converged
        # print("[*] Successfully converged on melt activities!  Took {} iterations.".format(self.iteration))
        self.iteration = 0  # reset the iteration count
