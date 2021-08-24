import pandas as pd
from copy import copy
from math import sqrt, log10

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
    if "Fe2O3" in reactants.keys():  # only want to describe Fe-containing species in terms of FeO
        del reactants["Fe2O3"]
    return reactants


class LiquidActivity:
    """
    Handles activity calculates for all species in the liquid phase.
    """

    def __init__(self, composition, complex_species):
        self.complex_species_data = pd.read_excel("data/MAGMA_Thermodynamic_Data.xlsx", sheet_name="Table 3",
                                                  index_col="Product")
        self.complex_species = complex_species  # a list of complex species to consider in the model
        if complex_species == "__all__":
            self.complex_species = self.complex_species_data.index.tolist()
        self.composition = composition
        self.activity_coefficients = self.__get_initial_activty_coefficients()
        self.activities = self.__initial_activity_setup()  # melt activities
        self.counter = 1  # for tracking Fe2O3
        self.iteration = 0  # for tracking the number of iterations in the activity convergence loop
        self.initial_melt_mass = self.__get_initial_melt_mass()  # calculate the initial mass of the melt in order to calculate vapor%

    def __get_initial_melt_mass(self):
        """
        Returns the initial mass of the melt for vaporization
        :return:
        """
        initial_melt_mass = 0.0
        for i in self.composition.cation_fraction.keys():
            # get the base oxide for the elment (i.e. SiO2 for Si)
            base_oxide = get_element_in_base_oxide(element=i, oxides=self.composition.mole_pct_composition)
            oxide_mw = self.composition.get_molecule_mass(molecule=base_oxide)  # get molecular weight of oxide
            oxide_stoich = get_molecule_stoichiometry(molecule=base_oxide)
            initial_melt_mass += self.composition.planetary_abundances[i] * oxide_mw * (1.0 / oxide_stoich[i])
        return initial_melt_mass

    def __get_initial_activty_coefficients(self):
        """
        Assume initial activity coefficients are 1.
        :return:
        """
        coeffs = {}
        for i in self.composition.oxide_mole_fraction.keys():  # loop through all cations in composition
            coeffs.update({i: 1.0})  # assume initial ideal behavior (gamma = 1)
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
        for i in self.composition.moles_composition:  # iterate through oxides
            if i == "Fe2O3":
                if self.counter == 1:
                    self.activities[i] = 0.0
                else:
                    self.activities[i] = self.activity_coefficients[i] * \
                                         self.composition.oxide_mole_fraction[
                                             i]
            else:
                # Henrian Behavior... a_i = gamma_i * x_i
                self.activities[i] = self.activity_coefficients[i] * \
                                     self.composition.oxide_mole_fraction[
                                         i]

    def __calculate_complex_species_activities(self, temperature):
        """
        Calculate the activities of complex species in the melt (i.e. MgSiO3).
        :param temperature:
        :return:
        """
        for i in self.complex_species:
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
        ratios = {}
        for i in self.activities:
            r = None
            if self.activities[i] != 0:  # some activities can be 0 and we want to zero division error
                r = abs(log10(self.activities[i] / self.previous_activities[i]))
            else:
                r = 0.0
            passed = False
            if r <= criterion:  # has the solution converged?
                passed = True
            ratios.update({i: passed})
        if False in ratios.values():  # if the solution has NOT converged
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
        self.previous_activities = copy(
            self.activities)  # make a copy of old activities so that we can reference it for solution convergence later
        for i in self.activity_coefficients.keys():
            if self.activities[i] != 0:  # don't do anything if activity = 0 to avoid divide by 0 errors
                stoich = get_molecule_stoichiometry(molecule=i, return_oxygen=False)  # we want to get the base cation of the base oxide, i.e. Si from SiO2
                for j in stoich.keys():
                    # get the appearances of the element in all complex species
                    complex_appearances = get_species_with_element_appearance(element=j, species=self.complex_species)
                    sum_activities_complex = 0  # the sum of activities of all complex species containing element i
                    for j in complex_appearances.keys():
                        # the element stoich times the activity of the containing complex species
                        # i.e. for Si, you would need 2 * CaMgSi2O6 since Si has a stoich of 2
                        sum_activities_complex += complex_appearances[j] * self.activities[j]
                    self.activity_coefficients[i] = self.activities[i] / sum_activities_complex
        return self.activity_coefficients

    def __adjust_activity_coefficients(self):
        """
        Adjust activity coefficients.
        :return:
        """
        for i in self.activity_coefficients:
            if self.iteration < 30:
                # adjust by geometric mean
                self.activity_coefficients[i] = sqrt(self.activity_coefficients[i] * self.previous_activities[i])
            elif 30 <= self.iteration < 500:
                # adjust in a different way
                self.activity_coefficients[i] = (self.activity_coefficients[i] * (
                        self.previous_activities[i] ** 2)) ** (1 / 3)
            else:
                self.activity_coefficients[i] = (self.activity_coefficients[i] * (
                        self.previous_activities[i] ** 4)) ** (1 / 5)
        return self.activity_coefficients

    def calculate_activities(self, temperature):
        """
        Master function that runs convergence solution on activity calculations.
        :return:
        """
        print("[*] Solving for melt activities...")
        # run the initial activity calculation
        self.__calculate_activities(temperature=temperature)  # calculate base oxide and complex species activities
        self.__calculate_complex_species_activities(temperature=temperature)  # calculate complex species activities
        print(self.activities)
        self.__calculate_activity_coefficients()  # calculate activity coefficients
        has_converged = self.__check_activity_coefficient_convergence()  # has the solution converged?
        while not has_converged:
            print("[~] At iteration {}...".format(self.iteration))
            self.__adjust_activity_coefficients()  # bump the activity coefficients
            self.__calculate_activities(temperature=temperature)  # calculate base oxide and complex species activities
            self.__calculate_complex_species_activities(temperature=temperature)  # calculate complex species activities
            self.__calculate_activity_coefficients()  # calculate activity coefficients
            has_converged = self.__check_activity_coefficient_convergence()  # has the solution converged?
            self.iteration += 1
        print("[*] Successfully converged on melt activities!")
        self.iteration = 0  # reset the iteration count
