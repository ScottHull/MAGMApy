import pandas as pd
from math import isnan, sqrt
import sys

from src.k_constants import get_K
from src.fugacity import get_oxygen_fugacity, fO2_Buffer
from src.composition import get_molecule_stoichiometry, get_stoich_from_sheet


def get_gas_reactants(df, species):
    """
    Gets the reactants of the gas product using the stoichiometry of Table 2 in the data Excel sheet.
    :param df:
    :param species:
    :return:
    """
    # TODO: have a way to handle gas vs. liquid reactants?
    headers = df.columns.tolist()[3:]
    row = df.loc[species].tolist()[3:]
    d = list(zip(headers, row))
    to_delete = []
    for index, i in enumerate(d):
        if isnan(i[1]):
            to_delete.append(i[0])
    d = dict(d)
    for i in to_delete:
        del d[i]
    return d


def __gas_isnan(val):
    try:
        return isnan(val)
    except:
        return False


def get_minor_gas_reactants(species, major_gasses, df):
    """
    Returns the stoichiometry of the species in terms of its major gas species,
    i.e. SiO, O2, Ca, Fe, MgO, etc.
    Assume that unless specified in the spreadsheet that each species is composed of its major gas analog.
    :param species:
    :param major_gasses:
    :return:
    """
    reactants = get_stoich_from_sheet(molecule=species, df=df)
    return reactants


def get_gas_species_major_oxide(species, major_oxides):
    """
    Given a gas species, will return its major system oxide.
    i.e. SiO will return SiO2, Fe will return FeO + Fe2O3, MgO will return MgO.
    :param species:
    :param major_oxides:
    :return:
    """
    included_major_oxides = None
    stoich = get_molecule_stoichiometry(molecule=species, return_oxygen=False)
    for i in stoich:
        for j in major_oxides:
            oxide_stoich = get_molecule_stoichiometry(molecule=j, return_oxygen=False)
            if i in oxide_stoich.keys():
                if i != "Fe2O3":
                    included_major_oxides = j
    return included_major_oxides


def get_most_abundant_gas_oxide(major_gas_species, partial_pressures):
    most_abundant = {}
    for i in major_gas_species:  # loop through all species in partial pressure dict
        if len(i) > 2:  # make sure it is not O2 or an element...
            # if this is the first loop, set the first examined oxide as the most abundant by default
            if len(most_abundant.keys()) == 0:
                most_abundant = {i: partial_pressures[i]}
            else:
                most_abundant_name = list(most_abundant.keys())[0]
                # if this partial pressure is greater than the one the system currently thinks is most abundant, change it
                if partial_pressures[i] > most_abundant[most_abundant_name]:
                    most_abundant = {i: partial_pressures[i]}
    return most_abundant


def get_element_appearances_in_gas_species(element, all_species, force_O=False):
    """
    Given an element, returns all gas species which contains it.
    :param element:
    :param all_species:
    :param force_O: Will only return O and O2 if the given element is O.
    :return:
    """
    d = {}
    for i in all_species.keys():
        if "_l" not in i:
            stoich = get_molecule_stoichiometry(molecule=i)
            if element in stoich.keys():
                d.update({i: stoich[element]})
    if force_O and element == "O":
        delete_these = []
        for i in d.keys():
            if i != "O" and i != "O2":
                delete_these.append(i)
        for i in delete_these:
            del d[i]
    return d


def get_unique_gas_elements(all_species):
    """
    Returns a list of all unique elements contained in the gas system.
    :param all_species:
    :return:
    """
    d = []
    for i in all_species:
        stoich = get_molecule_stoichiometry(molecule=i)
        for j in stoich.keys():
            d.append(j)
    return list(set(d))  # returns unique elements as a list.


class GasPressure:

    def __init__(self, composition, major_gas_species, minor_gas_species, fO2_buffer="QFM"):
        self.f = None
        self.cation_mass = None
        self.cation_moles = None
        self.total_pressure = None
        self.partial_pressures = None
        self.composition = composition
        self.minor_gas_species_data = pd.read_excel("data/MAGMA_Thermodynamic_Data.xlsx", sheet_name="Table 4",
                                                    index_col="Product")
        self.major_gas_species = major_gas_species
        self.minor_gas_species = minor_gas_species
        if minor_gas_species == "__all__":
            self.minor_gas_species = [i for i in self.minor_gas_species_data.index.tolist() if
                                      i not in self.major_gas_species]
        self.partial_pressures_major_species = self.__initial_partial_pressure_setup(
            species=self.major_gas_species)  # assume initial behavior of unity
        self.partial_pressures_minor_species = self.__initial_partial_pressure_setup(
            species=self.minor_gas_species)  # assume initial behavior of unity
        self.adjustment_factors = self.__initial_partial_pressure_setup(
            species=self.major_gas_species)  # assume initial behavior of unity
        self.pressure_to_number_density = 1.01325e6 / 1.38046e-16  # (dyn/cm**2=>atm) / Boltzmann's constant (R/AVOG)
        self.fO2_buffer = fO2_buffer
        self._buffer = fO2_Buffer()
        self.cation_mass_fraction = self.get_cation_fraction_from_moles(vapor_mass=0.0)  # mass fraction of cations in vapor
        self.oxygen_fugacity = -10000000

    def get_cation_fraction_from_moles(self, vapor_mass, include_O=True):
        """
        Gets cation mass fraction of liquid from moles of cations in liquid.
        :return:
        """
        initial_liquid_cations = self.composition.initial_liquid_number_of_moles
        liquid_cations = self.composition.liquid_abundances
        if vapor_mass == 0.0:  # return dictionary of 0 cation vapor mass if first iteration
            return {i: 0 for i in liquid_cations.keys()}
        self.cation_moles = {i: initial_liquid_cations[i] - liquid_cations[i] for i in liquid_cations.keys()}
        self.cation_mass = {i: self.cation_moles[i] * self.composition.get_molecule_mass(molecule=i)
                            for i in self.cation_moles.keys()}
        if include_O:  # assume all missing mass between liquid mass and cation mass is due to O
            if vapor_mass < 0:  # likely initial iteration where we dont yet know melt mass
                self.cation_mass.update({"O": 1e-99})  # make it very small
            else:
                self.cation_mass.update({"O": vapor_mass - sum(self.cation_mass.values())})
        self.cation_mass_fraction = {i: self.cation_mass[i] / sum(self.cation_mass.values()) for i in self.cation_mass.keys()}
        return self.cation_mass_fraction

    def get_f(self):
        """
        Returns the cation fraction in the vapor relative to vapor + melt.
        This is f in Nie and Dauphus 2019.
        :param vapor_mass:
        :param include_O:
        :return:
        """
        liquid_cations = self.composition.liquid_abundances
        vapor_cations = self.cation_moles
        self.f = {i: vapor_cations[i] / (vapor_cations[i] + liquid_cations[i]) for i in vapor_cations.keys()}

    def __initial_partial_pressure_setup(self, species):
        """
        Sets up the initial partial pressures of the major gas species to be 1.
        :return:
        """
        d = {}
        for i in species:
            d.update({i: 1.0})  # assume partial pressure of 1 initially
        return d

    def __calculate_major_gas_partial_pressures(self):
        """
        Calculates the partial pressures of the major gas species.
        Here, it is just the former pressure times the adjustment factor.
        :return:
        """
        for i in self.major_gas_species:
            self.partial_pressures_major_species[i] *= self.adjustment_factors[i]
        return self.major_gas_species

    def __get_phase(self, species):
        """
        Used in determining if the phase is a liquid or gas, used for getting the correct K constant for some species.
        :param species:
        :return:
        """
        if "_l" in species:
            return "liquid"
        else:
            return "gas"

    def __reformat_dict(self, combined_dict):
        """
        Updates the major and minor partial pressure dictionaries from the combined dictionary.
        :return:
        """
        major_species = self.partial_pressures_major_species.keys()
        for i in combined_dict.keys():
            if i in major_species:
                self.partial_pressures_major_species[i] = combined_dict[i]
            else:
                self.partial_pressures_minor_species[i] = combined_dict[i]

    def __calculate_ion_partial_pressures(self, temperature):
        # TODO: Hardcoding in the ion species...fix this
        # calculate pp_e-
        K_Na = get_K(df=self.minor_gas_species_data, species="Na+", temperature=temperature, phase="gas")
        K_K = get_K(df=self.minor_gas_species_data, species="K+", temperature=temperature, phase="gas")
        pp_Na = self.partial_pressures_major_species['Na']
        pp_K = self.partial_pressures_major_species['K']
        self.partial_pressures_minor_species["e-"] = sqrt((K_Na * pp_Na) + (K_K * pp_K))
        # adjust ion species based on e-
        for i in self.partial_pressures_minor_species.keys():
            if "+" in i:
                base_species_pp = self.partial_pressures_major_species[i.replace("+", "")]
                if base_species_pp != 0.0:
                    self.partial_pressures_minor_species[i] /= self.partial_pressures_minor_species["e-"]
        # end hardcoding

    def __calculate_minor_gas_partial_pressures(self, temperature):
        """
        Calculates the partial pressures of the minor gas species.
        This follows the relation K = P_prod / prod(P_react) --> P_prod = K * prod(P_react)
        :return:
        """
        combined_partial_pressures = {**self.partial_pressures_major_species, **self.partial_pressures_minor_species}
        for i in self.minor_gas_species:
            # define if we have a liquid or gas component, important for getting K of Fe2O3_l
            # for example, if we have MgSiO3, then we want MgO and SiO2
            reactants = get_minor_gas_reactants(species=i, major_gasses=self.major_gas_species,
                                                df=self.minor_gas_species_data)
            tmp_activity = get_K(df=self.minor_gas_species_data, species=i, temperature=temperature,
                                 phase="gas")
            for j in reactants.keys():
                # i.e. for K_SiO2, take product with partial pressures of Si and O2
                tmp_activity *= combined_partial_pressures[j] ** reactants[j]
            combined_partial_pressures[i] = tmp_activity
        self.__reformat_dict(
            combined_dict=combined_partial_pressures)  # replace partial pressures with just calculated ones
        self.__calculate_ion_partial_pressures(temperature=temperature)
        return self.partial_pressures_minor_species

    def __get_nd(self, pp, t):
        """
        Molecular number density formula for gasses.
        :param pp:
        :param t:
        :return:
        """
        return self.pressure_to_number_density * pp / t

    def __calculate_number_density_molecules(self, temperature):
        """
        Assume ideal behavior: PV = nRT --> n = P / RT, where n is now the number density rather than moles.
        We will use the pressure --> number density conversion factor (which has R rolled into it).
        Therefore, n = c * P_i / T, where P_i is the partial pressure and c is the conversion factor.
        :param temperature:
        :return:
        """
        self.number_densities_gasses = {}
        for i in self.major_gas_species + self.minor_gas_species:
            if "_l" not in i:  # we don't want liquid species, only gasses
                if i in self.major_gas_species:
                    pp = self.partial_pressures_major_species[i]
                else:
                    pp = self.partial_pressures_minor_species[i]
                self.number_densities_gasses.update({i: self.__get_nd(pp=pp, t=temperature)})
        return self.number_densities_gasses

    def __calculate_number_density_elements(self):
        """
        The number densities of elements in the gas is equal to the sum of its abundances in the molecular gas species.
        Requres that the number densities of the molecules be calculated first.
        :return:
        """
        self.number_densities_elements = {}
        unique_elements = get_unique_gas_elements(all_species=self.major_gas_species + self.minor_gas_species)
        for i in unique_elements:
            if i not in self.number_densities_elements.keys():
                self.number_densities_elements.update({i: 0})  # if the cation is not already in the dict, add it
            molecule_appearances = get_element_appearances_in_gas_species(element=i,
                                                                          all_species=self.number_densities_gasses)
            for m in molecule_appearances.keys():
                self.number_densities_elements[i] += molecule_appearances[m] * self.number_densities_gasses[m]
        return self.number_densities_elements

    def __ratio_number_density_to_oxygen(self):
        """
        This returns the ratio of the number density of the OXIDE GASSES to the number density of oxygen.
        :return:
        """
        base_oxides = self.composition.mole_pct_composition.keys()
        total_oxide_number_density = 0
        for i in base_oxides:
            stoich = get_molecule_stoichiometry(molecule=i)
            for j in stoich:
                if j != "O":
                    # We want to treat oxides on a single cation basis, i.e. Al2O3 -> AlO1.5
                    # Therefore, to get Al2O3 ->AlO1.5, we must multiply n_Al by (3/2 = 1.5)
                    total_oxide_number_density += (stoich["O"] / stoich[j]) * self.number_densities_elements[j]
        return total_oxide_number_density / self.number_densities_elements["O"]

    def __calculate_number_densities(self, temperature):
        """
        Main function for calculating the number densities of both molecules and elements in the vapor phase.
        :param temperature:
        :return:
        """
        self.__calculate_number_density_molecules(temperature=temperature)
        self.__calculate_number_density_elements()

    # def __get_most_abundant_oxide(self, liquid_system):
    #     oxides = self.composition.oxide_mole_fraction.keys()
    #     most_abundant_name = None
    #     most_abundant_cf = 0
    #     most_abudant_activity_coeff = 0
    #     most_abundant_pp = 0
    #     for i in oxides:
    #         mole_fraction = self.composition.oxide_mole_fraction[i.replace("_l", "")]
    #         activity_coeff = liquid_system.activity_coefficients[i.replace("_l", "")]
    #         partial_pressure = self.partial_pressures_minor_species[i + "_l"]
    #         if mole_fraction > most_abundant_cf:
    #             most_abundant_name = i
    #             most_abundant_cf = mole_fraction
    #             most_abudant_activity_coeff = activity_coeff
    #             most_abundant_pp = partial_pressure
    #     return most_abundant_name, most_abundant_cf, most_abudant_activity_coeff, most_abundant_pp
    #
    # def __hack_most_abundant_oxide(self, liquid_system, rat):
    #     """
    #     Returns the adjustment factor for O2.
    #     This is a hack function.  Come back later and gut this thing.
    #     :return:
    #     """
    #     most_abundant_name, most_abundant_cf, most_abudant_activity_coeff, most_abundant_pp = self.__get_most_abundant_oxide(
    #         liquid_system=liquid_system)
    #     if most_abundant_name == "FeO":
    #         adjustment_fe2o3 = self.adjustment_factors['Fe2O3_l']
    #         partial_pressure_fe2o3 = self.partial_pressures_minor_species['Fe2O3_l']
    #         return rat * (most_abundant_cf * most_abudant_activity_coeff + adjustment_fe2o3) / (
    #                 most_abundant_pp + partial_pressure_fe2o3)
    #     else:
    #         return rat * most_abundant_cf * most_abudant_activity_coeff / most_abundant_pp

    def __hack_most_abundant_oxide(self, liquid_system, rat):
        """
        Returns the adjustment factor for O2.
        This is a hack function.  Come back later and gut this thing.
        :return:
        """
        # TODO: gut this thing and make it so that its not hardcoded
        order = ["SiO2_l", "MgO_l", "FeO_l", "CaO_l", "Al2O3_l", "TiO2_l", "Na2O_l", "K2O_l"]
        for i in order:
            mole_fraction = self.composition.oxide_mole_fraction[i.replace("_l", "")]
            activity_coeff = liquid_system.activity_coefficients[i.replace("_l", "")]
            partial_pressure = self.partial_pressures_minor_species[i]
            if partial_pressure != 0 and mole_fraction != 0:
                if i == "FeO":
                    adjustment_fe2o3 = self.adjustment_factors['Fe2O3_l']
                    partial_pressure_fe2o3 = self.partial_pressures_minor_species['Fe2O3_l']
                    return rat * (mole_fraction * activity_coeff + adjustment_fe2o3) / (
                            partial_pressure + partial_pressure_fe2o3)
                else:
                    return rat * mole_fraction * activity_coeff / partial_pressure

    def __calculate_adjustment_factors(self, oxides_to_oxygen_ratio, liquid_system):
        """
        Calculates gas species adjustment factors.
        The abundance of O2 is governed by most abundance oxide in the melt (normally SiO2).  Once this oxide is
        completely vaporized, a_O2_g is computed from the remaining species in the melt.
        """
        adjustment_factors = {}
        for i in self.major_gas_species:
            if i != "O2" and i != "Fe2O3":
                major_oxide = get_gas_species_major_oxide(species=i,
                                                          major_oxides=self.composition.moles_composition.keys())
                cf = self.composition.oxide_mole_fraction[major_oxide]  # cation fraction in composition
                gamma = liquid_system.activity_coefficients[major_oxide]  # liquid activity coefficient
                # TODO: better way to get major oxide liquid?
                pp = self.partial_pressures_minor_species[major_oxide + "_l"]
                # calculate adjustment factors for major gas species
                # TODO: fix this hack hardcoding
                if pp != 0 and cf != 0 and gamma != 0:
                    if i == "Fe":
                        adjust = (cf * liquid_system.activity_coefficients["FeO"] + liquid_system.activities[
                            "Fe2O3"]) / (
                                         self.partial_pressures_minor_species['FeO_l'] +
                                         self.partial_pressures_minor_species[
                                             'Fe2O3_l'])
                    elif i == "MgO":
                        adjust = cf * gamma / pp
                    elif i == "SiO":
                        adjust = 1.0 / (oxides_to_oxygen_ratio * sqrt(pp / (cf * gamma)))
                    else:
                        adjust = 1.0 / sqrt(pp / (cf * gamma))
                elif cf == 0.0 or gamma == 0.0:
                    adjust = 0.0
                else:
                    adjust = 1.0
                adjustment_factors.update({i: adjust})
        # BELOW IS ADJUSTMENT FACTOR CODE FOR O2, WHICH IS GOVERNED BY MOST ABUNDANT OXIDE
        # get most abundant oxide (mao) and use it to calculate adjustment factor for O2
        adjustment_factors.update(
            {"O2": self.__hack_most_abundant_oxide(rat=oxides_to_oxygen_ratio, liquid_system=liquid_system)})
        return adjustment_factors

    def __have_adjustment_factors_converged(self):
        """
        If all adjustment factors are either 1 or 0, then the solution has converged and we can move on.
        If this criteria is not met, then we must repeat the gas chemistry calculations until we do converge.
        :return:
        """
        converged = {}
        for i in self.adjustment_factors:
            has_converged = False
            # want adjustment factors to be 1 or 0
            if (9.99997697418e-1 < self.adjustment_factors[i] < 1.00000230259) or self.adjustment_factors[i] == 0.0:
                has_converged = True
            converged.update({i: has_converged})
        if False in converged.values():  # if a species failed to converge, then False and we must redo the calculation
            return False
        else:  # if all species have converged, then we can report True and move on
            return True

    def __calculate_gas_molecules_pressures(self, temperature, liquid_system):
        """
        The activity of the oxides calculating during the liquid calculations and those calculated by the gas chemistry
        must agree.  If the oxide mole fraction for the element is 0, then that element should NOT be in the vapor
        (i.e. a_el = 0).
        :param liquid_system:
        :param temperature:
        :return:
        """
        iteration = 0
        has_converged = False
        while has_converged is False:
            self.__calculate_major_gas_partial_pressures()
            self.__calculate_minor_gas_partial_pressures(temperature=temperature)
            self.__calculate_number_densities(temperature=temperature)
            oxides_to_oxygen_ratio = self.__ratio_number_density_to_oxygen()
            self.adjustment_factors = self.__calculate_adjustment_factors(oxides_to_oxygen_ratio=oxides_to_oxygen_ratio,
                                                                          liquid_system=liquid_system)
            has_converged = self.__have_adjustment_factors_converged()
            iteration += 1
        # print("[*] Found gas partial pressures!  Took {} iterations.".format(iteration))

    def __calculate_mole_fractions(self):
        """
        The gas mole fractions are simply the partial pressures of each species divided by the total pressure.
        This returns the fractions of the species, not the cations as in the "total mole fractions".
        :return:
        """
        self.mole_fractions = {}
        for i in self.partial_pressures_minor_species.keys():
            self.mole_fractions.update({i: self.partial_pressures_minor_species[i] / self.total_pressure})
        for i in self.partial_pressures_major_species.keys():
            self.mole_fractions.update({i: self.partial_pressures_major_species[i] / self.total_pressure})
        return self.mole_fractions

    def __calculate_total_mole_fractions(self):
        """
        The total mole fraction of the element in the gas is its number density divided by the total number density.
        Note that these are just for the cations in the system, not the molecular species.
        :return:
        """
        self.total_mole_fraction = {}
        self.cation_number_density = 0
        # get number density of all cations.  we want to skip O
        for i in self.number_densities_elements.keys():
            # TODO: why not O?
            if i != "O":
                self.cation_number_density += self.number_densities_elements[i]
        for i in self.number_densities_elements.keys():
            # TODO: why not O?
            if i != "O":
                self.total_mole_fraction.update(
                    {i: self.number_densities_elements[i] / self.cation_number_density})
        return self.total_mole_fraction

    def __calculate_partial_pressure_elements(self):
        """
        Calculates the partial pressure of all elements
        :return:
        """
        self.partial_pressure_elements = {}
        elements = get_unique_gas_elements(all_species=self.major_gas_species)  # get all unique cations and anions
        for i in elements:  # build the dictionary to old the unique elements
            self.partial_pressure_elements.update({i: 0.0})
            # TODO: make force_O into a separate function
            force_O = False
            if i == "O":
                force_O = True
            # get all of the species
            element_appearances = get_element_appearances_in_gas_species(element=i, all_species={
                **self.partial_pressures_major_species, **self.partial_pressures_minor_species}, force_O=force_O)
            for j in element_appearances.keys():
                # get the partial pressure of the species
                if j in self.major_gas_species:
                    pp = self.partial_pressures_major_species[j]
                else:
                    pp = self.partial_pressures_minor_species[j]
                self.partial_pressure_elements[i] += pp  # add the partial pressure to the element total
        return self.partial_pressure_elements

    def calculate_pressures(self, temperature, liquid_system):
        """
        Main function for calculating molecular and elemental pressures in the gas.
        :param temperature:
        :param liquid_system:
        :return:
        """
        # print("[*] Solving for gas partial pressures...")
        self.__calculate_gas_molecules_pressures(temperature=temperature, liquid_system=liquid_system)
        self.__calculate_partial_pressure_elements()
        self.total_pressure = sum(self.partial_pressure_elements.values())
        self.__calculate_mole_fractions()
        self.__calculate_total_mole_fractions()
        self.partial_pressures = {**self.partial_pressures_major_species, **self.partial_pressures_minor_species}
        self.oxygen_fugacity = self._buffer.get_fO2_buffer(liquid_system.temperature, self.total_pressure,
                                                           get_oxygen_fugacity(self), self.fO2_buffer)
