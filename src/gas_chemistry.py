import pandas as pd
from math import isnan, sqrt

from src.k_constants import get_K
from src.composition import get_species_with_element_appearance, get_molecule_stoichiometry


def get_gas_reactants(df, species):
    """
    Gets the reactants of the gas product using the stoichiometry of Table 2 in the data Excel sheet.
    :param df:
    :param species:
    :return:
    """
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


def get_most_abundance_gas_oxide(partial_pressures):
    most_abundant = {}
    for i in partial_pressures.keys():  # loop through all species in partial pressure dict
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


class GasPressure:

    def __init__(self, composition, major_gas_species, minor_gas_species, ion_gas_species):
        self.composition = composition
        self.minor_gas_species_data = pd.read_excel("data/MAGMA_Thermodynamic_Data.xlsx", sheet_name="Table 2",
                                                    index_col="Product")
        self.major_gas_species = major_gas_species
        self.minor_gas_species = minor_gas_species
        if minor_gas_species == "__all__":
            self.minor_gas_species = [i for i in self.minor_gas_species_data.index.tolist() if
                                      i not in self.major_gas_species]
        self.ion_gas_species = ion_gas_species
        self.partial_pressures_molecules = self.__initial_partial_pressure_setup()  # assume initial behavior of unity
        self.adjustment_factors = self.__initial_partial_pressure_setup()  # assume initial behavior of unity
        self.pressure_to_number_density = 1.01325e6 / 1.38046e-16  # (dyn/cm**2=>atm) / Boltzmann's constant (R/AVOG)

    def __initial_partial_pressure_setup(self):
        """
        Sets up the initial partial pressures of the major gas species to be 1.
        :return:
        """
        d = {}
        for i in self.major_gas_species:
            d.update({i: 1.0})  # assume partial pressure of 1 initially
        return d

    def __calculate_major_gas_partial_pressures(self):
        """
        Calculates the partial pressures of the major gas species.
        :return:
        """
        for i in self.major_gas_species:
            self.partial_pressures_molecules[i] = self.partial_pressures_molecules[i] * self.adjustment_factors[i]
        return self.major_gas_species

    def __calculate_minor_gas_partial_pressures(self, temperature):
        """
        Calculates the partial pressures of the minor gas species.
        This follows the relation K = P_prod / prod(P_react) --> P_prod = K * prod(P_react)
        :return:
        """
        for i in self.minor_gas_species:
            # for example, if we have MgSiO3, then we want MgO and SiO2
            reactants = get_gas_reactants(df=self.minor_gas_species_data, species=i)
            tmp_activity = get_K(df=self.minor_gas_species_data, species=i, temperature=temperature)
            for j in reactants.keys():
                # i.e. for K_Mg2SiO4, we need to multiply it by MgO^2 and SiO2^1
                tmp_activity *= self.partial_pressures_molecules[j] ** reactants[j]
            self.partial_pressures_molecules[i] = tmp_activity
        return self.partial_pressures_molecules

    def __calculate_ion_gas_partial_pressures(self, temperature):
        pass

    def __calculate_number_density_molecules(self, temperature):
        """
        Assume ideal behavior: PV = nRT --> n = P / RT, where n is now the number density rather than moles.
        We will use the pressure --> number density conversion factor (which has R rolled into it).
        Therefore, n = c * P_i / T, where P_i is the partial pressure and c is the conversion factor.
        :param temperature:
        :return:
        """
        self.number_densities_molecules = {}
        # TODO: make sure that SiO2_l is not included here, or any other liquid phases.  We only want vapor.
        #  Likewise, make sure O is included.
        for i in self.partial_pressures_molecules:
            nd = self.pressure_to_number_density * self.partial_pressures_molecules[i] / temperature
            self.number_densities_molecules.update({i: nd})
        return self.number_densities_molecules

    def __calculate_number_density_elements(self):
        """
        The number densities of elements in the gas is equal to the sum of its abundances in the molecular gas species.
        Requres that the number densities of the molecules be calculated first.
        :return:
        """
        self.number_densities_elements = {}
        for i in self.number_densities_molecules.keys():
            stoich = get_molecule_stoichiometry(molecule=i)
            for j in stoich:
                if j not in self.number_densities_elements.keys():
                    self.number_densities_elements.update({j: 0})
                molecule_appearances = get_species_with_element_appearance(element=j,
                                                                           species=self.number_densities_molecules.keys())
                for m in molecule_appearances.keys():
                    self.number_densities_elements[j] += molecule_appearances[j] * self.number_densities_molecules[j]
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
        self.total_number_density = sum(self.number_densities_elements.values())

    def __calculate_adjustment_factors(self, oxides_to_oxygen_ratio, liquid_system):
        """
        Calculates gas species adjustment factors.
        The abundance of O2 is governed by most abundance oxide in the melt (normally SiO2).  Once this oxide is
        completely vaporized, a_O2_g is computed from the remaining species in the melt.
        """
        # TODO: implement specific routines for each potential molecule.
        adjustment_factors = {}
        for i in self.partial_pressures_molecules.keys():
            activity = None
            cf = liquid_system.cation_fraction[i]  # cation fraction in composition
            gamma = liquid_system.activity_coefficients[i]  # liquid activity coefficient
            pp = self.partial_pressures_molecules[i]
            if pp != 0.0:
                activity = 1.0 / (oxides_to_oxygen_ratio * sqrt(pp / (cf * gamma)))
            elif cf == 0.0:
                activity = 0.0
            else:
                activity = 1.0
            adjustment_factors.update({i: activity})
        # BELOW IS ADJUSTMENT FACTOR CODE FOR O2, WHICH IS GOVERNED BY MOST ABUNDANT OXIDE
        # get most abundant oxide (mao) and use it to calculate adjustment factor for O2
        mao = get_most_abundance_gas_oxide(partial_pressures=self.partial_pressures_molecules)
        # get the name of the most abundant gas species and its partial pressure
        mao_name, mao_partial_pressure = list(mao.keys())[0], list(mao.values())[0]
        mao_stoich = get_molecule_stoichiometry(molecule=mao_name)
        mao_main_cation = [i for i in mao_stoich.keys() if i != "O"][0]  # get the major cation, i.e. Si from SiO
        cf = liquid_system.cation_fraction[mao_main_cation]  # cation fraction in composition
        gamma = liquid_system.activity_coefficients[mao_main_cation]  # liquid activity coefficient
        adjustment_factors.update({"O2": oxides_to_oxygen_ratio * cf * gamma / mao_partial_pressure})
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
        has_converged = False
        while has_converged is False:
            self.__calculate_major_gas_partial_pressures()
            self.__calculate_minor_gas_partial_pressures(temperature=temperature)
            self.__calculate_ion_gas_partial_pressures(temperature=temperature)
            self.__calculate_number_densities(temperature=temperature)
            oxides_to_oxygen_ratio = self.__ratio_number_density_to_oxygen()
            self.adjustment_factors = self.__calculate_adjustment_factors(oxides_to_oxygen_ratio=oxides_to_oxygen_ratio,
                                                                          liquid_system=liquid_system)
            has_converged = self.__have_adjustment_factors_converged()
        # if this is the first run-through then we need to go back and do activity calculations for Fe2O3 and Fe3O4
        # 'count' tracks this
        if liquid_system.count == 1:
            # TODO: implement this
            liquid_system.count = 0

    def __calculate_gas_elements_pressures(self):
        """
        Calculate the partial pressures of individual elements as a sum of the gas species that contain them.
        :return:
        """
        self.partial_pressures_elements = {}
        for i in self.partial_pressures_molecules.keys():
            stoich = get_molecule_stoichiometry(molecule=i)
            for j in stoich.keys():
                if j not in self.partial_pressures_elements.keys():
                    self.partial_pressures_elements.update({j: 0.0})
                self.partial_pressures_elements[j] += self.partial_pressures_molecules[i]
        return self.partial_pressures_elements

    def __calculate_mole_fractions(self):
        """
        The gas mole fractions are simply the partial pressures of each species divided by the total pressure.
        :return:
        """
        self.mole_fractions = {}
        for i in self.partial_pressures_molecules.keys():
            self.mole_fractions.update({i: self.partial_pressures_molecules[i] / self.total_pressure})
        for i in self.partial_pressures_elements.keys():
            self.mole_fractions.update({i: self.partial_pressures_elements[i] / self.total_pressure})
        return self.mole_fractions

    def __calculate_total_mole_fractions(self):
        """
        The total mole fraction of the element in the gas is its number density divided by the total number density.
        :return:
        """
        self.total_mole_fraction = {}
        for i in self.number_densities_elements.keys():
            self.total_mole_fraction.update(
                {i: self.number_densities_elements[i] / self.total_number_density})
        return self.total_mole_fraction

    def calculate_pressures(self, temperature, liquid_system):
        """
        Main function for calculating molecular and elemental pressures in the gas.
        :param temperature:
        :param liquid_system:
        :return:
        """
        self.__calculate_gas_molecules_pressures(temperature=temperature, liquid_system=liquid_system)
        self.total_pressure = sum(self.partial_pressures_elements.values())
        self.__calculate_gas_elements_pressures()
        self.__calculate_mole_fractions()
        self.__calculate_total_mole_fractions()
