from src.composition import get_element_in_base_oxide, get_molecule_stoichiometry
import sys


class ThermoSystem:

    def __init__(self, composition, liquid_system, gas_system):
        self.composition = composition
        self.liquid_system = liquid_system
        self.gas_system = gas_system
        self.weight_fraction_vaporized = 0.0
        self.atomic_fraction_vaporized = 0.0
        self.weight_vaporized = 0.0

    def __renormalize_abundances(self):
        """
        Recalculates cation and oxide fractions following vaporization.
        Note that in the original MAGMA code, FSI relates to oxide fraction of Si and CONSI is the
        cation fraction of Si.
        :return:
        """
        # get total number of oxide molecules
        total_cations = sum(self.composition.cation_fraction.values())
        # get the total number of cations in the oxide molecules
        total_oxide_moles = 0
        for i in self.composition.cation_fraction.keys():
            base_oxide = get_element_in_base_oxide(element=i, oxides=self.composition.mole_pct_composition)
            stoich = get_molecule_stoichiometry(molecule=base_oxide, return_oxygen=False)
            for j in stoich.keys():  # should only have 1 cation in this loop
                total_oxide_moles += self.composition.cation_fraction[i] * (1.0 / stoich[j])  # i.e. 2 Al in 1 Al2O3
        # renormalize cation mole fractions
        for i in self.composition.oxide_mole_fraction.keys():
            stoich = get_molecule_stoichiometry(molecule=i, return_oxygen=False)
            for j in stoich:  # should only contain 1 cation
                self.composition.oxide_mole_fraction[i] = (self.composition.cation_fraction[j] * (1.0 / stoich[
                    j])) / total_oxide_moles
        # renormalize oxide mole fractions
        for i in self.composition.cation_fraction.keys():
            self.composition.cation_fraction[i] /= total_cations

    def __calculate_size_step(self, fraction=0.05):
        """
        Computes the size step for fractional volatilization.
        The most volatile element in the melt will be reduced by 5% (Schafer & Fegley 2009).  This is done for both the
        mole fractions of the elements AND the atomic abundances because once PLAN(element) becomes < 1e-35,
        it gets set to 0 to avoid errors due to the small values.
        The mole fractions are renormalized below and are used to compute the oxide mole fractions.
        The PLAN(element) values are used to compute the fraction vaporized.

        Note that in original MAGMA code:
            ATMSI = total mole fraction of element in the gas (i.e. not system, but GAS)
                This is stored as self.gas_system.total_mole_fraction
            CONSI = the total system relative abundances of the cations, as atom%
                This is stored as self.composition.cation_fraction
        :return:
        """
        # adjust system composition
        ATMAX = 0.0
        for i in self.gas_system.total_mole_fraction.keys():
            if self.composition.cation_fraction[i] > 1 * 10 ** -20:
                r = self.gas_system.total_mole_fraction[i] / self.composition.cation_fraction[i]
                if r > ATMAX:
                    ATMAX = r
        FACT = fraction / ATMAX  # most volatile element will be reduced by the given percentage

        # adjust system cation fractions
        for i in self.composition.cation_fraction.keys():
            self.composition.cation_fraction[i] -= FACT * self.gas_system.total_mole_fraction[i]
            if self.composition.cation_fraction[i] <= 1 * 10 ** -100:  # if the numbers approach 0, set them to 0
                self.composition.planetary_abundances[i] = 0.0
                self.composition.cation_fraction[i] = 0.0

        # adjust planetary composition
        ATMAX = 0.0
        for i in self.composition.planetary_abundances:
            if self.composition.planetary_abundances[i] > 1 * 10 ** -20:
                r = self.gas_system.total_mole_fraction[i] / self.composition.planetary_abundances[i]
                if r > ATMAX:
                    ATMAX = r
        FACT1 = fraction / ATMAX  # most volatile element will be reduced by the given percentage

        # adjust planetary composition
        for i in self.composition.planetary_abundances.keys():
            self.composition.planetary_abundances[i] -= FACT1 * self.gas_system.total_mole_fraction[i]
            if self.composition.planetary_abundances[i] <= 1 * 10 ** -100:  # if the numbers approach 0, set them to 0
                self.composition.planetary_abundances[i] = 0.0
                self.composition.cation_fraction[i] = 0.0

    def vaporize(self):
        self.__calculate_size_step()  # calculate volatility for fractional volatilization
        # calculate fraction of vaporized materials
        total_planetary_cations = sum(
            self.composition.planetary_abundances.values())  # sum of fractional cation abundances, PLAMANT
        self.composition.planetary_cation_ratio = total_planetary_cations / self.composition.initial_planetary_cations  # PLANRAT
        self.atomic_fraction_vaporized = 1.0 - self.composition.planetary_cation_ratio  # VAP

        # calculate weight% vaporized each element
        wt_vaporized = 0.0
        for i in self.composition.planetary_abundances.keys():
            # get the base oxide for the elment (i.e. SiO2 for Si)
            base_oxide = get_element_in_base_oxide(element=i, oxides=self.composition.mole_pct_composition)
            oxide_mw = self.composition.get_molecule_mass(molecule=base_oxide)  # get molecular weight of oxide
            oxide_stoich = get_molecule_stoichiometry(molecule=base_oxide)
            # convert moles to mass
            wt_vaporized += self.composition.planetary_abundances[i] * oxide_mw * (1.0 / oxide_stoich[i])
        self.weight_fraction_vaporized = (self.liquid_system.initial_melt_mass - wt_vaporized) / \
                                self.liquid_system.initial_melt_mass
        self.weight_vaporized = self.liquid_system.initial_melt_mass - wt_vaporized

        # renormalize abundances
        self.__renormalize_abundances()

        return self.weight_fraction_vaporized

    def __calculate_size_step_thermal(self, fraction=0.05):
        """
        Computes the size step for fractional volatilization.
        The most volatile element in the melt will be reduced by 5% (Schafer & Fegley 2009).  This is done for both the
        mole fractions of the elements AND the atomic abundances because once PLAN(element) becomes < 1e-35,
        it gets set to 0 to avoid errors due to the small values.
        The mole fractions are renormalized below and are used to compute the oxide mole fractions.
        The PLAN(element) values are used to compute the fraction vaporized.

        Note that in original MAGMA code:
            ATMSI = total mole fraction of element in the gas (i.e. not system, but GAS)
                This is stored as self.gas_system.total_mole_fraction
            CONSI = the total system relative abundances of the cations, as atom%
                This is stored as self.composition.cation_fraction
        :return:
        """
        # adjust planetary composition
        for i in self.composition.planetary_abundances.keys():
            partial_pressure_element = self.gas_system.number_densities_elements[i] * 8.314 * self.liquid_system.temperature
            self.composition.planetary_abundances[i] -= (self.composition.planetary_abundances[i] / partial_pressure_element) * self.gas_system.total_pressure
            if self.composition.planetary_abundances[i] <= 1 * 10 ** -100:  # if the numbers approach 0, set them to 0
                self.composition.planetary_abundances[i] = 0.0
                self.composition.cation_fraction[i] = 0.0

    def vaporize_thermal(self):
        # self.__calculate_size_step_thermal()  # calculate volatility for fractional volatilization
        # calculate fraction of vaporized materials
        total_planetary_cations = sum(
            self.composition.planetary_abundances.values())  # sum of fractional cation abundances, PLAMANT
        self.composition.planetary_cation_ratio = total_planetary_cations / self.composition.initial_planetary_cations  # PLANRAT
        self.atomic_fraction_vaporized = 1.0 - self.composition.planetary_cation_ratio  # VAP

        # calculate weight% vaporized each element
        wt_vaporized = 0.0
        for i in self.composition.planetary_abundances.keys():
            # get the base oxide for the elment (i.e. SiO2 for Si)
            base_oxide = get_element_in_base_oxide(element=i, oxides=self.composition.mole_pct_composition)
            oxide_mw = self.composition.get_molecule_mass(molecule=base_oxide)  # get molecular weight of oxide
            oxide_stoich = get_molecule_stoichiometry(molecule=base_oxide)
            wt_vaporized += self.composition.planetary_abundances[i] * oxide_mw * (1.0 / oxide_stoich[i])
        self.weight_fraction_vaporized = (self.liquid_system.initial_melt_mass - wt_vaporized) / \
                                self.liquid_system.initial_melt_mass
        self.weight_vaporized = self.liquid_system.initial_melt_mass - wt_vaporized

        # renormalize abundances
        self.__renormalize_abundances()

        return self.weight_fraction_vaporized
