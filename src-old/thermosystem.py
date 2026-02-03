import copy

from src.composition import get_element_in_base_oxide, get_molecule_stoichiometry
import sys


class ThermoSystem:

    def __init__(self, composition, liquid_system, gas_system):
        self.composition = composition
        self.liquid_system = liquid_system
        self.gas_system = gas_system
        self.weight_fraction_vaporized = 0.0
        self.atomic_fraction_vaporized = 0.0
        self.weight_vaporized = 0.0  # mass of the vaporized system
        self.melt_mass = 0.0  # mass of the melt
        self.most_volatile_species = None  # most volatile species, which is fractionally depleted first

    def renormalize_abundances(self):
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
        # get the melt mass from the previous step
        self.previous_melt_mass = copy.copy(self.liquid_system.melt_mass)
        if self.previous_melt_mass <= 1e-98:
            self.previous_melt_mass = self.liquid_system.initial_melt_mass
        # get the absolute mass of the melt
        self.liquid_system.melt_mass = self.liquid_system.initial_melt_mass - self.weight_vaporized
        # get liquid cation mass %
        self.liquid_system.get_cation_fraction_from_moles()
        # convert liquid moles to oxides and get oxide mass %
        self.liquid_system.get_liquid_oxide_mass_fraction()
        # get vapor cation mass %
        self.gas_system.get_cation_mass_fraction_from_moles(vapor_mass=self.weight_vaporized)
        # calculate f
        self.gas_system.get_f()
        # calculate the total vapor mass of all vapor produced
        self.gas_system.get_vapor_mass(initial_liquid_mass=self.liquid_system.initial_melt_mass,
                                       liquid_mass_at_time=self.liquid_system.melt_mass,
                                       previous_melt_mass=self.previous_melt_mass)

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
                # ratio of the element atomic fraction in the gas relative to the entire system (gas + liquid)
                r = self.gas_system.total_mole_fraction[i] / self.composition.cation_fraction[i]
                if r > ATMAX:
                    ATMAX = r
        FACT = fraction / ATMAX  # most volatile element will be reduced by the given percentage

        # adjust system cation fractions
        for i in self.composition.cation_fraction.keys():
            # system loses cations via fractional volatilization
            self.composition.cation_fraction[i] -= FACT * self.gas_system.total_mole_fraction[i]
            if self.composition.cation_fraction[i] <= 1 * 10 ** -100:  # if the numbers approach 0, set them to 0
                self.composition.liquid_abundances[i] = 0.0
                self.composition.cation_fraction[i] = 0.0

        # adjust liquid composition
        ATMAX = 0.0
        for i in self.composition.liquid_abundances:
            if self.composition.liquid_abundances[i] > 1 * 10 ** -20:
                # cation fraction in the gas relative to the liquid
                r = self.gas_system.total_mole_fraction[i] / self.composition.liquid_abundances[i]
                if r > ATMAX:
                    ATMAX = r
                    self.most_volatile_species = i
        FACT1 = fraction / ATMAX  # most volatile element will be reduced by the given percentage

        # adjust liquid composition
        for i in self.composition.liquid_abundances.keys():
            # liquid loses cations via fractional volatilization
            self.composition.liquid_abundances[i] -= FACT1 * self.gas_system.total_mole_fraction[i]
            if self.composition.liquid_abundances[i] <= 1 * 10 ** -100:  # if the numbers approach 0, set them to 0
                self.composition.liquid_abundances[i] = 0.0
                self.composition.cation_fraction[i] = 0.0

    def vaporize(self, fraction=0.05):
        self.__calculate_size_step(fraction=fraction)  # calculate volatility for fractional volatilization
        # calculate fraction of vaporized materials
        total_liquid_cations = sum(
            self.composition.liquid_abundances.values())  # sum of fractional cation abundances, PLAMANT
        self.composition.liquid_cation_ratio = total_liquid_cations / self.composition.initial_liquid_cations  # PLANRAT
        self.atomic_fraction_vaporized = 1.0 - self.composition.liquid_cation_ratio  # VAP

        # calculate weight% vaporized each element
        wt_vaporized = 0.0
        for i in self.composition.liquid_abundances.keys():
            # get the base oxide for the elment (i.e. SiO2 for Si)
            base_oxide = get_element_in_base_oxide(element=i, oxides=self.composition.mole_pct_composition)
            oxide_mw = self.composition.get_molecule_mass(molecule=base_oxide)  # get molecular weight of oxide
            oxide_stoich = get_molecule_stoichiometry(molecule=base_oxide)
            # convert moles to mass
            wt_vaporized += self.composition.liquid_abundances[i] * oxide_mw * (1.0 / oxide_stoich[i])
        self.weight_fraction_vaporized = (self.liquid_system.initial_melt_mass - wt_vaporized) / \
                                         self.liquid_system.initial_melt_mass
        self.weight_vaporized = self.liquid_system.initial_melt_mass - wt_vaporized

        # renormalize abundances
        self.renormalize_abundances()

        return self.weight_fraction_vaporized

    # def vaporize_thermal(self):
    #     # calculate fraction of vaporized materials
    #     total_liquid_cations = sum(
    #         self.composition.liquid_abundances.values())  # sum of fractional cation abundances, PLAMANT
    #     self.composition.liquid_cation_ratio = total_liquid_cations / self.composition.initial_liquid_cations  # PLANRAT
    #     self.atomic_fraction_vaporized = 1.0 - self.composition.liquid_cation_ratio  # VAP
    #
    #     # calculate weight% vaporized each element
    #     wt_vaporized = 0.0
    #     for i in self.composition.liquid_abundances.keys():
    #         # get the base oxide for the elment (i.e. SiO2 for Si)
    #         base_oxide = get_element_in_base_oxide(element=i, oxides=self.composition.mole_pct_composition)
    #         oxide_mw = self.composition.get_molecule_mass(molecule=base_oxide)  # get molecular weight of oxide
    #         oxide_stoich = get_molecule_stoichiometry(molecule=base_oxide)
    #         wt_vaporized += self.composition.liquid_abundances[i] * oxide_mw * (1.0 / oxide_stoich[i])
    #
    #     # renormalize abundances
    #     self.renormalize_abundances()
    #
    #     return self.weight_fraction_vaporized



class EquilibriumThermoSystem(ThermoSystem):

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
        # adjust liquid composition
        ATMAX = 0.0
        for i in self.composition.liquid_abundances:
            if self.composition.liquid_abundances[i] > 1 * 10 ** -20:
                # cation fraction in the gas relative to the liquid
                r = self.gas_system.total_mole_fraction[i] / self.composition.liquid_abundances[i]
                if r > ATMAX:
                    ATMAX = r
                    self.most_volatile_species = i
        FACT1 = fraction / ATMAX  # most volatile element will be reduced by the given percentage

        # adjust liquid composition
        for i in self.composition.liquid_abundances.keys():
            # liquid loses cations via fractional volatilization
            self.composition.liquid_abundances[i] -= FACT1 * self.gas_system.total_mole_fraction[i]
            if self.composition.liquid_abundances[i] <= 1 * 10 ** -100:  # if the numbers approach 0, set them to 0
                self.composition.liquid_abundances[i] = 0.0
                self.composition.cation_fraction[i] = 0.0

    def vaporize(self):
        self.__calculate_size_step()  # calculate volatility for fractional volatilization
        # calculate fraction of vaporized materials
        total_liquid_cations = sum(
            self.composition.liquid_abundances.values())  # sum of fractional cation abundances, PLAMANT
        self.composition.liquid_cation_ratio = total_liquid_cations / self.composition.initial_liquid_cations  # PLANRAT
        self.atomic_fraction_vaporized = 1.0 - self.composition.liquid_cation_ratio  # VAP

        # calculate weight% vaporized each element
        wt_vaporized = 0.0
        for i in self.composition.liquid_abundances.keys():
            # get the base oxide for the elment (i.e. SiO2 for Si)
            base_oxide = get_element_in_base_oxide(element=i, oxides=self.composition.mole_pct_composition)
            oxide_mw = self.composition.get_molecule_mass(molecule=base_oxide)  # get molecular weight of oxide
            oxide_stoich = get_molecule_stoichiometry(molecule=base_oxide)
            wt_vaporized += self.composition.liquid_abundances[i] * oxide_mw * (1.0 / oxide_stoich[i])
        self.weight_fraction_vaporized = (self.liquid_system.initial_melt_mass - wt_vaporized) / \
                                         self.liquid_system.initial_melt_mass
        self.weight_vaporized = self.liquid_system.initial_melt_mass - wt_vaporized
        self.gas_system.vapor_mass = self.weight_vaporized
        self.gas_system.vapor_mass_fraction = self.weight_fraction_vaporized

        # renormalize abundances
        self.renormalize_abundances()

        return self.weight_fraction_vaporized
