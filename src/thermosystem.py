from composition import get_element_in_base_oxide, get_molecule_stoichiometry

class ThermoSystem:
    def __init__(self, composition, liquid_system, gas_system):
        self.composition = composition
        self.liquid_system = liquid_system
        self.gas_system = gas_system

    def __calculate_size_step(self):
        """
        Computes the size step.
        The most volatile element in the melt will be reduced by 5%.  This is done for both the mole fractions of the
        elements AND the atomic abundances because once PLAN(element) becomes < 1e-35, it gets set to 0 to avoid
        errors due to the small values.
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
        for i in self.gas_system.total_mole_fraction:
            if self.composition.cation_fraction[i] > 0.01:
                r = self.gas_system.total_mole_fraction[i] / self.composition.cation_fraction[i]
                if r > ATMAX:
                    ATMAX = r
        FACT = 0.05 / ATMAX  # most volatile element will be reduced by 5%

        # adjust system composition
        for i in self.composition.cation_fraction:
            self.composition.cation_fraction[i] -= FACT * self.gas_system.total_number_density_elements[i]

        # TODO: is this necessary?
        # adjust planetary composition
        ATMAX = 0.0
        for i in self.composition.planetary_abundances:
            if self.composition.planetary_abundances[i] > 0.01:
                r = self.gas_system.total_number_density_elements[i] / self.composition.planetary_abundances[i]
                if r > ATMAX:
                    ATMAX = r
        FACT1 = 0.05 / ATMAX  # most volatile element will be reduced by 5%

        # adjust planetary composition
        for i in self.composition.planetary_abundances.keys():
            old_abun = self.composition.planetary_abundances[i]
            if old_abun != 0.0:  # if the previous abundance is 0, then no need to adjust it
                self.composition.planetary_abundances[i] -= FACT1 * self.gas_system.total_number_density_elements[i]
                if self.composition.planetary_abundances[i] <= 0.0:
                    self.composition.planetary_abundances[i] = 0.0
                    self.composition.cation_fraction[i] = 0.0

    def vaporize(self):
        self.__calculate_size_step()  # calculate volatility
        # calculate fraction of vaporized materials
        total_planetary_cations = sum(self.composition.planetary_abundances.values())  # sum of fractional cation abundances
        self.composition.planetary_cation_ratio = total_planetary_cations / self.composition.initial_planetary_cations
        self.vaporized_magma_fraction = 1.0 - self.composition.planetary_cation_ratio

        # calculate weight% vaporized each element
        wt_vaporized = 0.0
        for i in self.composition.planetary_abundances.keys():
            # get the base oxide for the elment (i.e. SiO2 for Si)
            base_oxide = get_element_in_base_oxide(element=i, oxides=self.composition.mole_pct_composition)
            oxide_mw = self.composition.get_molecule_mass(molecule=base_oxide)  # get molecular weight of oxide
            oxide_stoich = get_molecule_stoichiometry(molecule=oxide_mw)
            wt_vaporized += self.composition.planetary_abundances[i] * oxide_mw * (1.0 / oxide_stoich[i])

