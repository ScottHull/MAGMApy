

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
        :return:
        """
        ATMAX = 0.0
        for i in self.composition.cation_fraction:
            if self.composition.cation_fraction[i] > 0.01:
                r = self.gas_system.total_number_density_elements[i] / self.composition.cation_fraction[i]
                if r > ATMAX:
                    ATMAX = r
        FACT = 0.05 / ATMAX  # most volatile element will be reduced by 5%

        # TODO: add FACT1 because it is used in the second part of this.

        for i in self.composition.cation_fraction:
            self.composition.cation_fraction[i] -= FACT * self.gas_system.total_number_density_elements[i]

