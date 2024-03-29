import numpy as np
from scipy.interpolate import interp1d

class FullSequenceRayleighDistillation:

    def __init__(self, heavy_z, light_z, vapor_escape_fraction, system_element_mass, melt_element_mass,
                 vapor_element_mass, earth_isotope_composition, theia_ejecta_fraction, total_melt_mass,
                 total_vapor_mass, chemical_frac_factor_exponent=0.43, physical_frac_factor_exponent=0.5,
                 alpha_chem=None, alpha_phys=None):
        self.heavy_z = heavy_z
        self.light_z = light_z
        self.total_melt_mass = total_melt_mass
        self.total_vapor_mass = total_vapor_mass
        self.vapor_escape_fraction = vapor_escape_fraction / 100.0
        self.system_element_mass = system_element_mass
        self.melt_element_mass = melt_element_mass
        self.vapor_element_mass = vapor_element_mass
        self.earth_isotope_composition = earth_isotope_composition
        self.theia_ejecta_fraction = theia_ejecta_fraction / 100.0

        # pre-calculate some values
        if alpha_chem is None:
            self.chemical_fractionation_factor = (self.light_z / self.heavy_z) ** chemical_frac_factor_exponent
        else:
            self.chemical_fractionation_factor = alpha_chem
        if alpha_phys is None:
            self.physical_fractionation_factor = (self.light_z / self.heavy_z) ** physical_frac_factor_exponent
        else:
            self.physical_fractionation_factor = alpha_phys
        self.element_mass_fraction_in_melt = self.melt_element_mass / self.system_element_mass  # fraction of element in the melt (before recondensation)
        self.element_vapor_mass_escaping = self.vapor_escape_fraction * self.vapor_element_mass  # mass of element in the vapor escaping
        self.element_vapor_mass_retained = self.vapor_element_mass - self.element_vapor_mass_escaping  # mass of element in the vapor after vapor escape
        self.element_total_retained_mass = self.melt_element_mass + self.element_vapor_mass_retained  # mass of element in the system after vapor escape
        self.retained_element_mass = self.melt_element_mass + self.element_vapor_mass_retained
        # quick check for mass balance
        assert np.isclose(
            self.system_element_mass - self.retained_element_mass, self.element_vapor_mass_escaping
        )
        self.retained_element_mass_melt_fraction = self.melt_element_mass / self.retained_element_mass  # fraction of element in the melt after vapor escape, including recondensation
        self.retained_element_mass_vapor_fraction = self.element_vapor_mass_retained / self.retained_element_mass  # fraction of element in the vapor after vapor escape, including recondensation
        self.retained_element_mass_fraction = self.retained_element_mass / self.system_element_mass  # fraction of element in the system after vapor escape, including recondensation
        # bulk melt/vapor mass fractions (used for mixing model of retained melt and retained vapor)
        self.retained_vapor_mass = self.total_vapor_mass * (1 - self.vapor_escape_fraction)  # mass of vapor after vapor escape, including recondensation
        self.retained_melt_mass_fraction = self.total_melt_mass / (self.total_melt_mass + self.retained_vapor_mass)  # fraction of melt in the system after vapor escape, including recondensation
        self.retained_vapor_mass_fraction = self.retained_vapor_mass / (self.total_melt_mass + self.retained_vapor_mass)  # fraction of vapor in the system after vapor escape, including recondensation
        # print(self.retained_vapor_mass_fraction + self.retained_melt_mass_fraction, self.retained_vapor_mass_fraction, self.retained_melt_mass_fraction)

    def rayleigh_fractionate_residual(self, delta_initial, alpha, f):
        """
        Returns the isotope difference between the initial reservoir and the residual reservoir in delta notation.
        :param delta_initial:
        :param alpha:
        :param f:
        :return:
        """
        return delta_initial + ((1000 + delta_initial) * (f ** (alpha - 1) - 1))

    def rayleigh_fractionate_extract(self, delta_initial, alpha, f):
        """
        Returns the isotope difference between the initial reservoir and the extract reservoir in delta notation.
        :param delta_initial:
        :param alpha:
        :param f:
        :return:
        """
        return delta_initial + ((1000 + delta_initial) * (alpha * f ** (alpha - 1) - 1))

    def rayleigh_mixing(self, delta_1, delta_2, x_1):
        """
        Returns the mixing of two reservoirs in delta notation.
        :param delta_1:
        :param delta_2:
        :param x_1:
        :return:
        """
        return (x_1 * delta_1) + ((1 - x_1) * delta_2)

    def run_3_stage_fractionation(self, delta_initial=None):
        """
        Models the evaporation of the initial melt, the physical fractionation of the vapor, and then the
        recondensation into the melt of the retained vapor.
        :return:
        """
        if delta_initial is None:
            delta_initial = self.earth_isotope_composition
        # first, model evaporation of the initial melt
        delta_melt = self.rayleigh_fractionate_residual(delta_initial, self.chemical_fractionation_factor,
                                                        self.element_mass_fraction_in_melt)
        # model the vapor extract
        delta_bulk_vapor = self.rayleigh_fractionate_extract(delta_initial, self.chemical_fractionation_factor,
                                                             self.element_mass_fraction_in_melt)
        # next, model the physical fractionation of the vapor
        delta_retained_vapor = self.rayleigh_fractionate_residual(delta_bulk_vapor, self.physical_fractionation_factor,
                                                                  1 - self.vapor_escape_fraction)
        # model the escaping vapor
        delta_escaping_vapor = self.rayleigh_fractionate_extract(delta_bulk_vapor, self.physical_fractionation_factor,
                                                                 1 - self.vapor_escape_fraction)
        # now, mix the retained vapor with the melt
        delta_retained_melt = self.rayleigh_mixing(delta_melt, delta_retained_vapor, self.retained_element_mass_melt_fraction)
        # return everything
        return {
            'delta_initial': delta_initial,
            'delta_bulk_vapor': delta_bulk_vapor,
            'delta_retained_vapor': delta_retained_vapor,
            'delta_escaping_vapor': delta_escaping_vapor,
            'delta_melt': delta_melt,  # no recondensation
            'delta_melt_with_recondensation': delta_retained_melt,  # with recondensation
            'delta_moon_earth': delta_retained_melt - self.earth_isotope_composition,
            'delta_moon_earth_no_recondensation': delta_melt - self.earth_isotope_composition,
            'element_mass_fraction_in_melt_pre_recondensation': self.element_mass_fraction_in_melt,
            'element_mass_fraction_in_melt_post_recondensation': self.retained_element_mass_fraction,
            # 'retained element melt mass fraction': self.retained_melt_mass_fraction,
            'melt mass without recondensation': self.melt_element_mass,
            'bulk vapor mass': self.vapor_element_mass,
            'retained vapor mass': self.element_vapor_mass_retained,
            'escaping vapor mass': self.element_vapor_mass_escaping,
            'recondensed melt mass': self.retained_element_mass,
        }

    def run_theia_mass_balance(self, theia_range, delta_moon_earth):
        """
        Calculates the isotope composition of Theia by treating the ejecta as a Rayleigh mixture of Earth and Theia.
        Leaves the Theia isotope composition as an unknown and solves for it.
        :param theia_range:
        :param theia_ejecta_mass_fraction:
        :param delta_moon_earth:
        :return:
        """
        earth_ejecta_mass_fraction = 1 - self.theia_ejecta_fraction  # fraction of Earth's mass in the ejecta
        theia_search = {}  # dictionary to hold the results of the search
        for i in theia_range:
            delta_ejecta = self.rayleigh_mixing(self.earth_isotope_composition, i, earth_ejecta_mass_fraction)
            data = self.run_3_stage_fractionation(delta_initial=delta_ejecta)
            data['delta_ejecta'] = delta_ejecta
            data['delta_theia'] = i
            data['theia_ejecta_mass_fraction'] = self.theia_ejecta_fraction
            theia_search[i] = data
        # interpolate delta_theia to find the value that matches the delta_moon_earth
        # first, find the two values that bracket the delta_moon_earth
        theia_values = sorted(theia_search.keys())
        # find the two values that bracket the delta_moon_earth
        delta_earth_moon_values = [theia_search[i]['delta_moon_earth'] for i in theia_values]
        # make sure that the delta_moon_earth is within the range of the delta_earth_moon_values
        if delta_moon_earth < min(delta_earth_moon_values) or delta_moon_earth > max(delta_earth_moon_values):
            raise ValueError('delta_moon_earth is outside of the range of the delta_earth_moon_values')
        # interpolate the theia value as a function of delta_moon_earth
        theia_value = interp1d(
            delta_earth_moon_values,
            theia_values
        )(delta_moon_earth)  # the value of Theia's isotope composition that best matches the delta_moon_earth

        # rerun the calculation with the interpolated value of Theia's isotope composition
        delta_ejecta = self.rayleigh_mixing(self.earth_isotope_composition, theia_value, earth_ejecta_mass_fraction)
        data = self.run_3_stage_fractionation(delta_initial=delta_ejecta)
        data['delta_theia'] = theia_value
        data['delta_ejecta'] = delta_ejecta
        data['alpha_chemical'] = self.chemical_fractionation_factor
        data['alpha_physical'] = self.physical_fractionation_factor
        return theia_search, theia_value, data


class FullSequenceRayleighDistillation_SingleReservior(FullSequenceRayleighDistillation):

    def fractionate(self, reservoir_delta):
        """
        Returns the fractionated value of the element.
        :param reservoir_delta:  The delta-notation abundance of the isotope in equation
        :return:
        """
        return self.run_3_stage_fractionation(delta_initial=reservoir_delta)



class SaturatedRayleighFractionation:

    def __init__(self):
        pass

    def rayleigh_fractionate_residual(self, delta_initial, alpha, f, S):
        """
        Returns the isotope difference between the initial reservoir and the residual reservoir in delta notation.
        :param delta_initial:
        :param alpha:
        :param f:
        :return:
        """
        return (1 - S) * (delta_initial + ((1000 + delta_initial) * (f ** (alpha - 1) - 1)))

    def rayleigh_fractionate_extract(self, delta_initial, alpha, f, S):
        """
        Returns the isotope difference between the initial reservoir and the extract reservoir in delta notation.
        :param delta_initial:
        :param alpha:
        :param f:
        :return:
        """
        return (1 - S) * (delta_initial + ((1000 + delta_initial) * (alpha * f ** (alpha - 1) - 1)))

    def rayleigh_mixing(self, x_1, delta_1, delta_2):
        """
        Returns the mixing of two reservoirs in delta notation.
        :param delta_1:
        :param delta_2:
        :param x_1:
        :return:
        """
        return (x_1 * delta_1) + ((1 - x_1) * delta_2)

    # def fractionation(self):
    #     self.melt_delta = self.rayleigh_fractionate_residual(self.delta_initial, self.alpha_chem, self.f)
    #     self.vapor_delta = self.rayleigh_fractionate_extract(self.delta_initial, self.alpha_chem, self.f)
    #     self.physically_fractionated_retained_vapor = self.rayleigh_fractionate_residual(
    #         self.vapor_delta, self.alpha_phys, self.mass_in_vapor * (1 - self.hydrodynamic_escape_fraction)
    #     )
    #     self.physically_fractionated_escaping_vapor = self.rayleigh_fractionate_extract(
    #         self.vapor_delta, self.alpha_phys, self.mass_in_vapor * (1 - self.hydrodynamic_escape_fraction)
    #     )
    #     self.recondensed_melt_delta = self.rayleigh_mixing(
    #         self.f_melt_recondensed, self.melt_delta, self.physically_fractionated_retained_vapor
    #     )
    #     return {
    #         'melt_delta': self.melt_delta,
    #         'vapor_delta': self.vapor_delta,
    #         'physically_fractionated_retained_vapor': self.physically_fractionated_retained_vapor,
    #         'physically_fractionated_escaping_vapor': self.physically_fractionated_escaping_vapor,
    #         'recondensed_melt_delta': self.recondensed_melt_delta,
    #     }

