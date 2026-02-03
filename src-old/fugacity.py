from math import log


def get_oxygen_fugacity(gas_system):
    # TODO: this might have to be used for the liquid system instead of the gas
    """
    From Visscher and Fegley 2013.
    fO2 = gamma_O2 * P_O2 = gamma_O2 * X_O2 * P_tot
    where gamma_O2 is the fugacity coefficient of O2, X_O2 is the mole fraction abundance.
    Assume that  gamma_O2 = 1 under P/T range considered.
    So, fO2 =approx= P_O2 =approx= X_O2 * P_tot
    :param gas_system:
    :return:
    """
    pp_O2 = gas_system.partial_pressures["O2"]
    return log(pp_O2)


class fO2_Buffer:
    """
    Models the fO2 buffer from Frost et al. 1986.
    Temperature range: 565 - 1200 C
    """

    def __buffer_equation(self, A, B, C, temperature, pressure):
        """
        Returns ln(fO2).
        Requries T in K and P in bars (1 bar = 100000 Pa).
        :param A:
        :param B:
        :param C:
        :param temperature:
        :return:
        """
        return (A / temperature) + B + (C * (pressure - 1) / temperature)

    def C_to_K(self, C):
        """
        Converts C to K.
        :param C:
        :return:
        """
        return C + 273.15

    def iron_wustite(self, temperature, pressure):
        """
        Iron-Wustite Buffer
        2FeO = 2Fe + O2
        :param temperature:
        :return:
        """
        A = -27489
        B = 6.702
        C = 0.055
        return self.__buffer_equation(A, B, C, temperature, pressure)

    def quartz_fayalite_magnetite(self, temperature, pressure):
        """
        Quartz-Fayalite-Magnetite Buffer for alpha quartz
        3SiO3 + 2Fe3O4 = 3Fe2SiO4 + O2
        :param temperature:
        :param pressure:
        :return:
        """
        if temperature <= self.C_to_K(573):
            A = -26455.3
            B = 10.344
            C = 0.092
        else:
            A = -25096.3
            B = 8.735
            C = 0.110
        return self.__buffer_equation(A, B, C, temperature, pressure)

    def get_fO2_buffer(self, temperature, pressure, logfO2, buffer_name="QFM"):
        """
        Returns fO2.
        :param temperature:
        :param pressure:
        :return:
        """
        if buffer_name == "QFM":
            return logfO2 - self.quartz_fayalite_magnetite(temperature, pressure)
        elif buffer_name == "IW":
            return logfO2 - self.iron_wustite(temperature, pressure)
