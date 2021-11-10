from math import log10

def get_oxygen_fugacity(gas_system):
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
    return log10(pp_O2)
