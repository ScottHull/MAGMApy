


def calculate_delta_kin(element, beta=0.5):
    """
    Returns the kinetic fractation factor in Delta notation.
    See Nie and Dauphas 2019 Figure 2 caption.
    :param element:
    :return:
    """
    if element in ["K", "Rb"]:
        beta = 0.43
    element_isotopes = isotopes[element]
    heavy_isotope = max(element_isotopes.keys())
    light_isotope = min(element_isotopes.keys())
    heavy_isotope_mass = element_isotopes[heavy_isotope]
    light_isotope_mass = element_isotopes[light_isotope]
    return (((heavy_isotope_mass / light_isotope_mass) ** beta) - 1) * 1000


def nie_and_dauphas_rayleigh_fractionation(f, delta_kin, S=0.989, delta_eq=0.0):
    """
    Returns the isotope difference between two reservoirs in delta notation.
    :param f:
    :param delta_kin:
    :param S:
    :param delta_eq:
    :return:
    """
    return (delta_eq + (1 - S) * delta_kin) * log(f)
