from math import log
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

font = {'size'   : 18}

matplotlib.rc('font', **font)

isotopes = {
    'K': {
        41: 40.961826,
        39: 38.963707
    },
    'Zn': {
        66: 65.926037,
        64: 63.929147
    },
    'Rb': {
        87: 86.909183,
        85: 84.911789
    },
}  # dictionary of isotope masses

def calculate_delta_kin(element):
    """
    Returns the kinetic fractation factor in Delta notation.
    See Nie and Dauphas 2019 Figure 2 caption.
    :param element:
    :return:
    """
    if element in ["K", "Rb"]:
        beta = 0.43
    else:
        beta = 0.5
    element_isotopes = isotopes[element]
    heavy_isotope = max(element_isotopes.keys())
    light_isotope = min(element_isotopes.keys())
    heavy_isotope_mass = element_isotopes[heavy_isotope]
    light_isotope_mass = element_isotopes[light_isotope]
    return (((light_isotope_mass / heavy_isotope_mass) ** beta) - 1) * 1000


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


fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
ax.set_xlabel("$\Delta_{kin} \ln(f)$")
ax.set_ylabel("$\delta_{Moon} - \delta_{\oplus}$ (‰)")
ax.grid(alpha=0.4)
for element in isotopes.keys():
    f_range = list(np.arange(0.001, 1.00, 0.001))
    x = [calculate_delta_kin(element) * log(f) for f in f_range]
    y = [nie_and_dauphas_rayleigh_fractionation(f, calculate_delta_kin(element)) for f in f_range]
    ax.plot(
        x, y, linewidth=2.0, label=element
    )
ax.legend()

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
ax.set_xlabel("VMF (element-specific)")
ax.set_ylabel("$\delta_{Moon} - \delta_{\oplus}$ (‰)")
ax.grid(alpha=0.4)
for element in isotopes.keys():
    f_range = list(np.arange(0.001, 1.00, 0.001))
    x = [1 - f for f in f_range]  # VMF of element i
    y = [nie_and_dauphas_rayleigh_fractionation(f, calculate_delta_kin(element)) for f in f_range]
    ax.plot(
        x, y, linewidth=2.0, label=element
    )
ax.legend()

plt.show()
