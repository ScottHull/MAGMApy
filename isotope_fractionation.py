from src.plots import collect_data
# from .vaporize_to_fraction_runs import runs

import pandas as pd
import numpy as np
from math import log, sqrt
from copy import copy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

"""
Notes: may be best to run MAGMA to sufficient VMF and then interpolate desired VMF.
"""

runs = {
    # b = 0.73
    '5b073S': {
        "temperature": 6345.79,  # K
        "vmf": 35.73,  # %
    },
}

isotopes = {
    'K': {
        41: 40.961826,
        39: 38.963707
    },
    # 'Zn': {
    #     66: 65.926037,
    #     64: 63.929147
    # }
}


def get_closest_header_output(data, target):
    """
    Takes the difference between the target and the data key.  Returns the closest key.
    Used for interpolation of target VMF.
    :param data:
    :param target:
    :return:
    """
    min_diff = 10 ** 10
    closest_header = None
    for i in data.keys():
        diff = abs((i * 100) - target)
        if diff < min_diff:
            min_diff = diff
            closest_header = i
    return closest_header


def get_2_closest_headers_output(data, target):
    """
    Takes the difference between the target and the data key.  Returns the closest key and the second closest key.
    Used for interpolation of target VMF.
    :param data:
    :param target:
    :return:
    """
    closest_header = get_closest_header_output(data, target)
    data_copy = copy(data)
    del data_copy[closest_header]
    return closest_header, get_closest_header_output(data_copy, target)


def return_vmf_and_element_lists(data):
    """
    Returns a list of VMF values and a dictionary of lists of element values.
    :param data:
    :return:
    """
    vmf_list = list(sorted(data.keys()))
    elements_at_vmf = {element: [data[vmf][element] for vmf in vmf_list] for element in data[vmf_list[0]].keys()}
    return vmf_list, elements_at_vmf


def renormalize_interpolated_elements(elements):
    """
    Normalizes the values of the elements dictionary to 1.
    :param elements:
    :return:
    """
    total = sum(elements.values())
    for element in elements.keys():
        elements[element] = elements[element] / total
    return elements


def interpolate_elements_at_vmf(vmf_list, elements, target_vmf):
    """
    Takes 2 VMF keys and uses the data dictionary to interplate all value elements.
    :param closet_vmf_1:
    :param cloest_vmf_2:
    :param data:
    :return:
    """
    interpolated_elements = {}
    for element in elements.keys():
        interp = interp1d(vmf_list, elements[element])
        interpolated_elements[element] = interp(target_vmf)
    return interpolated_elements


def get_element_abundances_as_function_of_vmf(data):
    """
    Uses the VMF keys in the data dictionary and the elements in the embedded dictionaries to build lists of element
    evolution as a function of VMF.
    :param data:
    :return:
    """
    vmfs = list(sorted(data.keys()))
    elements = data[vmfs[0]].keys()
    return vmfs, {element: [data[vmf][element] for vmf in vmfs] for element in elements}


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


for run in runs.keys():
    data = collect_data(path="{}_reports/magma_cation_mass_fraction".format(run), x_header='mass fraction vaporized')
    vmf_list, elements = return_vmf_and_element_lists(data)
    interpolated_elements = interpolate_elements_at_vmf(vmf_list, elements, runs[run]["vmf"] / 100)

    figure = plt.figure(figsize=(16, 9))
    ax = figure.add_subplot(111)
    vmf_evolution, element_evolution = get_element_abundances_as_function_of_vmf(data)
    color_cycle = cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax.axvline(x=runs[run]["vmf"], color='red', linewidth=2, linestyle="dotted",
               label="SPH VMF: {}%".format(runs[run]["vmf"]))
    for index, element in enumerate(element_evolution.keys()):
        ax.plot(np.array(vmf_evolution) * 100, np.array(element_evolution[element]) * 100, color=color_cycle[index],
                linewidth=2, label=element)
        ax.scatter(
            runs[run]["vmf"], interpolated_elements[element] * 100, color=color_cycle[index], marker='x', s=100
        )

    ax.set_xlabel("VMF (%)")
    ax.set_ylabel("Liquid Element Fraction (%)")
    ax.set_title("{} MAGMA Liquid Evolution".format(run))
    ax.grid(alpha=0.4)
    ax.legend()

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    for index, element in enumerate(element_evolution.keys()):
        if element in ["K"]:
            # plot the evolution of f
            f_range = list(np.arange(0.001, 1.00, 0.001))
            x = [1 - f for f in f_range]  # VMF of element i
            y = [nie_and_dauphas_rayleigh_fractionation(f, calculate_delta_kin(element)) for f in f_range]
            ax.plot(
                x, y, linewidth=2.0, label=element
            )
            # now, plot where MAGMA is predicting at the current vmf
            delta_kin = calculate_delta_kin(element)
            delta_moon_vs_delta_earth = nie_and_dauphas_rayleigh_fractionation(interpolated_elements[element], delta_kin)
            print("{}, {}, {}".format(element, delta_kin, delta_moon_vs_delta_earth))
            ax.scatter(
                delta_kin * log(interpolated_elements[element]),
                nie_and_dauphas_rayleigh_fractionation(interpolated_elements[element], delta_kin),
                color=color_cycle[index], marker='x', s=200, label=element
            )

    ax.set_xlabel(r"$\Delta_{kin} ln(f)$")
    ax.set_ylabel(r"$delta_{Moon} - \delta_{\oplus}$")
    ax.set_title("{} MVE Isotope Fractionation".format(run))
    ax.grid(alpha=0.4)
    ax.legend()

    plt.show()
