import os
import shutil

from src.composition import Composition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data

import pandas as pd
import numpy as np
from math import log, sqrt
from copy import copy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

isotherms = [2000, 5000, 8000, 10000]  # deg K

# BSE composition oxide wt%, Visccher & Fegley 2013
composition = {
    "SiO2": 45.40,
    'MgO': 36.76,
    'Al2O3': 4.48,
    'TiO2': 0.21,
    'Fe2O3': 0.00000,
    'FeO': 8.10,
    'CaO': 3.65,
    'Na2O': 0.349,
    'K2O': 0.031,
    'ZnO': 6.7e-3,
}

major_gas_species = [
    "SiO", "O2", "MgO", "Fe", "Ca", "Al", "Ti", "Na", "K", "ZnO", "Zn"
]


def run_isotherms(isotherms: list, to_path="isotherms", to_vmf=90):
    """
    Run the MAGMApy code along the given list of isotherms.
    :return:
    """

    if not os.path.exists(to_path):
        os.mkdir(to_path)
    else:
        shutil.rmtree(to_path)
        os.mkdir(to_path)
    for temperature in isotherms:
        print(f"Running isotherm at {temperature} K")
        c = Composition(
            composition=composition
        )

        g = GasPressure(
            composition=c,
            major_gas_species=major_gas_species,
            minor_gas_species="__all__",
        )

        l = LiquidActivity(
            composition=c,
            complex_species="__all__",
            gas_system=g
        )

        t = ThermoSystem(composition=c, gas_system=g, liquid_system=l)

        reports = Report(composition=c, liquid_system=l, gas_system=g, thermosystem=t,
                         to_dir="{}/{}_reports".format(to_path, temperature))
        count = 1
        while t.weight_fraction_vaporized * 100 < to_vmf:
            output_interval = 100
            if t.weight_fraction_vaporized * 100.0 > 5:  # vmf changes very fast towards end of simulation
                output_interval = 5
            if 80 < t.weight_fraction_vaporized:
                output_interval = 50
            l.calculate_activities(temperature=temperature)
            g.calculate_pressures(temperature=temperature, liquid_system=l)
            if l.counter == 1:
                l.calculate_activities(temperature=temperature)
                g.calculate_pressures(temperature=temperature, liquid_system=l)
            fraction = 0.05  # fraction of most volatile element to lose
            t.vaporize(fraction=fraction)
            l.counter = 0  # reset Fe2O3 counter for next vaporization step
            print("[~] At iteration: {} (Weight Fraction Vaporized: {} %)".format(count,
                                                                                  round(
                                                                                      t.weight_fraction_vaporized * 100.0,
                                                                                      4)))
            if count % output_interval == 0 or count == 1:
                reports.create_composition_report(iteration=count)
                reports.create_liquid_report(iteration=count)
                reports.create_gas_report(iteration=count)
            count += 1


# run_isotherms(isotherms=isotherms, to_path="isotherms", to_vmf=90)

def return_vmf_and_element_lists(data):
    """
    Returns a list of VMF values and a dictionary of lists of element values.
    :param data:
    :return:
    """
    vmf_list = list(sorted(data.keys()))
    elements_at_vmf = {element: [data[vmf][element] for vmf in vmf_list] for element in data[vmf_list[0]].keys()}
    return np.array(vmf_list), elements_at_vmf

def return_vmf_and_species_lists(data, species: str):
    """
    Returns a list of VMF values and a dictionary of lists of element values.
    :param data:
    :return:
    """
    vmf_list = list(sorted(data.keys()))
    elements_at_vmf = {element: [data[vmf][element] for vmf in vmf_list] for element in data[vmf_list[0]].keys() if species.title() in element}
    return np.array(vmf_list), elements_at_vmf


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 9), sharex="all")
axs = axs.flatten()
# fig.tight_layout()
fig.supxlabel("VMF (%)")
axs[0].set_ylabel("Element Wt %")
axs[0].set_title("Liquid")
axs[1].set_title("Vapor")
for ax in axs:
    ax.grid(alpha=0.4)
for index, temperature in enumerate(isotherms):
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    magma_data = collect_data(path="isotherms/{}_reports/magma_cation_mass_fraction".format(temperature),
                              x_header='mass fraction vaporized')
    vapor_data = collect_data(path="isotherms/{}_reports/atmosphere_cation_mass_fraction".format(temperature),
                              x_header='mass fraction vaporized')
    magma_vmf_list, magma_elements_at_vmf = return_vmf_and_element_lists(magma_data)
    vapor_vmf_list, vapor_elements_at_vmf = return_vmf_and_element_lists(vapor_data)
    axs[0].plot(
        magma_vmf_list * 100, np.array(magma_elements_at_vmf["K"]) * 100, color=color_cycle[index], linestyle="solid",
        linewidth=2.0,
        label="{} K".format(temperature)
    )
    axs[1].plot(
        vapor_vmf_list * 100, np.array(vapor_elements_at_vmf["K"]) * 100, color=color_cycle[index], linestyle="solid",
        linewidth=2.0,
        label="{} K".format(temperature)
    )
axs[1].legend()

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 9), sharex="all")
axs = axs.flatten()
fig.tight_layout()
fig.supxlabel("VMF (%)")
axs[0].set_ylabel("Molar Abundance (relative to $10^6$ Si)")
axs[0].set_title("Liquid")
axs[1].set_title("Vapor")
for ax in axs:
    ax.grid(alpha=0.4)
for index, temperature in enumerate(isotherms):
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    magma_data = collect_data(path="isotherms/{}_reports/magma_composition".format(temperature),
                              x_header='mass fraction vaporized')
    vapor_data = collect_data(path="isotherms/{}_reports/atmosphere_cation_moles".format(temperature),
                              x_header='mass fraction vaporized')
    magma_vmf_list, magma_elements_at_vmf = return_vmf_and_element_lists(magma_data)
    vapor_vmf_list, vapor_elements_at_vmf = return_vmf_and_element_lists(vapor_data)
    axs[0].plot(
        magma_vmf_list * 100, np.array(magma_elements_at_vmf["K"]), color=color_cycle[index], linestyle="solid",
        linewidth=2.0,
        label="{} K".format(temperature)
    )
    axs[1].plot(
        vapor_vmf_list * 100, np.array(vapor_elements_at_vmf["K"]), color=color_cycle[index], linestyle="solid",
        linewidth=2.0,
        label="{} K".format(temperature)
    )
axs[1].legend()

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(9, 16), sharex="all")
axs = axs.flatten()
fig.tight_layout()
for ax in axs:
    ax.grid(alpha=0.4)

plot_index = 0
for index, temperature in enumerate(isotherms):
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    magma_data = collect_data(path="isotherms/{}_reports/activities".format(temperature),
                              x_header='mass fraction vaporized')
    vapor_data = collect_data(path="isotherms/{}_reports/partial_pressures".format(temperature),
                              x_header='mass fraction vaporized')
    magma_vmf_list, magma_elements_at_vmf = return_vmf_and_species_lists(magma_data, species="K")
    vapor_vmf_list, vapor_elements_at_vmf = return_vmf_and_species_lists(vapor_data, species="K")
    for i in magma_elements_at_vmf.keys():
        axs[plot_index].plot(
            magma_vmf_list * 100, np.array(magma_elements_at_vmf[i]), linestyle="solid",
            linewidth=2.0,
            label=i
        )
    for i in vapor_elements_at_vmf:
        axs[plot_index + 1].plot(
            vapor_vmf_list * 100, np.array(vapor_elements_at_vmf[i]), linestyle="solid",
            linewidth=2.0,
            label=i
        )
    axs[plot_index].set_title("{} K - Liquid".format(temperature))
    axs[plot_index + 1].set_title("{} K - Vapor".format(temperature))
    axs[plot_index].set_ylabel("Activity")
    axs[plot_index + 1].set_ylabel("Partial Pressure")
    plot_index += 2
axs[-1].set_xlabel("VMF (%)")
axs[-2].set_xlabel("VMF (%)")
axs[-1].legend(loc='upper left')
axs[-2].legend(loc='upper left')

plt.show()
