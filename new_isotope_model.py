import copy

from monte_carlo.monte_carlo import run_monte_carlo_vapor_loss
from theia.theia import get_theia_composition, recondense_vapor
from theia.chondrites import plot_chondrites, get_enstatite_bulk_theia_core_si_pct
from monte_carlo.monte_carlo import theia_mixing, run_full_MAGMApy
from src.plots import collect_data, collect_metadata
from src.composition import normalize, get_molecular_mass, ConvertComposition
from isotopes.rayleigh import SaturatedRayleighFractionation

import os
import re
import sys
import string
from scipy.interpolate import interp1d
from random import uniform, shuffle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor, as_completed
import labellines

# use colorblind-friendly colors
plt.style.use('seaborn-colorblind')

runs = [
    {
        "run_name": "Canonical Model",
        "temperature": 2682.61,  # K
        "vmf": 0.96,  # %
        "disk_theia_mass_fraction": 66.78,  # %
        "disk_mass": 1.02,  # lunar masses
        "vapor_loss_fraction": 74.0,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    },
    {
        "run_name": "Half-Earths Model",
        "temperature": 3517.83,  # K
        "vmf": 4.17,  # %
        "disk_theia_mass_fraction": 51.97,  # %
        "disk_mass": 1.70,  # lunar masses
        "vapor_loss_fraction": 16.0,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    }
]

def get_atom_from_molecule(molecule):
    """
    Given a molecule (e.g. SiO2), return each atom (e.g, Si, O)
    :param molecule:
    :return:
    """
    atoms = re.findall(r'([A-Z][a-z]*)(\d*)', molecule)
    return [
        atom[0] for atom in atoms
    ]
def get_cations_from_species(data):
    """
    Given a dictionary whose keys contain all molecular specues, get all unique atoms and return a list.
    :param data:
    :return:
    """
    cations = []
    for key in data.keys():
        for atom in get_atom_from_molecule(key):
            if atom not in cations:
                cations.append(atom)
    return cations

def get_cation_partial_pressure(data):
    species = get_cations_from_species(data)
    partial_pressures = {s: 0 for s in species}
    for key in data.keys():
        for atom in get_atom_from_molecule(key):
            partial_pressures[atom] += data[key]
    return partial_pressures

def format_species_string(species):
    """
    Splits by _ and converts all numbers to subscripts.
    :param species:
    :return:
    """
    formatted = species.split("_")[0]
    return rf"$\rm {formatted.replace('2', '_{2}').replace('3', '_{3}')}$"

for run in runs:
    # collect the partial pressures
    run["partial_pressures"] = collect_data(path=f"{run['run_name']}/partial_pressures", x_header='mass fraction vaporized')
    run['atomic_partial_pressures'] = {
        key: get_cation_partial_pressure(data) for key, data in run["partial_pressures"].items()
    }
    run['total_pressure'] = {
        key: sum(data.values()) for key, data in run['atomic_partial_pressures'].items()
    }
    # check to make sure that the atomic and molecular total pressures are the same
    check_total_pressures = {
        key: sum(data.values()) for key, data in run['partial_pressures'].items()
    }
    for key, data in run['partial_pressures'].items():
        assert sum(data.values()) == check_total_pressures[key]  # if it passes this check, total pressure is OK
    # collect the melt and vapor atomic mass compositions
    run["melt_composition"] = collect_data(path=f"{run['run_name']}/magma_element_mass", x_header='mass fraction vaporized')
    run["fractional_vapor_composition"] = {}  # assume that the difference between the melt at the current step minus the melt at the previous step is the vapor composition
    for index, (key, data) in enumerate(run["melt_composition"].items()):
        elements = list(data.keys())
        if index == 0:
            run["fractional_vapor_composition"][key] = {
                element: 0 for element in elements
            }
        else:
            run["fractional_vapor_composition"][key] = {
                element: data[element] - run["melt_composition"][list(run["melt_composition"].keys())[index - 1]][element]
                for element in elements
            }


# plot the partial pressures for each run in a 2 row 2 column plot
fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex='all')
ax = ax.flatten()
plot_index = 0
for index, run in enumerate(runs):
    molecular_species = run["partial_pressures"][list(run["partial_pressures"].keys())[0]].keys()
    atomic_species = run["atomic_partial_pressures"][list(run["atomic_partial_pressures"].keys())[0]].keys()
    # plot molecular partial pressures
    for species in molecular_species:
        ax[plot_index].plot(np.array(list(run['partial_pressures'].keys())) * 100,
                            [run['partial_pressures'][key][species] for key in run['partial_pressures'].keys()],
                            label=format_species_string(species))
    ax[plot_index].set_ylabel("Partial Pressure (bar)")
    ax[plot_index].plot(
        np.array(list(run["total_pressure"].keys())) * 100, run["total_pressure"].values(), linestyle="dotted",
        color='black', label="Total Fractional Pressure"
    )
    ax[plot_index].axvline(x=run["vmf"], linestyle="--", color='black')
    # plot atomic partial pressures
    for species in atomic_species:
        ax[plot_index + 1].plot(np.array(list(run['atomic_partial_pressures'].keys())) * 100,
                            [run['atomic_partial_pressures'][key][species] for key in
                             run['atomic_partial_pressures'].keys()], label=format_species_string(species))
        ax[plot_index + 1].set_ylabel("Atomic Partial Pressure (bar)")
    ax[plot_index + 1].plot(
        np.array(list(run["total_pressure"].keys())) * 100, run["total_pressure"].values(), linestyle="dotted",
        color='black', label="Total Fractional Pressure"
    )
    ax[plot_index + 1].axvline(x=run["vmf"], linestyle="--", color='black')
    ax[index].set_ylabel("Partial Pressure (bar)")
    plot_index += 2

for a in ax:
    a.grid()
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(10 ** -1.5, 80)
    a.set_ylim(10 ** -3, 10 ** 1.1)
    labellines.labelLines(a.get_lines(), zorder=2.5, align=True,
                          xvals=[uniform(10 ** -1, 8) for i in a.get_lines()], fontsize=8)
for a in [ax[2], ax[3]]:
    a.set_xlabel("VMF (%)")

for a, l in zip(ax, [runs[0]["run_name"], runs[0]["run_name"], runs[1]["run_name"], runs[1]["run_name"]]):
   # annotate the run name in the upper left corner
    a.annotate(l, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, ha='left', va='top', color='black')

plt.tight_layout()
# plt.show()
plt.savefig("partial_pressures.png", format='png', dpi=200)
# close the figure
plt.close()


# calculate P_i assuming S_i = 0.998
S_i = 0.989
delta_K_Lunar_BSE = 0.415  # mean of 41K/39K isotope ratios from lunar samples

# make a plot of P_i for each species for each run in a 1 row 2 column plot
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex='all')
ax = ax.flatten()
plot_index = 0
for run in runs:
    # P_i for each species is the partial pressure of the species times S_i
    atomic_species = run["atomic_partial_pressures"][list(run["atomic_partial_pressures"].keys())[0]].keys()
    run["P_i"] = {
        key:
            {
                species: run["atomic_partial_pressures"][key][species] for species in atomic_species
            }
        for key in run["atomic_partial_pressures"].keys()
    }
    # plot P_i for each species
    for species in atomic_species:
        ax[plot_index].plot(np.array(list(run['P_i'].keys())) * 100,
                            [run['P_i'][key][species] for key in run['P_i'].keys()],
                            label=format_species_string(species))
    # plot the total boundary layer pressure, which is the sum of all P_i
    ax[plot_index].plot(
        np.array(list(run["P_i"].keys())) * 100, [sum(data.values()) for data in run["P_i"].values()],
        linestyle="dotted", color='black', label="Total Boundary Pressure")
    ax[plot_index].set_ylabel(r"$P_i$ (bar)")
    ax[plot_index].axvline(x=run["vmf"], linestyle="--", color='black')
    ax[plot_index].set_xlabel("VMF (%)")
    plot_index += 1

for a in ax:
    a.grid()
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(10 ** -1.5, 80)
    a.set_ylim(10 ** -5, 10 ** 1.2)
    labellines.labelLines(a.get_lines(), zorder=2.5, align=True,
                          xvals=[uniform(10 ** -1, 2) for i in a.get_lines()], fontsize=12)
    a.set_xlabel("VMF (%)")

for a, l in zip(ax, [runs[0]["run_name"], runs[1]["run_name"]]):
   # annotate the run name in the upper left corner
    a.annotate(l, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top', color='black')

plt.tight_layout()
# plt.show()
plt.savefig("p_i.png", format='png', dpi=200)
# close the figure
plt.close()

# calculate the isotope fractionation
delta_vapor = None
delta_melt = delta_K_Lunar_BSE
total_melt_mass = None
total_vapor_mass = 0
for run in runs:
    rf = SaturatedRayleighFractionation()
    for vmf, melt_composition in run['melt_composition'].items():
        alpha_chem = (39 / 41) ** 0.43
        alpha_phys = (39 / 41) ** 0.5
        frac_vapor_mass = run['fractional_vapor_composition'][vmf]['K']
        melt_mass = melt_composition['K']
        total_melt_mass = melt_mass
        total_mass = melt_mass + frac_vapor_mass
        # calculate the isotope fractionation due to the initial evaporation
        melt_delta = rf.rayleigh_fractionate_residual(
            delta_initial=delta_melt, alpha=alpha_chem, f=melt_mass / (melt_mass + frac_vapor_mass), S=S_i
        )
        melt_vapor = rf.rayleigh_fractionate_extract(
            delta_initial=delta_melt, alpha=alpha_chem, f=melt_mass / (melt_mass + frac_vapor_mass), S=S_i
        )
        # if there is no vapor yet produced, then the vapor delta is the initial extract vapor



