import copy

from monte_carlo.monte_carlo import run_monte_carlo_vapor_loss
from theia.theia import get_theia_composition, recondense_vapor
from monte_carlo.monte_carlo import test
from src.plots import collect_data, collect_metadata
from src.composition import normalize, get_molecular_mass
from isotopes.rayleigh import FullSequenceRayleighDistillation

import os
import sys
import string
from scipy.interpolate import interp1d
from random import uniform, shuffle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# use colorblind-friendly colors
plt.style.use('seaborn-colorblind')

# ============================== Define Compositions ==============================

bse_composition = {  # Visscher and Fegley (2013)
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

bsm_composition = {  # Visscher and Fegley (2013)
    "SiO2": 44.60,
    'MgO': 35.10,
    'Al2O3': 3.90,
    'TiO2': 0.17,
    'Fe2O3': 0.00000,
    'FeO': 12.40,
    'CaO': 3.30,
    'Na2O': 0.050,
    'K2O': 0.004,
    'ZnO': 2.0e-4,
}

bulk_moon_composition = {  # including core Fe as FeO, my mass balance calculation
    "SiO2": 47.0,
    'MgO': 29.0,
    'Al2O3': 6.0,
    'TiO2': 0.3,
    'Fe2O3': 0.0,
    'FeO': 13.0,
    'CaO': 4.6,
    'Na2O': 9.0e-2,
    'K2O': 1.0e-2,
    'ZnO': 2.39e-5,
}

# ============================== Define Input Parameters ==============================

run_name = "500b073S"
temperature = 2682.61  # K
vmf = 0.96  # %
disk_theia_mass_fraction = 66.78  # %
disk_mass = 1.02  # lunar masses
vapor_loss_fraction = 74.0  # %
run_new_simulation = False  # True to run a new simulation, False to load a previous simulation

# ============================== Define Constants ==============================

disk_earth_mass_fraction = 100 - disk_theia_mass_fraction  # %, fraction
mass_moon = 7.34767309e22  # kg, mass of the moon
disk_mass_kg = disk_mass * mass_moon  # kg, mass of the disk
earth_mass_in_disk_kg = disk_mass_kg * disk_earth_mass_fraction / 100  # kg, mass of the earth in the disk
theia_mass_in_disk = disk_mass_kg - earth_mass_in_disk_kg  # kg, mass of theia in the disk

# ============================== Do some file management ==============================
ejecta_file_path = f"{run_name}_ejecta_data.txt"
theia_file_path = f"{run_name}_theia_composition.txt"

# delete the output files if its a new simulation and they already exist
if run_new_simulation:
    for f in [ejecta_file_path, theia_file_path]:
        try:
            os.remove(f)
        except OSError:
            pass

# ============================== Calculate Bulk Ejecta Composition ==============================

# run the monte carlo simulation
# the initial guess is the BSE composition
# this will calculate the bulk ejecta composition that is required to reproduce the bulk moon composition
# (liquid + retained vapor that is assumed to recondense) at the given VMF and temperature
if run_new_simulation:
    ejecta_data = test(
        guess_initial_composition=bsm_composition, target_composition=bulk_moon_composition, temperature=temperature,
        vmf=vmf, vapor_loss_fraction=vapor_loss_fraction, full_report_path=f"{run_name}"
    )
else:
    # read in the data dictionary from the file
    ejecta_data = eval(open(ejecta_file_path, 'r').read())

# ============================== Calculate Bulk Silicate Theia (BST) Composition ==============================

if run_new_simulation:
    theia_data = get_theia_composition(starting_composition=ejecta_data['ejecta composition'],
                                       earth_composition=bse_composition, disk_mass=disk_mass_kg,
                                       earth_mass=earth_mass_in_disk_kg)
else:
    # read in the data dictionary from the file
    theia_data = eval(open(theia_file_path, 'r').read())

# ============================== Save Results  ==============================
# write the ejecta data (dictionary) to a file in text format
if run_new_simulation:
    with open(ejecta_file_path, "w") as f:
        f.write(str({k: v for k, v in ejecta_data.items() if k not in ['c', 'l', 'g', 't']}))
    # now, write the theia composition dictionary to a file in text format
    with open(theia_file_path, "w") as f:
        f.write(str({k: v for k, v in theia_data.items() if k not in ['c', 'l', 'g', 't']}))

# ============================== Plot Bulk Ejecta + BST Relative to BSE ==============================

# calculate the composition of bulk silicate Theia (BST)
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
# increase the font size
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel('Oxide', fontsize=20)
ax.set_ylabel("Oxide Wt. % (Relative to BSE)", fontsize=20)
ax.grid()
# make all numbers in the molecules subscript without italics
formatted_oxides = [rf"$\rm {oxide.replace('2', '_{2}').replace('3', '_{3}')}$" for oxide in bse_composition.keys() if
                    oxide != 'Fe2O3']
for t, c in [
    ("Bulk Moon", bulk_moon_composition),
    ("BSM", bsm_composition),
    ("Bulk Ejecta", ejecta_data['ejecta composition']),
    ("Bulk Silicate Theia", theia_data['theia_weight_pct'])
]:
    ax.plot(
        [i for i in bse_composition.keys() if i != "Fe2O3"], [c[oxide] / bse_composition[oxide]
                                                              for oxide in bse_composition.keys() if oxide != "Fe2O3"],
        linewidth=3, label=t
    )
# plot a horizontal line at 1.0
ax.plot([i for i in bse_composition.keys() if i != "Fe2O3"], [1.0 for _ in bse_composition.keys() if _ != "Fe2O3"],
        linewidth=3, color='black', label="1:1 BSE")
# make x axis labels at 45 degree angle
plt.xticks(rotation=45)
# add a legend with large font
ax.legend(fontsize=20)
# set xlim to start at 0 and end at the length of the x axis
ax.set_xlim(0, len(bse_composition.keys()) - 2)
# replace x-axis labels with the formatted oxide names
ax.set_xticklabels(formatted_oxides)
plt.tight_layout()
plt.savefig(f"{run_name}_bulk_ejecta_and_bst_relative_to_bse.png", dpi=300)
plt.show()

# # ============================== Plot Vapor Species Composition As Function of VMF ==============================
# # TODO: make sure this is a summation of the vapor composition, not just the vapor composition at equilibrium at
# #  the given VMF
#
# # get the vapor mole fraction at all VMFs
# vapor_comp_mole_fraction_all_vmf = collect_data(path=f"{run_name}/atmosphere_mole_fraction",
#                                                 x_header='mass fraction vaporized')
# vapor_comp_mole_fraction_all_vmf = {key: normalize({i: j for i, j in value.items() if "_l" not in i})
#                                     for key, value in vapor_comp_mole_fraction_all_vmf.items()}
# vmfs = sorted([float(i) for i in vapor_comp_mole_fraction_all_vmf.keys()])
# species = sorted([i for i in vapor_comp_mole_fraction_all_vmf[vmfs[0]].keys()])
# fig = plt.figure(figsize=(16, 9))
# ax = fig.add_subplot(111)
# # increase the font size
# ax.tick_params(axis='both', which='major', labelsize=20)
# ax.set_xlabel(r'VMF (%)', fontsize=20)
# ax.set_ylabel(r'log Mole Fraction ($X_{i}$)', fontsize=20)
# ax.grid()
# # plot the mole fraction of each species as a function of VMF
# for s in species:
#     ax.plot(np.array(vmfs) * 100, [vapor_comp_mole_fraction_all_vmf[vmf][s] for vmf in vmfs], linewidth=2.0, label=s)
# # make y axis log scale
# ax.set_yscale('log')
# # make a vertical line at the given VMF
# ax.axvline(x=vmf, linewidth=3, color="black", label="SPH VMF")
# plt.tight_layout()
# # annotate each species with its name right above the line and 25% of the way along the x axis
# annotation_vmf_location = 35
# for s in species:
#     # get the mole fraction at the VMF closest to the annotation VMF
#     annotation_vmf = min(vmfs, key=lambda x: abs(x - annotation_vmf_location / 100))
#     # get the index of the annotation VMF
#     annotation_vmf_index = vmfs.index(annotation_vmf)
#     ax.annotate(s.split("_")[0], (annotation_vmf_location, vapor_comp_mole_fraction_all_vmf[vmfs[annotation_vmf_index]][s]), fontsize=20)
# plt.show()

# ============================== Plot Vapor Element Mole Fraction As Function of VMF ==============================
vapor_element_masses = collect_data(path=f"{run_name}/total_vapor_element_mass",
                                    x_header='mass fraction vaporized')
# convert the vapor element masses to mole fractions for each VMF
vapor_element_mole_fractions = {}
for vmf_val, element_masses in vapor_element_masses.items():
    # get the total mass of the vapor
    total_mass = sum(element_masses.values())
    # convert the element masses to mole fractions
    vapor_element_moles = {element: mass / get_molecular_mass(element) for element, mass in element_masses.items()}
    vapor_mole_fraction = {element: moles / sum(vapor_element_moles.values()) for element, moles in
                           vapor_element_moles.items()}
    vapor_element_mole_fractions[vmf_val] = vapor_mole_fraction

# plot it
vmfs = sorted([float(i) for i in vapor_element_mole_fractions.keys()])
species = sorted([i for i in vapor_element_mole_fractions[vmfs[0]].keys()])
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
# increase the font size
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel(r'VMF (%)', fontsize=20)
ax.set_ylabel(r'log Mole Fraction ($X_{i}$)', fontsize=20)
ax.grid()
# plot the mole fraction of each species as a function of VMF
for s in species:
    ax.plot(np.array(vmfs) * 100, [vapor_element_mole_fractions[vmf][s] for vmf in vmfs], linewidth=2.0, label=s)
# make y axis log scale
ax.set_yscale('log')
# make a vertical line at the given VMF
ax.axvline(x=vmf, linewidth=3, color="black", linestyle="--", label="SPH VMF")
plt.tight_layout()
# annotate each species with its name right above the line and 25% of the way along the x axis
annotation_vmf_location = 35
annotation_vmf_location = {
    i: annotation_vmf_location for i in species
}
annotation_vmf_location["K"] = 5
annotation_vmf_location["Zn"] = 5
annotation_vmf_location["Ca"] = 5
annotation_vmf_location['Ti'] = 30
annotation_vmf_location['Mg'] = 20
annotation_vmf_location['Al'] = 20
for s in species:
    # add some spread so there isn't overlap
    # get the mole fraction at the VMF closest to the annotation VMF
    annotation_vmf = min(vmfs, key=lambda x: abs(x - annotation_vmf_location[s] / 100))
    # get the index of the annotation VMF
    annotation_vmf_index = vmfs.index(annotation_vmf)
    ax.annotate(s.split("_")[0],
                (annotation_vmf_location[s],
                 vapor_element_mole_fractions[vmfs[annotation_vmf_index]][s] +
                 vapor_element_mole_fractions[vmfs[annotation_vmf_index]][s] * 0.08),
                fontsize=20)
# make 0 the minimum value of the x-axis
ax.set_xlim(left=0)
plt.savefig(f"{run_name}_element_vapor_mass.png", dpi=300)
plt.show()

# ============================== Plot Melt + Vapor as a Function of VMF ==============================

# make a figure with two subplots that share the same x and y axes
fig, axs = plt.subplots(3, 1, figsize=(9, 16))
axs = axs.flatten()
axs[0].set_title("Canonical", fontsize=20)
axs[0].set_ylabel(r'Melt - Mole Fraction', fontsize=20)
axs[1].set_ylabel(r'Fractional Vapor - Mole Fraction', fontsize=20)
axs[2].set_ylabel(r'Bulk Vapor - Mole Fraction', fontsize=20)
axs[2].set_xlabel(r'VMF (%)', fontsize=20)
# increase the font size and add a grid
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axvline(x=vmf, linewidth=3, color="black", linestyle="--", label="SPH VMF")
# get the melt and vapor masses across all VMFs
melt_mole_frac = collect_data(path=f"{run_name}/magma_oxide_mole_fraction", x_header='mass fraction vaporized')
# print mole fraction of ZnO as a function of VMF
vapor_mole_frac = collect_data(path=f"{run_name}/atmosphere_mole_fraction", x_header='mass fraction vaporized')

# get the bulk vapor species mass
vapor_species_mass = collect_data(path=f"{run_name}/total_vapor_species_mass", x_header='mass fraction vaporized')
# for each VMF, convert the vapor species mass to moles and then to mole fractions
vapor_species_mole_fractions = {}  # bulk vapor
for vmf, species_mass in vapor_species_mass.items():
    # convert the species masses to mole fractions
    vapor_species_moles = {element: mass / get_molecular_mass(element) if mass > 0 else 0 for element, mass in species_mass.items()}
    vapor_species_mole_fraction = {element: moles / sum(vapor_species_moles.values()) for element, moles in
                                   vapor_species_moles.items()}
    vapor_species_mole_fractions[vmf] = vapor_species_mole_fraction

colors_melt = sns.color_palette('husl', n_colors=len(melt_mole_frac[vmfs[0]]))
colors_vapor = sns.color_palette('husl', n_colors=len(vapor_mole_frac[vmfs[0]]))
# shuffle(colors_melt)
# shuffle(colors_vapor)
# plot the melt/vapor mole fraction as a function of VMF
vmf_to_plot_vapor = 10 ** -1.5
vmf_to_plot_override_vapor = {
    'Zn': [10 ** -3, 10 ** -2.8],
    'O2': [10 ** -0.7, 10 ** -0.6],
    # 'SiO2': [10 ** -0.9, 10 ** -2.2],
    # 'NaO': [10 ** -0.95, 10 ** -2.5],
    "SiO": [10 ** - 2.4, 10 ** -0.54],
}
vmf_to_plot_melt = 10 ** -1.1
vmf_to_plot_override_melt = {
    "MgO": [10 ** -2, 10 ** -0.65],
    "SiO2": [10 ** -1.5, 10 ** -0.35],
    # 'TiO2': [10 ** -1.2, 10 ** -2.9],
    # "K2O": [10 ** -1.5, 10 ** -4.7],
    "ZnO": [10 ** -3.3, 10 ** -6.6],
    "Al2O3": [10 ** -1.5, 10 ** -1.82],
}
vmf_to_plot_bulk_vapor = 10 ** -1.5
vmf_to_plot_override_bulk_vapor = {
    'Zn': [10 ** -2.1, 10 ** -2.45],
    'O2': [10 ** -0.7, 10 ** -0.6],
    # 'SiO2': [10 ** -0.9, 10 ** -2.2],
    # 'NaO': [10 ** -0.95, 10 ** -2.5],
    "SiO": [10 ** - 2.4, 10 ** -0.54],
}
for index, s in enumerate(melt_mole_frac[vmfs[0]].keys()):
    # format s so that all numbers are subscript
    s_formatted = s.split('_')[0].replace("2", "_{2}").replace("3", "_{3}")
    x, y = 0, 0
    if not s.split("_")[0] in vmf_to_plot_override_melt.keys():
        # get the closest VMF to the one we want to plot
        closest_vmf = min(vmfs, key=lambda x: abs(x - vmf_to_plot_melt / 100))
        # get the index of the closest VMF
        vmf_index = vmfs.index(closest_vmf)
        x = vmfs[vmf_index] * 100
        y = melt_mole_frac[vmfs[vmf_index]][s]
    else:
        x, y = vmf_to_plot_override_melt[s.split("_")[0]]
    axs[0].plot(np.array(vmfs) * 100, [melt_mole_frac[vmf][s] for vmf in vmfs], linewidth=2.0,
                color=colors_melt[index], label=index)
    # annotate each species with its name right at the line and 25% of the way along the x axis
    axs[0].annotate(rf"$\rm {s_formatted}$",
                    (x, y + (y * 0.08)),
                    fontsize=18, color=colors_melt[index])
for index, s in enumerate(vapor_mole_frac[vmfs[0]].keys()):
    s_formatted = s.split('_')[0].replace("2", "_{2}").replace("3", "_{3}")
    x, y = 0, 0
    if not s.split("_")[0] in vmf_to_plot_override_vapor.keys():
        # get the closest VMF to the one we want to plot
        closest_vmf = min(vmfs, key=lambda x: abs(x - vmf_to_plot_vapor / 100))
        # get the index of the closest VMF
        vmf_index = vmfs.index(closest_vmf)
        x = vmfs[vmf_index] * 100
        y = vapor_mole_frac[vmfs[vmf_index]][s]
    else:
        x, y = vmf_to_plot_override_vapor[s.split("_")[0]]
    axs[1].plot(np.array(vmfs) * 100, [vapor_mole_frac[vmf][s] for vmf in vmfs], linewidth=2.0,
                color=colors_vapor[index], label=index)
    # annotate each species with its name right at the line and 25% of the way along the x axis
    axs[1].annotate(rf"$\rm {s_formatted}$",
                    (x, y + (y * 0.08)),
                    fontsize=18, color=colors_vapor[index])

for index, s in enumerate(vapor_species_mole_fractions[vmfs[0]].keys()):
    s_formatted = s.split('_')[0].replace("2", "_{2}").replace("3", "_{3}")
    x, y = 0, 0
    if not s.split("_")[0] in vmf_to_plot_override_bulk_vapor.keys():
        # get the closest VMF to the one we want to plot
        closest_vmf = min(vmfs, key=lambda x: abs(x - vmf_to_plot_bulk_vapor / 100))
        # get the index of the closest VMF
        vmf_index = vmfs.index(closest_vmf)
        x = vmfs[vmf_index] * 100
        y = vapor_species_mole_fractions[vmfs[vmf_index]][s]
    else:
        x, y = vmf_to_plot_override_bulk_vapor[s.split("_")[0]]
    axs[2].plot(np.array(vmfs) * 100, [vapor_species_mole_fractions[vmf][s] for vmf in vmfs], linewidth=2.0,
                color=colors_vapor[index], label=index)
    # annotate each species with its name right at the line and 25% of the way along the x axis
    axs[2].annotate(rf"$\rm {s_formatted}$",
                    (x, y + (y * 0.08)),
                    fontsize=18, color=colors_vapor[index])

# set minimum plotted x value
letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs):
    ax.set_xlim(left=min(np.array(list(melt_mole_frac.keys())) * 100), right=80)
    # label each subplot with a letter in the upper-left corner
    ax.annotate(
        letters[index], xy=(0.05, 0.95), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontweight="bold", fontsize=20
    )
axs[0].set_ylim(bottom=10 ** -7, top=10 ** 0)
axs[1].set_ylim(bottom=10 ** -3, top=10 ** 0)
axs[2].set_ylim(bottom=10 ** -3, top=10 ** 0)
plt.savefig(f"{run_name}_melt_vapor_mole_fractions.png", dpi=300)
plt.tight_layout()
plt.show()

# ===================== Plot the most volatile element as a function of VMF =====================

fig, ax = plt.subplots(figsize=(8, 6))
# make font size larger
ax.tick_params(axis='both', which='major', labelsize=20)
# get the melt and vapor masses across all VMFs
vmf_and_most_volatile_element = [[key, values['most volatile species']] for key, values in collect_metadata(
    path=f"{run_name}/magma_oxide_mole_fraction", x_header='mass fraction vaporized'
).items()]
# sort the list by VMF
vmf_and_most_volatile_element.sort(key=lambda x: x[0])
unique_volatile_species = [i[1] for i in vmf_and_most_volatile_element]
# for each unique species, get the minimum and maximum VMF at which it is the most volatile
unique_volatile_species = list(set(unique_volatile_species))
volatile_species_to_plot = []
for s in unique_volatile_species:
    volatile_species_to_plot.append([
        s,
        min([i[0] for i in vmf_and_most_volatile_element if i[1] == s]) * 100,
        max([i[0] for i in vmf_and_most_volatile_element if i[1] == s]) * 100
    ])
volatile_species_to_plot.sort(key=lambda x: x[1])
for index, s in enumerate(volatile_species_to_plot):
    forward_diff = 0
    backward_diff = 0
    if index < len(volatile_species_to_plot) - 1:
        forward_diff = abs(volatile_species_to_plot[index + 1][1] - s[2])
    if index > 1:
        backward_diff = abs(s[1] - volatile_species_to_plot[index - 1][2])

    s[1] = s[1] - (backward_diff / 2)
    s[2] = s[2] + (forward_diff / 2)
# sort volatile_species_to_plot by the minimum VMF at which the species is the most volatile
# plot the most volatile species as a function of VMF
for s in volatile_species_to_plot:
    ax.plot(
        [s[1], s[2]], [s[0], s[0]],
        linewidth=5.0,
    )
ax.set_xlabel("Vapor Mass Fraction (%)", fontsize=20)
ax.set_ylabel("Most Volatile Element", fontsize=20)
# ax.set_xlim(left=0, right=80)
# use log scale on x-axis
# make a vertical line at the VMF
ax.axvline(x=vmf, color='k', linestyle='--', linewidth=2.0)
ax.set_xscale("log")
ax.grid()
plt.tight_layout()
plt.savefig(f"{run_name}_most_volatile_element.png", dpi=300)
plt.show()

# ============================ Fraction of Element Lost to Escaping Vapor ============================
# get the mass of each element in the bulk vapor
vapor_element_masses = collect_data(path=f"{run_name}/total_vapor_element_mass", x_header='mass fraction vaporized')
# get the mass of each element in the bulk melt
melt_element_masses = collect_data(path=f"{run_name}/magma_element_mass", x_header='mass fraction vaporized')
melt_metadata = collect_metadata(path=f"{run_name}/magma_element_mass", x_header='mass fraction vaporized')
# total mass of each element in the system
# go through each VMF and add up the mass of each element in the melt and vapor
total_element_masses = {}
for vmf_val in melt_element_masses.keys():
    total_element_masses[vmf_val] = {}
    for element in melt_element_masses[vmf_val].keys():
        total_element_masses[vmf_val][element] = melt_element_masses[vmf_val][element] + vapor_element_masses[vmf_val][
            element]
# verify that the total element mass is conserved across all VMFs
for vmf_val in melt_element_masses.keys():
    for element in melt_element_masses[vmf_val].keys():
        assert np.isclose(
            melt_element_masses[vmf_val][element] + vapor_element_masses[vmf_val][element],
            total_element_masses[vmf_val][element]
        )
    # make sure that the total mass of the system equals the initial liquid mass
    assert np.isclose(
        melt_metadata[vmf_val]['initial liquid mass'],
        sum(total_element_masses[vmf_val].values())
    )
# fraction of each element lost to vapor (without recondensation)
fraction_lost_to_vapor = {}
for vmf_val in melt_element_masses.keys():
    fraction_lost_to_vapor[vmf_val] = {}
    for element in melt_element_masses[vmf_val].keys():
        fraction_lost_to_vapor[vmf_val][element] = melt_element_masses[vmf_val][element] / \
                                                   total_element_masses[vmf_val][element]
# fraction of each element lost to vapor (with recondensation)
fraction_lost_to_vapor_recondensed = {}
for vmf_val in melt_element_masses.keys():
    fraction_lost_to_vapor_recondensed[vmf_val] = {}
    for element in melt_element_masses[vmf_val].keys():
        fraction_lost_to_vapor_recondensed[vmf_val][element] = (melt_element_masses[vmf_val][element] + (
        (vapor_element_masses[vmf_val][element] * (1 - (vapor_loss_fraction / 100))))) / total_element_masses[vmf_val][
                                                                   element]
# interpolate the fraction of each element lost to vapor (without recondensation) at the VMF of interest
fraction_lost_to_vapor_during_vaporization_at_vmf = {}
fraction_lost_to_vapor_with_recondensation_at_vmf = {}
# get the vmf immediately above and below the vmf of interest
vmf_above = min([i for i in fraction_lost_to_vapor.keys() if i > vmf / 100])
vmf_below = max([i for i in fraction_lost_to_vapor.keys() if i < vmf / 100])
for element in fraction_lost_to_vapor[vmf_above].keys():
    # interpolate each element's fraction lost to vapor (not recondensed) at the vmf of interest
    fraction_lost_to_vapor_during_vaporization_at_vmf[element] = interp1d(
        [vmf_below, vmf_above],
        [fraction_lost_to_vapor[vmf_below][element], fraction_lost_to_vapor[vmf_above][element]]
    )(vmf / 100)
    # interpolate each element's fraction lost to vapor (with recondensation) at the vmf of interest
    fraction_lost_to_vapor_with_recondensation_at_vmf[element] = interp1d(
        [vmf_below, vmf_above],
        [fraction_lost_to_vapor_recondensed[vmf_below][element], fraction_lost_to_vapor_recondensed[vmf_above][element]]
    )(vmf / 100)

# write both fraction_lost_to_vapor_at_vmf and fraction_lost_to_vapor_recondensed_at_vmf to a file
if f"{run_name}_fraction_lost_to_vapor_at_vmf.txt" in os.listdir():
    os.remove(f"{run_name}_fraction_lost_to_vapor_at_vmf.txt")
with open(f"{run_name}_fraction_lost_to_vapor_at_vmf.txt", 'w') as f:
    header = fraction_lost_to_vapor_during_vaporization_at_vmf.keys()
    # make header a comma-separated string
    header_f = "phase," + ",".join(header)
    f.write(f"{header_f}\n")
    f.write(
        "non-recondensed," + ",".join(
            [str(fraction_lost_to_vapor_during_vaporization_at_vmf[i]) for i in header]) + "\n"
    )
    f.write(
        "recondensed," + ",".join([str(fraction_lost_to_vapor_with_recondensation_at_vmf[i]) for i in header]) + "\n"
    )
f.close()

# plot the fraction of each element lost to vapor (with and without recondensation) at the VMF of interest
# make a figure with 2 rows and 1 column
fig, axs = plt.subplots(2, 1, figsize=(9, 14))
axs = axs.flatten()
axs[0].set_title("Canonical", fontsize=20)
# add a grid and x, y limits
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid()
    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.set_xlim(ax.set_xlim(left=min(np.array(list(melt_element_masses)) * 100), right=80))
# in the first row, plot the fraction of each element lost to vapor (without recondensation)
all_elements = melt_element_masses[vmf_above].keys()
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 10
colors = plt.cm.jet(np.linspace(0, 1, len(all_elements)))
for index, element in enumerate(all_elements):
    axs[0].plot(
        np.array(list(melt_element_masses.keys())) * 100.0,
        [melt_element_masses[vmf_val][element] / total_element_masses[vmf_val][element] for vmf_val in
         melt_element_masses.keys()],
        linewidth=2.0,
        color=colors[index],
        # label=element
    )
    axs[1].plot(
        np.array(list(melt_element_masses.keys())) * 100.0,
        [(melt_element_masses[vmf_val][element] + (
        (vapor_element_masses[vmf_val][element] * (1 - (vapor_loss_fraction / 100))))) / total_element_masses[vmf_val][
             element] for vmf_val in melt_element_masses.keys()],
        linewidth=2.0,
        color=colors[index],
        label=element
    )
# annotate in the upper right corner that this is prior to recondensation
for ax, title in [(axs[0], "Pre-Recondensation"), (axs[1], "Post-Recondensation")]:
    ax.annotate(
        title,
        xy=(10**-2.5, 0.1),
        xycoords="axes fraction",
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=16,
        fontweight="bold",
    )
for ax in [axs[0], axs[1]]:
    ax.set_ylabel("Residual Fraction in Melt", fontsize=20)
for ax in axs[-1:]:
    ax.set_xlabel("VMF (%)", fontsize=20)

for ax in [axs[0], axs[1]]:
    ax.axvline(x=vmf, linewidth=3, color="black", linestyle="--")

# annotate a letter in the upper left corner of each plot
for index, ax in enumerate(axs):
    ax.annotate(
        letters[index], xy=(0.05, 0.95), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontweight="bold", fontsize=20
    )

plt.tight_layout()

# place the legend to the right of both plots
legend = fig.legend(loc=7, fontsize=16)
for line in legend.get_lines():  # increase line widths in legend
    try:
        line.set_linewidth(4.0)
    except:
        pass
for handle in legend.legendHandles:  # increase marker sizes in legend
    try:
        handle.set_sizes([120.0])
    except:
        pass
fig.subplots_adjust(right=0.85)

plt.savefig(f"{run_name}_fraction_lost_to_vapor_at_vmf.png", dpi=300)
plt.show()


# make a spider plot of the fraction lost with and without recondensation at the VMF of interest
# TODO: make this fraction relative to Earth
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("Canonical", fontsize=20)
# add a grid and x, y limits
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid()
ax.plot(
    fraction_lost_to_vapor_during_vaporization_at_vmf.keys(),
    fraction_lost_to_vapor_during_vaporization_at_vmf.values(),
    linewidth=2.0,
    label="Without Recondensation"
)
ax.plot(
    fraction_lost_to_vapor_with_recondensation_at_vmf.keys(),
    fraction_lost_to_vapor_with_recondensation_at_vmf.values(),
    linewidth=2.0,
    label="With Recondensation"
)
ax.set_ylabel("Mass Loss Fraction", fontsize=20)
ax.axhline(1, linewidth=3, color="black")
ax.legend(fontsize=16)

plt.tight_layout()
plt.savefig(f"{run_name}_fraction_lost_to_vapor_at_vmf_spider.png", dpi=300)
plt.show()

# TODO: make the same figure but as a coupled bar graph


# ========================== Export interpolated data to a file ==========================
# get the mass of each element in the bulk vapor
vapor_element_masses = collect_data(path=f"{run_name}/total_vapor_element_mass", x_header='mass fraction vaporized')
# get the mass of each element in the bulk melt
melt_element_masses = collect_data(path=f"{run_name}/magma_element_mass", x_header='mass fraction vaporized')
melt_metadata = collect_metadata(path=f"{run_name}/magma_element_mass", x_header='mass fraction vaporized')
# interpolate melt and vapor mass data to the VMF of interest
vmf_above = min([i for i in fraction_lost_to_vapor.keys() if i > vmf / 100])
vmf_below = max([i for i in fraction_lost_to_vapor.keys() if i < vmf / 100])
melt_mass_at_vmf = {}
bulk_vapor_mass_at_vmf = {}
escaping_vapor_mass_at_vmf = {}
retained_vapor_mass_at_vmf = {}
for element in fraction_lost_to_vapor[vmf_above].keys():
    # interpolate each element's fraction lost to vapor (not recondensed) at the vmf of interest
    melt_mass_at_vmf[element] = interp1d(
        [vmf_below, vmf_above],
        [melt_element_masses[vmf_below][element], melt_element_masses[vmf_above][element]]
    )(vmf / 100)
    # interpolate each element's fraction lost to bulk vapor
    bulk_vapor_mass_at_vmf[element] = interp1d(
        [vmf_below, vmf_above],
        [vapor_element_masses[vmf_below][element], vapor_element_masses[vmf_above][element]]
    )(vmf / 100)
    # get the escaping vapor mass
    escaping_vapor_mass_at_vmf[element] = bulk_vapor_mass_at_vmf[element] * (vapor_loss_fraction / 100)
    # get the retained vapor mass
    retained_vapor_mass_at_vmf[element] = bulk_vapor_mass_at_vmf[element] - escaping_vapor_mass_at_vmf[element]
# assume mass balance
total_mass = sum(melt_mass_at_vmf.values()) + sum(bulk_vapor_mass_at_vmf.values())
# print(
#     total_mass, melt_metadata[vmf_above]['initial liquid mass'], melt_metadata[vmf_above]['mass liquid'] + melt_metadata[vmf_above]['mass vapor']
# )
# assert total_mass == melt_metadata[vmf_above]['initial liquid mass'] == melt_metadata[vmf_above]['mass liquid'] + melt_metadata[vmf_above]['mass vapor']
# export the data to a file
if os.path.exists(f"{run_name}_mass_distribution.csv"):
    os.remove(f"{run_name}_mass_distribution.csv")
with open(f"{run_name}_mass_distribution.csv", "w") as f:
    header = "component," + ",".join([str(i) for i in melt_mass_at_vmf.keys()]) + "\n"
    f.write(header)
    f.write("melt mass," + ",".join([str(i) for i in melt_mass_at_vmf.values()]) + "\n")
    f.write("bulk vapor mass," + ",".join([str(i) for i in bulk_vapor_mass_at_vmf.values()]) + "\n")
    f.write("bulk system mass," + ",".join([str(i) for i in (np.array(list(melt_mass_at_vmf.values())) + np.array(list(bulk_vapor_mass_at_vmf.values()))).tolist()]) + "\n")
    f.write("escaping vapor mass," + ",".join([str(i) for i in escaping_vapor_mass_at_vmf.values()]) + "\n")
    f.write("retained vapor mass," + ",".join([str(i) for i in retained_vapor_mass_at_vmf.values()]) + "\n")
f.close()

# ========================== Model Rayleigh Isotope Fractionation ==========================
# read in the mass distribution file
mass_distribution = pd.read_csv(f"{run_name}_mass_distribution.csv", index_col='component')
# model 41K/39K fractionation
k_isotopes = FullSequenceRayleighDistillation(
    heavy_z=41, light_z=39, vapor_escape_fraction=vapor_loss_fraction,
    system_element_mass=mass_distribution['K']['bulk system mass'], melt_element_mass=mass_distribution['K']['melt mass'],
                 vapor_element_mass=mass_distribution['K']['bulk vapor mass'], earth_isotope_composition=-0.479,
                theia_ejecta_fraction=disk_theia_mass_fraction
)
k_isotopes_starting_earth_isotope_composition = k_isotopes.run_3_stage_fractionation()  # assumes ejecta is fully Earth-like
k_isotopes_mixed_model, k_isotopes_mixed_model_best_fit = k_isotopes.run_theia_mass_balance(
    theia_range=np.arange(-.6, .2, 0.05),
    delta_moon_earth=0.415
)  # assumes ejecta is a mix of Earth and Theia

# make a subplot with 2 columns and 3 rows
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
# increase the font size
axs = axs.flatten()
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=14)
axs[0].plot(
    k_isotopes_mixed_model.keys(),
    [k_isotopes_mixed_model[i]['delta_moon_earth'] for i in k_isotopes_mixed_model.keys()],
    linewidth=2.0,
    color='black',
    label="Model",
)
# shade a region between the error bars of the observed value
axs[0].axhspan(
    0.415 - 0.015, 0.415 + 0.015, alpha=0.2, color='blue')
axs[0].axhline(0.415, color='blue', label=r"$\Delta_{\rm Lunar - BSE}^{\rm 41/39K}$ (Observed)")
axs[0].axvline(x=k_isotopes_mixed_model_best_fit, linestyle='--', linewidth=2.0, label=r"$\delta \rm ^{41/39}K_{\rm Theia}$ (Best Fit)")
# annotate the best fit value with vertical text
axs[0].annotate(
    r"$\delta \rm ^{41}K_{\rm Theia} = $" + f"{k_isotopes_mixed_model_best_fit:.3f}",
    (k_isotopes_mixed_model_best_fit - (k_isotopes_mixed_model_best_fit * .2), 0.1),
    rotation=90, fontsize=12
)
axs[0].axvspan(-0.479 - 0.027, -0.479 + 0.027, alpha=0.2, color='red')
axs[0].axvline(x=-0.479, color='red', label=r"$\delta \rm ^{41/39}K_{\rm Earth}$ (Observed)")
axs[0].set_title(r"$\rm ^{41/39}K$", fontsize=20)

zn_isotopes = FullSequenceRayleighDistillation(
    heavy_z=66, light_z=64, vapor_escape_fraction=vapor_loss_fraction,
    system_element_mass=mass_distribution['Zn']['bulk system mass'], melt_element_mass=mass_distribution['Zn']['melt mass'],
                 vapor_element_mass=mass_distribution['Zn']['bulk vapor mass'], earth_isotope_composition=0.28,
                theia_ejecta_fraction=disk_theia_mass_fraction
)
zn_isotopes_starting_earth_isotope_composition = zn_isotopes.run_3_stage_fractionation()  # assumes ejecta is fully Earth-like
zn_isotopes_mixed_model, zn_isotopes_mixed_model_best_fit = zn_isotopes.run_theia_mass_balance(
    theia_range=np.arange(-520, -480, 2),
    delta_moon_earth=1.12
)  # assumes ejecta is a mix of Earth and Theia

# make a plot of the 66Zn/64Zn fractionation with the Earth-Theia mixing model
axs[1].plot(
    zn_isotopes_mixed_model.keys(),
    [zn_isotopes_mixed_model[i]['delta_moon_earth'] for i in zn_isotopes_mixed_model.keys()],
    linewidth=2.0,
    color='black',
    label="Model"
)
# shade a region between the error bars of the observed value
axs[1].axhspan(
    1.4 - 0.5, 1.4 + 0.5, alpha=0.2, color='blue')
axs[1].axhline(1.4, color='blue', label=r"$\Delta_{\rm Lunar - BSE}^{\rm 66/64Zn}$ (Observed)")
axs[1].axvline(x=zn_isotopes_mixed_model_best_fit, linestyle='--', linewidth=2.0, label=r"$\delta \rm ^{66/64}zn_{\rm Theia}$ (Best Fit)")
axs[1].axvspan(0.28 - 0.05, 0.28 + 0.05, alpha=1, color='red')
axs[1].axvline(x=0.28, color='red', label=r"$\delta \rm ^{66/64}Zn_{\rm Earth}$ (Observed)")
axs[1].set_title(r"$\rm ^{66/64}Zn$", fontsize=20)
axs[1].annotate(
    r"$\delta \rm ^{66}Zn_{\rm Theia} = $" + f"{zn_isotopes_mixed_model_best_fit:.3f}",
    (zn_isotopes_mixed_model_best_fit - (zn_isotopes_mixed_model_best_fit * .02), -22),
    rotation=90, fontsize=12
)

# annotate the model in the upper right corner
# axs[0].annotate(
#     f"Canonical", xy=(0.85, 0.95), xycoords='axes fraction', horizontalalignment='right',
#     verticalalignment='top', fontweight='bold', fontsize=12
# )
# axs[2].annotate(
#     f"Half-Earths", xy=(0.85, 0.95), xycoords='axes fraction', horizontalalignment='right',
#     verticalalignment='top', fontweight='bold', fontsize=12
# )
# axs[4].annotate(
#     f"Mars", xy=(0.85, 0.95), xycoords='axes fraction', horizontalalignment='right',
#     verticalalignment='top', fontweight='bold', fontsize=12
# )

# annotate a letter in the upper left corner
letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs):
    # label each subplot with a letter in the upper-left corner
    ax.annotate(
        letters[index], xy=(0.05, 0.95), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontweight="bold", fontsize=20
    )
    ax.grid()
    # ax.legend()

for ax, t in [(axs[-2], r"$\delta \rm ^{41}K_{Theia}$"), (axs[-1], r"$\delta \rm ^{66}Zn_{Theia}$")]:
    ax.set_xlabel(t, fontsize=20)
for ax, t in [(axs[0], "Canonical"), (axs[2], "Half Earths"), (axs[4], "Mars")]:
    ax.set_ylabel(r"$\Delta_{\rm Lunar-BSE}$ " + f"({t})", fontsize=20)

plt.tight_layout()
plt.savefig(f"{run_name}_k_zn_isotope_fractionation_earth_theia_mixing.png", dpi=300)
plt.show()
