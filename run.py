import copy

from monte_carlo.monte_carlo import run_monte_carlo_vapor_loss
from theia.theia import get_theia_composition, recondense_vapor
from monte_carlo.monte_carlo import test
from src.plots import collect_data
from src.composition import normalize, get_molecular_mass

import os
import string
from random import uniform, shuffle
import numpy as np
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
    "SiO2": 41.73699,
    'MgO': 32.82540,
    'Al2O3': 3.66824,
    'TiO2': 0.19372,
    'Fe2O3': 0.0,
    'FeO': 18.46253,
    'CaO': 3.07599,
    'Na2O': 0.03335,
    'K2O': 0.00355,
    'ZnO': 0.00023,
}

# ============================== Define Input Parameters ==============================

run_name = "500b073S"
temperature = 2682.61  # K
vmf = 0.96  # %
disk_theia_mass_fraction = 66.78  # %
disk_mass = 1.02  # lunar masses
vapor_loss_fraction = 75.0  # %
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
        guess_initial_composition=bse_composition, target_composition=bulk_moon_composition, temperature=temperature,
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
formatted_oxides = [rf"$\rm {oxide.replace('2', '_2')}$" for oxide in bse_composition.keys() if oxide != 'Fe2O3']
for t, c in [
    ("BSM", bulk_moon_composition),
    ("Bulk Ejecta Composition", ejecta_data['ejecta composition']),
    ("Bulk Silicate Theia", theia_data['theia_weight_pct'])
]:
    ax.plot(
        [i for i in bse_composition.keys() if i != "Fe2O3"], [c[oxide] / bse_composition[oxide]
                                 for oxide in bse_composition.keys() if oxide != "Fe2O3"],
        linewidth=3, label=t
    )
# plot a horizontal line at 1.0
ax.plot([i for i in bse_composition.keys() if i != "Fe2O3"], [1.0 for _ in bse_composition.keys() if _ != "Fe2O3"],
        linewidth=3, linestyle="--", label="1:1 BSE")
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
for vmf, element_masses in vapor_element_masses.items():
    # get the total mass of the vapor
    total_mass = sum(element_masses.values())
    # convert the element masses to mole fractions
    vapor_element_moles = {element: mass / get_molecular_mass(element) for element, mass in element_masses.items()}
    vapor_mole_fraction = {element: moles / sum(vapor_element_moles.values()) for element, moles in vapor_element_moles.items()}
    vapor_element_mole_fractions[vmf] = vapor_mole_fraction

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

# ============================== Plot Melt + Recondensed Vapor as a Function of VMF ==============================

# make a figure with two subplots that share the same x and y axes
fig, axs = plt.subplots(2, 1, figsize=(9, 14))
axs = axs.flatten()
axs[0].set_ylabel(r'log Mole Fraction ($X_{i}$)', fontsize=20)
axs[1].set_ylabel(r'log Mole Fraction ($X_{i}$)', fontsize=20)
axs[1].set_xlabel(r'VMF (%)', fontsize=20)
# increase the font size and add a grid
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axvline(x=vmf, linewidth=3, color="black", linestyle="--", label="SPH VMF")
# get the melt and vapor masses across all VMFs
melt_mole_frac = collect_data(path=f"{run_name}/magma_oxide_mole_fraction", x_header='mass fraction vaporized')
vapor_mole_frac = collect_data(path=f"{run_name}/atmosphere_mole_fraction", x_header='mass fraction vaporized')
# colors_melt = plt.cm.jet(np.linspace(0, 1, len(melt_mole_frac[vmfs[0]])))
# colors_vapor = plt.cm.jet(np.linspace(0, 1, len(vapor_mole_frac[vmfs[0]])))
colors_melt = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 10
colors_vapor = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 10
# randomize the colors
shuffle(colors_melt)
shuffle(colors_vapor)
# plot the melt mole fraction as a function of VMF
vmf_index = 2
for index, s in enumerate(melt_mole_frac[vmfs[0]].keys()):
    axs[0].plot(np.array(vmfs) * 100, [melt_mole_frac[vmf][s] for vmf in vmfs], linewidth=2.0,
                color=colors_melt[index], label=index)
    # annotate each species with its name right at the line and 25% of the way along the x axis
    axs[0].annotate(s.split("_")[0],
                    (vmfs[vmf_index] * 100,
                        melt_mole_frac[vmfs[vmf_index]][s] +
                        melt_mole_frac[vmfs[vmf_index]][s] * 0.08),
                    fontsize=16, color=colors_melt[index])
for index, s in enumerate(vapor_mole_frac[vmfs[0]].keys()):
    axs[1].plot(np.array(vmfs) * 100, [vapor_mole_frac[vmf][s] for vmf in vmfs], linewidth=2.0,
                color=colors_vapor[index], label=index)
    # annotate each species with its name right at the line and 25% of the way along the x axis
    axs[1].annotate(s.split("_")[0],
                    (vmfs[vmf_index] * 100,
                        vapor_mole_frac[vmfs[vmf_index]][s] +
                        vapor_mole_frac[vmfs[vmf_index]][s] * 0.08),
                    fontsize=16, color=colors_vapor[index])

# set minimum plotted x value
letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs):
    ax.set_xlim(left=min(np.array(list(melt_mole_frac.keys())) * 100), right=80)
    ax.set_ylim(bottom=10 ** -3)
    # label each subplot with a letter in the upper-left corner
    ax.annotate(
        letters[index], xy=(0.05, 0.95), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontweight="bold", fontsize=20
    )
plt.savefig(f"{run_name}_melt_vapor_mole_fractions.png", dpi=300)
plt.tight_layout()
plt.show()