from src.composition import Composition, ConvertComposition, normalize, interpolate_composition_at_vmf, \
    mole_fraction_to_weight_percent, get_mean_molecular_mass
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data

from monte_carlo.monte_carlo import run_monte_carlo, run_monte_carlo_mp, write_file

import os
import csv
import numpy as np
import multiprocessing as mp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# use seaborn-colorblind palette
plt.style.use('seaborn-colorblind')

# SPH parameters at peak shock conditions
runs = {
    '500b073S-no-circ-vmf': {
        "temperature": 2682.61,  # this is the temperature of the disk
        # "vmf": 19.21,
        "vmf": 0.96 * 0.75,
        'theia_pct': 66.78,  # %
        'earth_pct': 100 - 66.78,  # %
        'disk_mass': 1.02,  # M_L
    },
}

# come back with atmosphere mass loss fraction from hydrodynamics code
atmosphere_mass_loss_fraction = 0.75  # insert value here

MASS_MOON = 7.34767309e22  # kg

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
    "SiO2": 41.95862216,
    'MgO': 33.02124749,
    'Al2O3': 3.669027499,
    'TiO2': 0.159931968,
    'Fe2O3': 0.0,
    'FeO': 18.03561908,
    'CaO': 3.10456173,
    'Na2O': 0.047038814,
    'K2O': 0.003763105,
    'ZnO': 0.000188155,
}

target_composition = bulk_moon_composition


# define a bunch of functions to calculate the bulk disk composition and Theia's bulk composition
def get_theia_composition(starting_composition, earth_composition, disk_mass, earth_mass):
    starting_weights = {oxide: starting_composition[oxide] / 100 * disk_mass for oxide in starting_composition.keys()}
    earth_composition = normalize(earth_composition)
    bse_weights = {oxide: earth_composition[oxide] / 100 * earth_mass for oxide in earth_composition.keys()}
    theia_weights = {oxide: starting_weights[oxide] - bse_weights[oxide] for oxide in starting_weights.keys()}
    theia_weight_pct = {oxide: theia_weights[oxide] / sum(theia_weights.values()) * 100 for oxide in
                        theia_weights.keys()}
    theia_moles = ConvertComposition().mass_to_moles(theia_weights)
    theia_cations = ConvertComposition().oxide_to_cations(theia_moles)
    theia_x_si = {cation: theia_cations[cation] / theia_cations['Si'] for cation in theia_cations.keys()}
    theia_x_al = {cation: theia_cations[cation] / theia_cations['Al'] for cation in theia_cations.keys()}
    return theia_weight_pct, theia_moles, theia_cations, theia_x_si, theia_x_al


def read_composition_file(file_path: str, metadata_rows=3):
    metadata = {}
    data = {}
    with open(file_path, 'r') as infile:
        reader = list(csv.reader(infile))
        for j in range(0, metadata_rows):
            line = reader[j]
            try:  # try to convert to float
                metadata.update({line[0]: float(line[1])})
            except:  # probably a string
                metadata.update({line[0]: line[1]})
        for j in range(metadata_rows, len(reader)):
            line = reader[j]
            data.update({line[0]: float(line[1])})
    return metadata, data


def find_disk_composition(run):
    """
    Find the starting disk composition that reproduces the BSM for each run.
    :return:
    """
    """
    We will iteratively solve for a disk composition that gives back the bulk Moon composition.
    We will use the BSE as an initial guess, and then iteratively adjust the disk composition.
    """
    temperature, vmf, theia_mass_fraction, earth_mass_fraction, disk_mass = runs[run].values()
    to_dir = run
    starting_comp_filename = f"{run}_starting_composition.csv"
    theia_silicate_comp_filename = f"{run}_theia_silicate_composition.csv"
    theia_bulk_comp_filename = f"{run}_theia_composition.csv"
    starting_composition = run_monte_carlo(initial_composition=bse_composition,
                                           target_composition=target_composition,
                                           temperature=temperature,
                                           vmf=vmf, full_report_path=to_dir, full_run_vmf=90.0,
                                           starting_comp_filename=starting_comp_filename)
    disk_bulk_composition_metadata, disk_bulk_composition = read_composition_file(to_dir + "/" + starting_comp_filename)
    theia_weight_pct, theia_moles, theia_cations, theia_x_si, theia_x_al = get_theia_composition(
        disk_bulk_composition, bse_composition, disk_mass * MASS_MOON,
                                                disk_mass * MASS_MOON * earth_mass_fraction / 100
    )
    write_file(data=theia_weight_pct, filename=theia_bulk_comp_filename, metadata={'vmf': vmf}, to_path=to_dir)
    write_file(data=theia_weight_pct, filename=theia_silicate_comp_filename, metadata={'vmf': vmf}, to_path=to_dir)

def get_vapor_composition_at_vmf(run):
    """
    For each run, interpolate the vapor composition at the given VMF value.
    :return:
    """
    temperature, vmf, theia_mass_fraction, earth_mass_fraction, disk_mass = runs[run].values()
    vapor_composition_species = interpolate_composition_at_vmf(run, vmf, subdir="atmosphere_mole_fraction")
    return vapor_composition_species

def return_vmf_and_element_lists(data):
    """
    Returns a list of VMF values and a dictionary of lists of element values.
    :param data:
    :return:
    """
    vmf_list = list(sorted(data.keys()))
    elements_at_vmf = {element: [data[vmf][element] for vmf in vmf_list] for element in data[vmf_list[0]].keys()}
    return vmf_list, elements_at_vmf


run = '500b073S-no-circ-vmf'
# run MAGMA to find the disk composition that reproduces the bulk Moon composition
find_disk_composition(run)

# plot the magma composition as a function of VMF
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
ax.set_title(run)
ax.set_xlabel("VMF (%)")
ax.set_ylabel("Oxide Abundance (wt%)")
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 10
ax.set_title(run)
data = collect_data(path=f"{run}/magma_oxide_mass_fraction", x_header='mass fraction vaporized')
vmf_list, elements = return_vmf_and_element_lists(data)
for index2, oxide in enumerate(elements):
    color = color_cycle[index2]
    ax.plot(np.array(vmf_list) * 100, np.array(elements[oxide]) * 100, color=color)
    # ax.scatter(runs[run]['vmf'], interpolated_elements[oxide] * 100, color=color, s=200, marker='x')
    ax.axhline(target_composition[oxide], color=color, linewidth=2.0, linestyle='--')
    ax.scatter([], [], color=color, marker='s', label="{} (MAGMA)".format(oxide))
ax.plot([], [], color='k', linestyle="--", label="Moon")
ax.axvline(runs[run]['vmf'], linewidth=2.0, color='k', label="Predicted VMF")

ax.legend(loc='upper right')

plt.show()

# get the vapor composition at the given VMF
vapor_comp_mole_fraction = {key: value / 100 for key, value in normalize({key: value for key, value in
                                                                          get_vapor_composition_at_vmf(run).items() if
                                                                          "_l" not in key}).items()}

# get the vapor mole fraction at all VMFs
vapor_comp_mole_fraction_all_vmf = collect_data(path=f"{run}/atmosphere_mole_fraction",
                                                x_header='mass fraction vaporized')
vapor_comp_mole_fraction_all_vmf = {key: normalize({i: j for i, j in value.items() if "_l" not in i})
                                    for key, value in vapor_comp_mole_fraction_all_vmf.items()}

# convert to weight percent
vapor_comp_weight_pct = mole_fraction_to_weight_percent(vapor_comp_mole_fraction)
vapor_comp_weight_pct_all_vmf = {vmf: mole_fraction_to_weight_percent(vapor_comp_mole_fraction_all_vmf[vmf])
                                 for vmf in vapor_comp_mole_fraction_all_vmf}

# take this and insert into the hydrodynamics code, will calculate mean atmosphere molecular mass there
print(f"Vapor Composition (wt%) at VMF {runs[run]['vmf']}: ", vapor_comp_weight_pct)

# get the mean molecular mass of the atmosphere as a function of VMF
mean_molecular_mass = [get_mean_molecular_mass(vapor_comp_weight_pct_all_vmf[vmf]) for vmf in
                       vapor_comp_weight_pct_all_vmf.keys()]
# plot it
plt.plot(np.array(list(vapor_comp_mole_fraction_all_vmf.keys())) * 100.0, np.array(mean_molecular_mass) / 1000,
         linewidth=2.0)
# plot a vertical line at the given VMF
plt.axvline(x=runs[run]['vmf'], color='red', linestyle='--', linewidth=2.0)
plt.xlabel("VMF (%)")
plt.ylabel("Mean Molecular Mass (kg/mol)")
plt.title("Mean Molecular Mass of Vapor as a Function of VMF")
plt.grid()
plt.show()

# plot the vapor composition at all VMFs and the vapor composition at the given VMF
fig, ax = plt.subplots(figsize=(16, 9))
# get the color cycle for the matplotlib plot as a list
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 20
all_species = list(vapor_comp_weight_pct.keys())
for i, species in enumerate(all_species):
    ax.plot([vmf * 100.0 for vmf in vapor_comp_weight_pct_all_vmf.keys()],
            [vapor_comp_weight_pct_all_vmf[vmf][species] for vmf in vapor_comp_weight_pct_all_vmf.keys()],
            color=color_cycle[i], label=species)
# plot the given VMF as a vertical line
ax.axvline(x=runs[run]['vmf'], color='k', linestyle='--')
# plot the vapor composition at the given VMF as a scatter plot
for j, species in enumerate(vapor_comp_weight_pct.keys()):
    index = all_species.index(species)
    ax.scatter(runs[run]['vmf'], vapor_comp_weight_pct[species], color=color_cycle[index], label=species)
    # annotate next to the scattered point
    ax.annotate(species, (runs[run]['vmf'] + 0.5, vapor_comp_weight_pct[species]), color=color_cycle[index])
# convert y-axis to log scale
ax.set_yscale('log')
ax.set_xlabel('Vapor mass fraction')
ax.set_ylabel('Weight percent')
ax.set_title(f"Vapor composition at all VMFs and at VMF = {runs[run]['vmf']}")
ax.grid()
plt.show()

# now, run the hydrodynamics code

# we want to get the absolute masses in the vapor
disk_vapor_mass = runs[run]['disk_mass'] * runs[run]['vmf'] / 100.0
# find the pre-loss atmosphere species masses
original_atmosphere_component_masses = {
    species: disk_vapor_mass * MASS_MOON * vapor_comp_weight_pct[species] / 100.0 for species in
    vapor_comp_weight_pct.keys()
}
# find the post-loss atmosphere species masses
post_loss_atmosphere_component_masses = {
    species: atmosphere_mass_loss_fraction * original_atmosphere_component_masses[species]
    for species in original_atmosphere_component_masses.keys()
}

# plot the pre-loss and post-loss atmosphere species masses
fig, ax = plt.subplots(figsize=(16, 9))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 20
for index, species in enumerate(original_atmosphere_component_masses.keys()):
    ax.scatter(
        index + 1, original_atmosphere_component_masses[species], color=colors[all_species.index(species)],
        s=100, marker="*", label=species,
    )
    ax.scatter(
        index + 1, post_loss_atmosphere_component_masses[species], color=colors[all_species.index(species)],
        s=100, label=species,
    )
    # draw an arrow connecting the edges of the scatter point to show the mass loss
    ax.arrow(
        index + 1, original_atmosphere_component_masses[species], 0,
        post_loss_atmosphere_component_masses[species] - original_atmosphere_component_masses[species],
        color=colors[all_species.index(species)], head_width=0.4,
        head_length=abs(original_atmosphere_component_masses[species] -
                        post_loss_atmosphere_component_masses[species]) * 0.1,
        length_includes_head=True,
    )
    # get the order of magnitude of the largest y value
    y_max = max(original_atmosphere_component_masses[species], post_loss_atmosphere_component_masses[species])
    # annotate the species name next to the arrow
    ax.annotate(species, (index + 0.5, post_loss_atmosphere_component_masses[species] - 0.14 * y_max),
                color=colors[all_species.index(species)])
ax.grid()
ax.set_xlabel('Species')
ax.set_ylabel('Mass (kg)')
ax.set_title(f"Mass depletion of atmosphere species at VMF = {runs[run]['vmf']}")
# log y scale
ax.set_yscale('log')
plt.show()
