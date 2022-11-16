from src.composition import Composition, ConvertComposition, normalize, interpolate_composition_at_vmf, \
    mole_fraction_to_weight_percent
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

# SPH parameters at peak shock conditions
runs = {
    '500b073S': {
        "temperature": 3063.18893,
        "vmf": 19.21,
        'theia_pct': 66.78,  # %
        'earth_pct': 100 - 66.78,  # %
        'disk_mass': 1.02,  # M_L
    },
}

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
    starting_composition = run_monte_carlo(initial_composition=bse_composition,
                                           target_composition=bulk_moon_composition,
                                           temperature=temperature,
                                           vmf=vmf, full_report_path=to_dir, full_run_vmf=90.0,
                                           starting_comp_filename=starting_comp_filename)
    disk_bulk_composition_metadata, disk_bulk_composition = read_composition_file(to_dir + "/" + starting_comp_filename)
    theia_weight_pct, theia_moles, theia_cations, theia_x_si, theia_x_al = get_theia_composition(
        disk_bulk_composition, bse_composition, disk_mass * MASS_MOON,
                                                disk_mass * MASS_MOON * earth_mass_fraction / 100
    )


def get_vapor_composition_at_vmf(run):
    """
    For each run, interpolate the vapor composition at the given VMF value.
    :return:
    """
    temperature, vmf, theia_mass_fraction, earth_mass_fraction, disk_mass = runs[run].values()
    vapor_composition_species = interpolate_composition_at_vmf(run, vmf, subdir="atmosphere_mole_fraction")
    return vapor_composition_species


run = '500b073S'
# run MAGMA to find the disk composition that reproduces the bulk Moon composition
# find_disk_composition(run)

# get the vapor composition at the given VMF
vapor_comp_mole_fraction = {key: value / 100 for key, value in normalize({key: value for key, value in
                                get_vapor_composition_at_vmf(run).items() if "_l" not in key}).items()}

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
print(vapor_comp_weight_pct)

# plot the vapor composition at all VMFs and the vapor composition at the given VMF
fig, ax = plt.subplots(figsize=(16, 9))
# get the color cycle for the matplotlib plot as a list
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 20
all_species = list(vapor_comp_weight_pct.keys())
for i in vapor_comp_weight_pct_all_vmf.keys():
    print(sum(vapor_comp_weight_pct_all_vmf[i].values()), vapor_comp_weight_pct_all_vmf[i])
for i, species in enumerate(all_species):
    ax.plot([vmf * 100.0 for vmf in vapor_comp_weight_pct_all_vmf.keys()],
            [vapor_comp_weight_pct_all_vmf[vmf][species] for vmf in vapor_comp_weight_pct_all_vmf.keys()],
            color=color_cycle[i], label=species)
    # annotate the line at 10% VMF
    ax.annotate(species, (10, vapor_comp_weight_pct_all_vmf[next(iter(vapor_comp_weight_pct_all_vmf), 10)][species]), color=color_cycle[i])
# plot the given VMF as a vertical line
ax.axvline(x=runs[run]['vmf'], color='k', linestyle='--')
# plot the vapor composition at the given VMF as a scatter plot
for j, species in enumerate(vapor_comp_weight_pct.keys()):
    index = all_species.index(species)
    ax.scatter(runs[run]['vmf'], vapor_comp_weight_pct[species], color=color_cycle[index], label=species)
    # annotate next to the scattered point
    ax.annotate(species, (runs[run]['vmf'] + 0.5, vapor_comp_weight_pct[species]), color=color_cycle[index])
ax.set_xlabel('Vapor mass fraction')
ax.set_ylabel('Weight percent')
ax.set_title(f"Vapor composition at all VMFs and at VMF = {runs[run]['vmf']}")
ax.grid()
plt.show()

# now, run the hydrodynamics code

# come back with atmosphere mass loss fraction from hydrodynamics code
atmosphere_mass_loss_fraction = None  # insert value here
