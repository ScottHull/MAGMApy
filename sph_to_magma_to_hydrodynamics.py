from src.composition import Composition, ConvertComposition, normalize, interpolate_composition_at_vmf
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

def find_disk_composition():
    """
    Find the starting disk composition that reproduces the BSM for each run.
    :return:
    """
    for run in runs.keys():
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

def get_vapor_composition_at_vmf():
    """
    For each run, interpolate the vapor composition at the given VMF value.
    :return:
    """
    # TODO: make sure _l phases arent included in vapor mole fractions.
    for run in runs.keys():
        temperature, vmf, theia_mass_fraction, earth_mass_fraction, disk_mass = runs[run].values()
        vapor_composition_species = interpolate_composition_at_vmf(run, vmf, subdir="atmosphere_mole_fraction")
        print(
            run, vapor_composition_species
        )

get_vapor_composition_at_vmf()
