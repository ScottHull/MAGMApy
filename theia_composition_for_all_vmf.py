from src.composition import Composition, ConvertComposition
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

"""
Takes the MAMGApy code and uses it to run a Monte Carlo search for the composition of Theia.
"""

bse_composition = {
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

bsm_composition = {
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

# bsm_composition = {  # including core Fe as FeO
#     "SiO2": 41.95862216,
#     'MgO': 33.02124749,
#     'Al2O3': 3.669027499,
#     'TiO2': 0.159931968,
#     'Fe2O3': 0.0,
#     'FeO': 18.03561908,
#     'CaO': 3.10456173,
#     'Na2O': 0.047038814,
#     'K2O': 0.003763105,
#     'ZnO': 0.000188155,
# }


def renormalize_composition(oxide_masses: dict):
    """
    Normalizes the dictionary to 100%.
    :param oxide_masses:
    :return:
    """
    total_mass = sum(oxide_masses.values())
    return {oxide: oxide_masses[oxide] / total_mass * 100 for oxide in oxide_masses.keys()}


def get_oxide_masses(oxide_wt_pct: dict, mass: float):
    """
    Takes the total mass and a dictionary of oxide weight percents and returns the absolute oxide masses.
    :param oxide_wt_pct:
    :return:
    """
    oxide_masses = {}
    for oxide in oxide_wt_pct.keys():
        oxide_masses[oxide] = oxide_wt_pct[oxide] * mass / 100
    return oxide_masses


def interpolate_elements_at_vmf(at_vmf, at_composition, previous_vmf, previous_composition, target_vmf):
    """
    Interpolates composition at target VMF.
    :return:
    """

    vmfs = [previous_vmf, at_vmf]
    compositions = {oxide: [previous_composition[oxide], at_composition[oxide]] for oxide in
                    previous_composition.keys()}
    interpolated_elements = {}
    for oxide in compositions.keys():
        interp = interp1d(vmfs, compositions[oxide])
        interpolated_elements[oxide] = interp(target_vmf)
    return renormalize_composition(interpolated_elements)


def run_full_MAGMApy(composition, temperature, to_vmf=90):
    major_gas_species = [
        "SiO", "O2", "MgO", "Fe", "Ca", "Al", "Ti", "Na", "K", "ZnO", "Zn"
    ]

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

    reports = Report(composition=c, liquid_system=l, gas_system=g, thermosystem=t)
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
                                                                              round(t.weight_fraction_vaporized * 100.0,
                                                                                    4)))
        if count % output_interval == 0 or count == 1:
            reports.create_composition_report(iteration=count)
            reports.create_liquid_report(iteration=count)
            reports.create_gas_report(iteration=count)
        count += 1
    return c, l, g, t


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


def find_best_fit_vmf(vmfs: list, composition: dict, target_composition: dict, restricted_composition=None):
    """
    Find the vmf with the lowest residual error between all composition.
    :param target_composition: given in wt% (assumes composition is normalized to 100)
    :param restricted_composition: Do not include these composition in the fit.
    :param vmfs:
    :param composition: wt% composition of the liquid, assumes normaalized to 1
    :return:
    """
    if restricted_composition is None:
        restricted_composition = []
    best_vmf = vmfs[0]
    best_error = np.inf
    for vmf in vmfs:
        error = 0
        for element in composition.keys():
            if element not in restricted_composition:
                error += ((target_composition[element] / 100) - composition[element][vmfs.index(vmf)]) ** 2
        if error < best_error:
            best_error = error
            best_vmf = vmf
    return best_vmf * 100


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


def get_theia_composition(starting_composition, earth_composition, disk_mass, earth_mass):
    starting_weights = {oxide: starting_composition[oxide] / 100 * disk_mass for oxide in starting_composition.keys()}
    earth_composition = renormalize_composition(earth_composition)
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

# SPH PARAMETERS
runs = {
    # # b = 0.73
    # '5b073S': {
    #     "temperature": 6345.79,  # K
    #     "vmf": 35.73,  # %
    #     'theia_pct': 63.13,  # %
    #     'earth_pct': 100 - 63.13,  # %
    #     'disk_mass': 1.3,  # M_L
    # },
    '500b073S': {
        "temperature": 3664.25,
        "vmf": 19.21,
        'theia_pct': 66.78,  # %
        'earth_pct': 100 - 66.78,  # %
        'disk_mass': 1.02,  # M_L
    },
    # '1000b073S': {
    #     "temperature": 3465.15,
    #     "vmf": 10.75,
    #     'theia_pct': 69.79,  # %
    #     'earth_pct': 100 - 69.79,  # %
    #     'disk_mass': 1.45,  # M_L
    # },
    # '2000b073S': {
    #     "temperature": 3444.84,
    #     "vmf": 8.57,
    #     'theia_pct': 69.4,  # %
    #     'earth_pct': 100 - 69.4,  # %
    #     'disk_mass': 1.85,  # M_L
    # },
    # '5b073N': {
    #     "temperature": 6004.06,
    #     "vmf": 25.7,
    #     'theia_pct': 69.97,  # %
    #     'earth_pct': 100 - 69.97,  # %
    #     'disk_mass': 1.07,  # M_L
    # },
    # '500b073N': {
    #     "temperature": 6637.25,
    #     "vmf": 27.34,
    #     'theia_pct': 71.7,  # %
    #     'earth_pct': 100 - 71.7,  # %
    #     'disk_mass': 0.83,  # M_L
    # },
    # '1000b073N': {
    #     "temperature": 6280.91,
    #     "vmf": 29.53,
    #     'theia_pct': 74.1,  # %
    #     'earth_pct': 100 - 74.1,  # %
    #     'disk_mass': 0.87,  # M_L
    # },
    # '2000b073N': {
    #     "temperature": 4342.08,
    #     "vmf": 10.61,
    #     'theia_pct': 72.95,  # %
    #     'earth_pct': 100 - 72.95,  # %
    #     'disk_mass': 1.08,  # M_L
    # },
    #
    # # b = 0.75
    # '5b075S': {
    #     "temperature": 8536.22,
    #     "vmf": 67.98,
    #     'theia_pct': 60.32,  # %
    #     'earth_pct': 100 - 60.32,  # %
    #     'disk_mass': 0.47,  # M_L
    # },
    # '500b075S': {
    #     "temperature": 6554.56,
    #     "vmf": 39.77,
    #     'theia_pct': 67.9,  # %
    #     'earth_pct': 100 - 67.9,  # %
    #     'disk_mass': 0.78,  # M_L
    # },
    # '1000b075S': {
    #     "temperature": 6325.06,
    #     "vmf": 42.97,
    #     'theia_pct': 61.42,  # %
    #     'earth_pct': 100 - 61.42,  # %
    #     'disk_mass': 0.26,  # M_L
    # },
    # '2000b075S': {
    #     "temperature": 4882.66,
    #     "vmf": 28.67,
    #     'theia_pct': 71.14,  # %
    #     'earth_pct': 100 - 71.14,  # %
    #     'disk_mass': 0.83,  # M_L
    # },
    # '5b075N': {
    #     "temperature": 9504.66,
    #     "vmf": 78.25,
    #     'theia_pct': 56.68,  # %
    #     'earth_pct': 100 - 56.68,  # %
    #     'disk_mass': 0.48,  # M_L
    # },
    # '500b075N': {
    #     "temperature": 6970.22,
    #     "vmf": 46.72,
    #     'theia_pct': 71.91,  # %
    #     'earth_pct': 100 - 71.91,  # %
    #     'disk_mass': 0.95,  # M_L
    # },
    # '1000b075N': {
    #     "temperature": 6872.69,
    #     "vmf": 40.77,
    #     'theia_pct': 69.31,  # %
    #     'earth_pct': 100 - 69.31,  # %
    #     'disk_mass': 0.73,  # M_L
    # },
    # '2000b075N': {
    #     "temperature": 6911.39,
    #     "vmf": 37.78,
    #     'theia_pct': 69.19,  # %
    #     'earth_pct': 100 - 69.19,  # %
    #     'disk_mass': 0.74,  # M_L
    # },
}

MASS_MOON = 7.34767309e22  # kg
disk_mass = 1.0  # M_L
earth_pct = 100.0  # %
earth_mass_fraction = 30.0  # %

def run_isotherm(args):
    temperature, to_dir = args
    for vmf in np.arange(10, 100, 10):
        starting_comp_to_dir = to_dir + "/disk_starting_comp"
        try:
            if not os.path.exists(starting_comp_to_dir):
                os.mkdir(starting_comp_to_dir)
        except FileExistsError:
            pass
        starting_composition = run_monte_carlo(initial_composition=bse_composition,
                                               target_composition=bsm_composition, temperature=temperature,
                                               vmf=vmf, full_report_path=starting_comp_to_dir, full_run_vmf=None,
                                            starting_comp_filename="{}_{}_starting_comp.csv".format(temperature, vmf),
                                               delete_dir=False)
        try:
            disk_bulk_composition_metadata, disk_bulk_composition = read_composition_file(
                starting_comp_to_dir + "/{}_{}_starting_comp.csv".format(temperature, vmf))
            theia_weight_pct, theia_moles, theia_cations, theia_x_si, theia_x_al = get_theia_composition(
                disk_bulk_composition, bse_composition, disk_mass * MASS_MOON,
                                                        disk_mass * MASS_MOON * earth_mass_fraction / 100
            )
            metadata = {
                "temperature": temperature,
                "vmf": vmf,
                "mg/si": theia_x_si['Mg'],
                "mg/al": theia_x_al['Mg'],
            }
            theia_to_dir = to_dir + "/silicate_theia_composition"
            if not os.path.exists(theia_to_dir):
                os.mkdir(theia_to_dir)
            write_file(data=theia_weight_pct, metadata=metadata, filename="{}_{}_silicate_theia_composition.csv".format(temperature, vmf), to_path=theia_to_dir)
        except FileNotFoundError:
            pass

to_dir = "isotherm_initial_composition_solutions"
if not os.path.exists(to_dir):
    os.mkdir(to_dir)
pool = mp.Pool(10)
pool.map(run_isotherm, [[temperature, to_dir] for temperature in [3500, 5000, 8000, 10000]])
pool.close()
pool.join()
