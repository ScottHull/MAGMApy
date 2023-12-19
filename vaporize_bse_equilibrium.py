from src.composition import Composition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import EquilibriumThermoSystem
from src.report import Report
from src.plots import collect_data, make_figure

from random import uniform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import labellines
from scipy.interpolate import interp1d


canonical = pd.read_csv("/Users/scotthull/Desktop/canonical_df2.csv")

temperature_increment = 200  # K
max_temperature = round(canonical['temperature'].max() + 2 * temperature_increment, -2)  # K
min_temperature = round(canonical['temperature'].min() - 2 * temperature_increment, -2)  # K
# round to nearest 100 K
title = "Canonical Impact"
new_simulation = False

# BSE composition, Visccher & Fegley 2013
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
    "SiO", "O2", "MgO", "Fe", "Ca", "Al", "Ti", "Na", "K", "Zn"
]
lunar_bulk_compositions = pd.read_csv("data/lunar_bulk_compositions.csv", index_col="Oxide")
ordered_oxides = [
    'Al2O3', 'TiO2', 'CaO', 'MgO', 'FeO', 'SiO2', 'K2O', 'Na2O', 'ZnO'
]
ordered_elements = [
    'Al', 'Ti', 'Ca', 'Mg', 'Fe', 'Si', 'K', 'Na', 'Zn'
]

if new_simulation:
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
    t = EquilibriumThermoSystem(composition=c, gas_system=g, liquid_system=l)
    reports = Report(composition=c, liquid_system=l, gas_system=g, thermosystem=t, to_dir=title)
    count = 1
    for temperature in reversed(
            np.arange(min_temperature, max_temperature + temperature_increment, temperature_increment)):
        print(f"Temperature: {temperature} K")
        l.calculate_activities(temperature=temperature)
        g.calculate_pressures(temperature=temperature, liquid_system=l)
        if l.counter == 1:
            l.calculate_activities(temperature=temperature)
            g.calculate_pressures(temperature=temperature, liquid_system=l)
        t.vaporize()
        l.counter = 0  # reset Fe2O3 counter for next vaporization step
        # if count % 5 == 0 or count == 1:
        reports.create_composition_report(iteration=count)
        reports.create_liquid_report(iteration=count)
        reports.create_gas_report(iteration=count)
        count += 1

def get_composition_at_temperature_func(d: dict):
    """
    Given a VMF, interpolate the composition of the d dictionary at that VMF.
    :param d:
    :param vmf_val:
    :return:
    """
    vmfs = list(d.keys())
    species = list(d[vmfs[0]].keys())
    interpolated_composition_funcs = {}
    for s in species:
        interpolated_composition_funcs[s] = interp1d(
            vmfs,
            [i[s] for i in d.values()]
        )
    return interpolated_composition_funcs

melt_oxide = collect_data(path=f"{title}/magma_oxide_mass_fraction", x_header='temperature (K)')
melt_element_mass = collect_data(path=f"{title}/magma_element_mass", x_header='temperature (K)')
bulk_vapor_element_mass = collect_data(path=f"{title}/total_vapor_element_mass", x_header='temperature (K)')
melt_oxide_funcs = get_composition_at_temperature_func(melt_oxide)
melt_element_mass_funcs = get_composition_at_temperature_func(melt_element_mass)
bulk_vapor_element_mass_funcs = get_composition_at_temperature_func(bulk_vapor_element_mass)
# use the temperature from the canonical model to get the melt composition at that temperature
for oxide in ordered_oxides:
    canonical[f"{oxide}_melt"] = melt_oxide_funcs[oxide](canonical["temperature"]) * canonical['vmf_wo_circ']
for element in ordered_elements:
    canonical[f"{element}_melt"] = melt_element_mass_funcs[element](canonical["temperature"])
    canonical[f"{element}_vapor"] = bulk_vapor_element_mass_funcs[element](canonical["temperature"])
