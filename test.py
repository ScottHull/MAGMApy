from src.composition import Composition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data, make_figure

import sys
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 18}
matplotlib.rc('font', **font)

temperature = 2200

composition = {
    "SiO2": 62.93000,
    'MgO': 3.79000,
    'Al2O3': 15.45000,
    'TiO2': 0.70000,
    'Fe2O3': 0.00000,
    'FeO': 5.78000,
    'CaO': 5.63000,
    'Na2O': 3.27000,
    'K2O': 2.45000
}

major_gas_species = [
    "SiO", "O2", "MgO", "Fe", "Ca", "Al", "Ti", "Na", "K"
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
while count < 5001:
    if count > 1:
        prev_fraction = t.atomic_fraction_vaporized
    l.calculate_activities(temperature=temperature)
    g.calculate_pressures(temperature=temperature, liquid_system=l)
    if l.counter == 1:
        l.calculate_activities(temperature=temperature)
        g.calculate_pressures(temperature=temperature, liquid_system=l)
    t.vaporize()
    l.counter = 0  # reset Fe2O3 counter for next vaporizaiton step
    print("[~] At iteration: {} (Magma Fraction Vaporized: {} %)".format(count, t.atomic_fraction_vaporized * 100.0))
    if count % 20 == 0 or count == 1:
        reports.create_composition_report(iteration=count)
        reports.create_liquid_report(iteration=count)
        reports.create_gas_report(iteration=count)
    count += 1
