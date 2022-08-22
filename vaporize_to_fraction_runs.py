from src.composition import Composition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data, make_figure

from scipy.interpolate import interp1d

"""
This script shows how to run MAGMApy to a given vapor mass fraction (VMF) along an isotherm.
"""

runs = {
    # b = 0.73
    '5b073S': {
        "temperature": 6345.79,  # K
        "vmf": 35.73,  # %
    },
    '500b073S': {
        "temperature": 3664.25,
        "vmf": 19.21,
    },
    '1000b073S': {
        "temperature": 3465.15,
        "vmf": 10.75,
    },
    '2000b073S': {
        "temperature": 3444.84,
        "vmf": 8.57,
    },
    '5b073N': {
        "temperature": 6004.06,
        "vmf": 25.7,
    },
    '500b073N': {
        "temperature": 6637.25,
        "vmf": 27.34,
    },
    '1000b073N': {
        "temperature": 6280.91,
        "vmf": 29.53,
    },
    '2000b073N': {
        "temperature": 4342.08,
        "vmf": 10.61,
    },

    # b = 0.75
    '5b075S': {
        "temperature": 8536.22,
        "vmf": 67.98,
    },
    '500b075S': {
        "temperature": 6554.56,
        "vmf": 39.77,
    },
    '1000b075S': {
        "temperature": 6325.06,
        "vmf": 42.97,
    },
    '2000b075S': {
        "temperature": 4882.66,
        "vmf": 28.67,
    },
    '5b075N': {
        "temperature": 9504.66,
        "vmf": 78.25,
    },
    '500b075N': {
        "temperature": 6970.22,
        "vmf": 46.72,
    },
    '1000b075N': {
        "temperature": 6872.69,
        "vmf": 40.77,
    },
    '2000b075N': {
        "temperature": 6911.39,
        "vmf": 37.78,
    },
}

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

to_vmf = 60  # %

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

for run in runs.keys():
    reports = Report(composition=c, liquid_system=l, gas_system=g, thermosystem=t, to_dir="{}_reports".format(run))
    temperature = runs[run]["temperature"]
    count = 1
    while t.weight_fraction_vaporized * 100 < to_vmf:
        output_interval = 50
        if t.weight_fraction_vaporized * 100.0 > 5:  # vmf changes very fast towards end of simulation
            output_interval = 5
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
                                                                                    2)))
        if count % output_interval == 0 or count == 1:
            reports.create_composition_report(iteration=count)
            reports.create_liquid_report(iteration=count)
            reports.create_gas_report(iteration=count)
        count += 1
    break
