from src.composition import Composition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data, make_figure

import numpy as np
import matplotlib.pyplot as plt

"""
This script shows how to run MAGMApy given a temperature path path.
"""

start_temperature = 4000  # K
end_temperature = 1800  # K
temperature_increment = 100  # K

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
for temperature in reversed(np.arange(end_temperature, start_temperature - temperature_increment, temperature_increment)):
    l.calculate_activities(temperature=temperature)
    g.calculate_pressures(temperature=temperature, liquid_system=l)
    if l.counter == 1:
        l.calculate_activities(temperature=temperature)
        g.calculate_pressures(temperature=temperature, liquid_system=l)
    t.vaporize_thermal()
    l.counter = 0  # reset Fe2O3 counter for next vaporization step
    print("[~] At iteration: {} (Weight Fraction Vaporized: {} %) (temperature: {} K)".format(count,
                                                                                              t.weight_fraction_vaporized * 100.0,
                                                                                              temperature))
    if count % 5 == 0 or count == 1:
        reports.create_composition_report(iteration=count)
        reports.create_liquid_report(iteration=count)
        reports.create_gas_report(iteration=count)
    count += 1


def get_annotation_location(species, x_data, y_data, target_x):
    if species == "MgO":
        target_x = 0.70
    elif species == "Na":
        target_x = 0.22
    min_diff = 10 * 10 ** 10
    x = None
    y = None
    for index, i in enumerate(x_data):
        diff = abs(i - target_x)
        if diff < min_diff:
            min_diff = diff
            x = i
            y = y_data[index] + .001
    return x, y

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Mole Fraction")
ax.set_title("Vapor Composition")
# ax.set_ylim(-3, 0)
species_to_plot = ["Na", "O", "SiO", "Mg_g", "O2", "Fe", "FeO_g", "SiO2_g", "MgO", "K2O_g", "ZnO_g"]
data = collect_data(path="reports/atmosphere_mole_fraction", x_header='temperature (K)')
for i in species_to_plot:
    x_data = [j for j in data.keys()]
    y_data = []
    tmp = [data[j][i] for j in data.keys()]
    for j in tmp:
        if j > 0:
            # y_data.append(log10(j))
            y_data.append(j)
        else:
            y_data.append(np.nan)
    ax.plot(
        x_data,
        y_data,
        linewidth=2.0,
        label=i
    )
    # ax.annotate(i, get_annotation_location(species=i, x_data=x_data, y_data=y_data, target_x=1850))
ax.grid()
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.show()