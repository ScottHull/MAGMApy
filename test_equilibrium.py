from src.composition import Composition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data, make_figure

from random import uniform
import numpy as np
import matplotlib.pyplot as plt
import labellines

"""
This script shows how to run MAGMApy given a temperature path path.
"""

max_temperature = 4200  # K
min_temperature = 1800  # K
temperature_increment = 200  # K

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
for temperature in reversed(
        np.arange(min_temperature, max_temperature + temperature_increment, temperature_increment)):
    l.calculate_activities(temperature=temperature)
    g.calculate_pressures(temperature=temperature, liquid_system=l)
    if l.counter == 1:
        l.calculate_activities(temperature=temperature)
        g.calculate_pressures(temperature=temperature, liquid_system=l)
    t.vaporize()
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
ax.set_ylabel("Partial Pressure")
ax.set_title("Vapor Composition")
# ax.set_ylim(-3, 0)
data = collect_data(path="reports/partial_pressures", x_header='temperature (K)')
for i in data[list(data.keys())[0]]:
    if "_l" not in i:
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
            linewidth=1.0,
            label=i.split("_")[0]
        )
        # ax.annotate(i, get_annotation_location(species=i, x_data=x_data, y_data=y_data, target_x=1850))
labellines.labelLines(ax.get_lines(), zorder=2.5, align=True,
                              xvals=[uniform(3500, max_temperature) for i in ax.get_lines()], fontsize=8)
ax.set_xlim(min_temperature, max_temperature)
ax.set_ylim(10 ** -6, 100)
ax.grid()
ax.set_yscale('log')
ax.yaxis.tick_right()
plt.tight_layout()
plt.show()
