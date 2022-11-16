from src.composition import Composition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data, make_figure

import sys
from math import log10
from numpy import nan
import matplotlib
import matplotlib.pyplot as plt

# font = {'size': 18}
# matplotlib.rc('font', **font)

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
    'K2O': 2.45000,
    'ZnO': 0.00000,
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

# plot to validate against Schaffer and Fegley 2009
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
            y = y_data[index] + 0.001
    return x, y

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Mass Fraction Vaporized")
ax.set_ylabel("Mole Fraction")
ax.set_title("Vapor Composition")
# ax.set_ylim(-3, 0)
species_to_plot = ["Na", "O", "SiO", "Mg_g", "O2", "Fe", "FeO_g", "SiO2_g", "FeO_g", "MgO", "K2O_g"]
data = collect_data(path="reports/atmosphere_mole_fraction", x_header='mass fraction vaporized')
for i in species_to_plot:
    x_data = [j for j in data.keys() if j <= 0.8]
    y_data = []
    tmp = [data[j][i] for j in data.keys() if j <= 0.8]
    for j in tmp:
        if j > 0:
            y_data.append(log10(j))
            # y_data.append(j)
        else:
            y_data.append(nan)
    ax.plot(
        x_data,
        y_data,
        linewidth=2.0,
        label=i
    )
    ax.annotate(i, get_annotation_location(species=i, x_data=x_data, y_data=y_data, target_x=0.175))
ax.grid()
# ax.legend()
plt.tight_layout()
plt.show()
