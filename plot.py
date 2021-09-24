import matplotlib.pyplot as plt
from math import log10
from numpy import nan
import numpy as np

from src.plots import collect_data, make_figure

species_to_plot = ["Na", "O", "SiO", "Mg_g", "O2", "Fe", "FeO_g", "SiO2_g", "FeO_g", "MgO"]
data = collect_data(path="reports/atmosphere_mole_fraction", x_header='mass fraction vaporized')


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
            y = y_data[index] + 0.05
    return x, y





fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Mass Fraction Vaporized")
ax.set_ylabel("log Mole Fraction")
ax.set_ylim(-3, 0)
for i in species_to_plot:
    x_data = [j for j in data.keys() if j <= 0.8]
    y_data = []
    tmp = [data[j][i] for j in data.keys() if j <= 0.8]
    for j in tmp:
        if j > 0:
            y_data.append(log10(j))
        else:
            y_data.append(nan)
    ax.plot(
        x_data,
        y_data,
        linewidth=2.0,
        label=i
    )
    ax.annotate(i, get_annotation_location(species=i, x_data=x_data, y_data=y_data, target_x=0.32))
ax.grid()
# ax.legend()
plt.show()
