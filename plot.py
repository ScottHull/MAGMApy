import matplotlib.pyplot as plt
from math import log10
from numpy import nan
import numpy as np

from src.plots import collect_data, make_figure

species_to_plot = ["Na", "O", "SiO", "Mg_g", "O2", "Fe", "FeO_g", "SiO2_g", "FeO_g", "MgO"]
data = collect_data(path="reports/atmosphere_mole_fraction", x_header='mass fraction vaporized')


def get_annotation_location(x_data, y_data, x_range, y_range):
    z = zip(x_data, y_data)
    p = []
    for i in z:
        if (x_range[0] <= i[0] <= x_range[1]) and (y_range[0] <= i[1] <= y_range[1]):
            p.append(i)
    med_x = np.median([i[0] for i in p])
    x_pos = np.where(x_data == med_x)
    y_pos = y_data[x_pos]
    return x_pos, y_pos


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
    ax.annotate(i, get_annotation_location(x_data=x_data, y_data=y_data, x_range=(0, 0.8), y_range=(-3, 0)))
ax.grid()
# ax.legend()
plt.show()
