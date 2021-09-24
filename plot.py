import matplotlib.pyplot as plt
from math import log10
from numpy import nan

from src.plots import collect_data, make_figure

species_to_plot = ["Na", "O", "SiO", "Mg_g", "O2", "Fe", "FeO_g", "SiO2_g", "FeO_g", "MgO"]
data = collect_data(path="reports/atmosphere_mole_fraction", x_header='mass fraction vaporized')


def median(lst):
    lst.sort()  # Sort the list first
    if len(lst) % 2 == 0:  # Checking if the length is even
        # Applying formula which is sum of middle two divided by 2
        return (lst[len(lst) // 2] + lst[(len(lst) - 1) // 2]) / 2
    else:
        # If length is odd then get middle value
        return lst[len(lst) // 2]


def get_annotation_location(x_data, y_data, x_range, y_range):
    z = zip(x_data, y_data)
    p = []
    for i in z:
        if (x_range[0] <= i[0] <= x_range[1]) and (y_range[0] <= i[1] <= y_range[1]):
            p.append(i)
    m = median(lst=[i[0] for i in p])
    index = [i[0] for i in p].index(m)
    return p[index]


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
