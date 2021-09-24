import matplotlib.pyplot as plt
from math import log10
from numpy import nan

from src.plots import collect_data, make_figure

species_to_plot = ["Na", "O", "SiO", "Mg_g", "O2", "Fe", "FeO_g", "SiO2_g", "FeO_g", "MgO"]
data = collect_data(path="reports/atmosphere_mole_fraction", x_header='mass fraction vaporized')

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
    middle_index = int(len(x_data) / 2)
    annotate_x_pos = x_data[middle_index]
    annotate_y_pos = y_data[middle_index] + ((max(x_data) - min(x_data)) * 0.05)
    ax.annotate(i, (annotate_x_pos, annotate_y_pos))
ax.grid()
# ax.legend()
plt.show()
