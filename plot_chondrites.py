import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt

from src.composition import ConvertComposition
from theia.chondrites import plot_chondrites


path = "data/Chondrite MgSi vs AlSi.txt"
df = pd.read_csv(path, delimiter='\t')
df = df[df['Include?'] == "Y"]
# get all unique values of the type column
types = df["General Type"].unique()

lunar_bulk_compositions = pd.read_csv("data/lunar_bulk_compositions.csv", index_col="Oxide")


def get_ellipse_params(points, ax, scale=1.5, **kwargs):
    '''
    Calculate the parameters needed to graph an ellipse around a cluster of points in 2D.

        Calculate the height, width and angle of an ellipse to enclose the points in a cluster.
        Calculate the width by finding the maximum distance between the x-coordinates of points
        in the cluster, and the height by finding the maximum distance between the y-coordinates
        in the cluster. Multiple both by a scale factor to give padding around the points when
        constructing the ellipse. Calculate the angle by taking the inverse tangent of the
        gradient of the regression line. Note that tangent solutions repeat every 180 degrees,
        and so to ensure the correct solution has been found for plotting, add a correction
        factor of +/- 90 degrees if the magnitude of the angle exceeds 45 degrees.

        Args:
            points (ndarray): The points in a cluster to enclose with an ellipse, containing n
                              ndarray elements representing each point, each with d elements
                              representing the coordinates for the point.

        Returns:
            width (float):  The width of the ellipse.
            height (float): The height of the ellipse.
            angle (float):  The angle of the ellipse in degrees.
    '''
    if points.ndim == 1:
        width, height, angle = 0.1, 0.1, 0
        return width, height, angle

    else:
        SCALE = scale
        width = np.amax(points[:, 0]) - np.amin(points[:, 0])
        height = np.amax(points[:, 1]) - np.amin(points[:, 1])

        # Calculate angle
        x_reg, y_reg = [[p[0]] for p in points], [[p[1]] for p in points]
        grad = LinearRegression().fit(x_reg, y_reg).coef_[0][0]
        angle = np.degrees(np.arctan(grad))

        # Account for multiple solutions of arctan
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90

        ellipse = Ellipse(xy=(np.mean(points[:, 0]), np.mean(points[:, 1])), width=width * SCALE, height=height * SCALE, angle=angle, **kwargs)
        ax.add_patch(ellipse)

        return width * SCALE, height * SCALE, angle

def sort_lunar_models(models):
    """
    Takes a list of strings and sorts them based on the year at the end of the string.
    :param models:
    :return:
    """
    years = [int(model.replace("Fractional Model", "").replace("Equilibrium Model", "").strip().split(" ")[-1]) for model in models]
    return [model for _, model in sorted(zip(years, models))]


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
# get a unique color for each type
colors = plt.cm.rainbow(np.linspace(0, 1, len(types)))
for index, t in enumerate(types):
    print(f"Plotting {t} (num points = {len(df[df['Type'] == t])})")
    # get the rows for each type
    rows = df[df["General Type"] == t]
    # rows = rows[rows["Type"] == "CO3"]
    # plot the rows
    ax.scatter(rows["Al/Si"], rows["Mg/Si"], color=colors[index], label=t)

    if len(rows) > 1:
        n = np.array(list(zip(rows["Al/Si"].values, rows["Mg/Si"].values)))
        width, height, angle = get_ellipse_params(n, ax, edgecolor='k', fill=True,
                                                  facecolor='grey', alpha=0.3, scale=1.2)
        # annotate the type on the outside edge of the ellipse
        ax.annotate(t, xy=(np.mean(rows["Al/Si"]), np.mean(rows["Mg/Si"])),
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
# increase font size
plt.rcParams.update({'font.size': 16})
ax.set_xlabel("Al/Si", fontsize=16)
ax.set_ylabel("Mg/Si", fontsize=16)
# ax.set_xlim(0, 0.2)
# ax.set_ylim(0, 1.4)
ax.legend()
ax.grid()
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
colors = sns.color_palette('husl', n_colors=len(lunar_bulk_compositions.keys()))
plot_chondrites(ax)
for index, model in enumerate(sort_lunar_models(lunar_bulk_compositions.keys())):
    composition = {
        oxide: lunar_bulk_compositions.loc[oxide, model] for oxide in lunar_bulk_compositions.index
    }
    composition = ConvertComposition().oxide_wt_to_cation_wt(composition)
    ax.scatter(composition["Al"] / composition['Si'], composition["Mg"] / composition['Si'], s=300,
               color=colors[index], edgecolor='k', label=model)
plt.rcParams.update({'font.size': 16})
ax.set_xlabel("Al/Si (mass ratio)", fontsize=16)
ax.set_ylabel("Mg/Si (mass ratio)", fontsize=16)
# ax.set_xlim(0, 0.2)
# ax.set_ylim(0, 1.4)
ax.legend(loc='lower right', fontsize=16)
ax.grid()
plt.tight_layout()
plt.savefig("lunar_models_mg_si_vs_al_si.png", dpi=300)
plt.show()
