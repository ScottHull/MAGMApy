import os
import ast
import numpy as np
import warnings
import pandas as pd
import copy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import labellines
from concurrent.futures import ThreadPoolExecutor, as_completed

# use colorblind-friendly colors
plt.style.use('seaborn-colorblind')
# increase font size
plt.rcParams.update({"font.size": 20})
# turn off all double scaling warnings

runs = [
    {
        "run_name": "Canonical Model 2",
        "temperature": 2657.97,  # K
        "vmf": 3.80,  # %
        "0% VMF mass frac": 87.41,  # %
        "100% VMF mass frac": 0.66,  # %
        "disk_theia_mass_fraction": 66.78,  # %
        "disk_mass": 1.02,  # lunar masses
        "vapor_loss_fraction": 0.74,  # %
        "new_simulation": True,  # True to run a new simulation, False to load a previous simulation
    },
    {
        "run_name": "Half Earths Model 2",
        "temperature": 3514.15,  # K
        "vmf": 14.50,  # %
        "0% VMF mass frac": 81.3,  # %
        "100% VMF mass frac": 1.9,  # %
        "disk_theia_mass_fraction": 51.97,  # %
        "disk_mass": 1.70,  # lunar masses
        "vapor_loss_fraction": 16.0,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    }
]

# order composition by volatility
oxides_ordered = [
    "Al2O3", "TiO2", "CaO", "MgO", "FeO", "SiO2", "K2O", "Na2O", "ZnO"
]
elements_ordered = ["Al", "Ti", "Ca", "Mg", "Fe", "Si", "K", "Na", "Zn", "O"]

lunar_bulk_compositions = pd.read_csv("data/lunar_bulk_compositions.csv", index_col="Oxide")

# make a 2 column 2 row plot
fig, axs = plt.subplots(2, 2, figsize=(16, 16))

for run in runs:
    for lbc in lunar_bulk_compositions.columns:
        for recondense in ['no_recondensation', 'full_recondensation']:
            run_name = f"{run['run_name']}_{lbc}_{recondense}_theia_mixing_model"
            fname = f"{run_name}.csv"
            if os.path.exists(fname):
                print(f"Loading {fname}")
                data = ast.literal_eval(open(fname, "r").read())
                print(data.keys())

                # check bulk mass balance
                assert np.isclose(data['total_ejecta_mass'], data['bse_sourced_mass'] + data['theia_sourced_mass'],
                                  sum(data['total_ejecta_element_mass_before_vaporization'].values()))
                # check vapor mass balance (without recondensation)
                assert np.isclose(
                    sum(data['total_ejecta_mass_after_vapor_removal_without_recondensation'].values()) + sum(
                        data['total_vapor_mass'].values()), data['total_ejecta_mass'])

                axs[0, 0].plot(oxides_ordered, [data['bulk_ejecta_composition'][oxide] for oxide in oxides_ordered], linewidth=3.0, label=f"{lbc}")
                axs[0, 1].plot(oxides_ordered, [data['theia_composition'][oxide] for oxide in oxides_ordered], linewidth=3.0, label=f"{lbc}")

axs[0, 0].legend()

for ax in axs.flatten():
    ax.grid()

plt.tight_layout()
plt.show()
