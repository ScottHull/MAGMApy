import copy

from monte_carlo.monte_carlo import run_monte_carlo_vapor_loss
from theia.theia import get_theia_composition, recondense_vapor
from monte_carlo.monte_carlo import theia_mixing, run_full_MAGMApy
from src.plots import collect_data, collect_metadata
from src.composition import normalize, get_molecular_mass
from isotopes.rayleigh import FullSequenceRayleighDistillation

import os
import sys
import string
from scipy.interpolate import interp1d
from random import uniform, shuffle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# use colorblind-friendly colors
plt.style.use('seaborn-colorblind')

RUN_NEW_SIMULATIONS = False
NUM_THREADS = 40
GATHER = True
# root_path = ""
root_path = "C:/Users/Scott/OneDrive/Desktop/vaporize_theia/"
# root_path = "/scratch/shull4/vaporize_theia/"

if RUN_NEW_SIMULATIONS:
    if not os.path.exists(root_path) and len(root_path) > 0:
        os.mkdir(root_path)

# ============================== Define Compositions ==============================

bse_composition = {  # Visscher and Fegley (2013)
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

oxides = [i for i in bse_composition.keys() if i != "Fe2O3"]
lunar_bulk_compositions = pd.read_csv("data/lunar_bulk_compositions.csv", index_col="Oxide")
favored_composition = [lunar_bulk_compositions["O'Neill 1991"].loc[oxide] for oxide in oxides]
mass_moon = 7.34767309e22  # kg, mass of the moon

# ============================== Define Runs ==============================

runs = [
    {
        "run_name": "Canonical Model",
        "temperature": 2682.61,  # K
        "vmf": 0.96,  # %
        "disk_theia_mass_fraction": 66.78,  # %
        "disk_mass": 1.02,  # lunar masses
        "vapor_loss_fraction": 74.0,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    },
    {
        "run_name": "Half-Earths Model",
        "temperature": 3517.83,  # K
        "vmf": 4.17,  # %
        "disk_theia_mass_fraction": 51.97,  # %
        "disk_mass": 1.70,  # lunar masses
        "vapor_loss_fraction": 16.0,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    }
]


def __run(run, bse_composition, lunar_bulk_composition, recondensed, run_name, run_path):
    ejecta_data = theia_mixing(
        guess_initial_composition=bse_composition, target_composition=lunar_bulk_composition,
        temperature=run["temperature"],
        vmf=run['vmf'] / 100, vapor_loss_fraction=run['vapor_loss_fraction'] / 100,
        full_report_path=run_path, target_melt_composition_type=recondensed, bse_composition=bse_composition
    )
    # write the ejecta data to a file
    run_full_MAGMApy(
        composition=ejecta_data['ejecta_composition'],
        target_composition=lbc,
        temperature=run["temperature"],
        to_vmf=90, to_dir=run_path
    )
    # get Theia's composition
    theia_data = get_theia_composition(starting_composition=ejecta_data['ejecta_composition'],
                                       earth_composition=bse_composition, disk_mass=run["disk_mass"] * mass_moon,
                                       earth_mass=run["disk_mass"] * mass_moon -
                                                  (run["disk_mass"] * mass_moon * (
                                                          run['disk_theia_mass_fraction'] / 100.0)))
    # write the ejecta data (dictionary) to a file in text format
    with open(run_path + "/ejecta_composition.csv", "w") as f:
        f.write(str({k: v for k, v in ejecta_data.items() if k not in ['c', 'l', 'g', 't']}))
    # now, write the theia composition dictionary to a file in text format
    with open(run_path + "/theia_composition.csv", "w") as f:
        f.write(str({k: v for k, v in theia_data.items() if k not in ['c', 'l', 'g', 't']}))


def get_all_models(gather=False):
    all_models = []
    if not gather:
        for run in runs:
            run_name = run["run_name"]
            for model in list(lunar_bulk_compositions.keys())[1:]:
                for m in ['recondensed', 'not_recondensed']:
                    name = f"{run_name}_{model}_{m}"
                    path = f"{root_path}{run_name}_{model}_{m}"
                    all_models.append((name, path))
    else:  # the model names are the directory names in the root path
        for model in os.listdir(root_path):
            name = model
            path = f"{root_path}{model}/{model}"
            all_models.append((name, path))
    return all_models

all_models = get_all_models(gather=GATHER)

# ============================== Run Simulations ==============================
if RUN_NEW_SIMULATIONS:
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {}
        for run in runs:
            for model in list(lunar_bulk_compositions.keys())[1:3]:
                lbc = {
                    oxide: lunar_bulk_compositions[model].loc[oxide] for oxide in bse_composition.keys() if
                    oxide != "Fe2O3"
                }
                run_name = run["run_name"]
                for m in ['recondensed', 'not_recondensed']:
                    # __run(run, bse_composition, lbc, m, run_name, f"{root_path}{run_name}_{model}_{m}")
                    #             run_path = f"{root_path}{run_name}_{lbc}_{m}"
                    #             run_name = f"{run_name}_{lbc}_{m}"
                    futures.update({executor.submit(__run, run, bse_composition, lbc, m, run_name,
                                                    f"{root_path}{run_name}_{model}_{m}"): run_name})
        for future in as_completed(futures):
            r = futures[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (r, exc))

# get the range of ejecta compositions and theia compositions
ejecta_compositions = {}
theia_compositions = {}
for model in all_models:
    ejecta_data = eval(open(model[1] + "/ejecta_composition.csv", 'r').read())
    theia_composition = eval(open(model[1] + "/theia_composition.csv", 'r').read())
    ejecta_compositions.update({model[0]: ejecta_data['ejecta_composition']})
    theia_compositions.update({model[0]: theia_composition['theia_weight_pct']})

# get the min and max values for each oxide
min_max_ejecta_compositions = {'with recondensation': {oxide: [1e99, -1e99] for oxide in oxides}, 'without recondensation': {oxide: [1e99, -1e99] for oxide in oxides}}
min_max_theia_compositions = {'with recondensation': {oxide: [1e99, -1e99] for oxide in oxides}, 'without recondensation': {oxide: [1e99, -1e99] for oxide in oxides}}
for oxide in bse_composition.keys():
    for model, path in all_models:
        if oxide != "Fe2O3":
            if "_not_recondensed" in model:
                if ejecta_compositions[model][oxide] < min_max_ejecta_compositions['without recondensation'][oxide][0]:
                    min_max_ejecta_compositions['without recondensation'][oxide][0] = ejecta_compositions[model][oxide]
                if ejecta_compositions[model][oxide] > min_max_ejecta_compositions['without recondensation'][oxide][1]:
                    min_max_ejecta_compositions['without recondensation'][oxide][1] = ejecta_compositions[model][oxide]
                if theia_compositions[model][oxide] < min_max_theia_compositions['without recondensation'][oxide][0]:
                    min_max_theia_compositions['without recondensation'][oxide][0] = theia_compositions[model][oxide]
                if theia_compositions[model][oxide] > min_max_theia_compositions['without recondensation'][oxide][1]:
                    min_max_theia_compositions['without recondensation'][oxide][1] = theia_compositions[model][oxide]
            else:
                if ejecta_compositions[model][oxide] < min_max_ejecta_compositions['with recondensation'][oxide][0]:
                    min_max_ejecta_compositions['with recondensation'][oxide][0] = ejecta_compositions[model][oxide]
                if ejecta_compositions[model][oxide] > min_max_ejecta_compositions['with recondensation'][oxide][1]:
                    min_max_ejecta_compositions['with recondensation'][oxide][1] = ejecta_compositions[model][oxide]
                if theia_compositions[model][oxide] < min_max_theia_compositions['with recondensation'][oxide][0]:
                    min_max_theia_compositions['with recondensation'][oxide][0] = theia_compositions[model][oxide]
                if theia_compositions[model][oxide] > min_max_theia_compositions['with recondensation'][oxide][1]:
                    min_max_theia_compositions['with recondensation'][oxide][1] = theia_compositions[model][oxide]

# ========================== PLOT THE RANGE OF EJECTA COMPOSITIONS ==========================
fig, axs = plt.subplots(1, 2, figsize=(16, 9))
axs = axs.flatten()
axs[0].set_title("Without Recondensation")
axs[1].set_title("With Recondensation")
for ax in axs:
    ax.grid()
    ax.axhline(y=1, color="black", linewidth=4, alpha=1, label="BSE")
for i, model in enumerate(all_models):
    to_index = 1
    min_max_ejecta_compositions_sel = min_max_ejecta_compositions['with recondensation']
    if "not_recondensed" in model[0]:
        to_index = 0
        min_max_ejecta_compositions_sel = min_max_ejecta_compositions['without recondensation']
    # shade the region between the min and max values
    axs[to_index].fill_between(oxides,
                               [min_max_ejecta_compositions_sel[oxide][0] / bse_composition[oxide] for oxide in oxides],
                               [min_max_ejecta_compositions_sel[oxide][1] / bse_composition[oxide] for oxide in oxides],
                               alpha=0.5, color='grey')

axs[0].plot(
    oxides,
    [ejecta_compositions["Canonical Model_O'Neill 1991_not_recondensed"][oxide] /
    bse_composition[oxide] for oxide in oxides],
    linewidth=2.0,
    color='blue',
    label="O'Neill 1991 Model"
)
axs[1].plot(
    oxides,
    [ejecta_compositions["Canonical Model_O'Neill 1991_recondensed"][oxide] /
    bse_composition[oxide] for oxide in oxides],
    linewidth=2.0,
    color='blue',
    label="O'Neill 1991 Model"
)
axs[1].legend(loc='upper right', fontsize=16)

plt.show()




# ========================== PLOT THE RANGE OF THEIA COMPOSITIONS ==========================
fig, axs = plt.subplots(1, 2, figsize=(16, 9))
axs = axs.flatten()
axs[0].set_title("Without Recondensation")
axs[1].set_title("With Recondensation")
axs[0].set_ylabel("Bulk Composition / BSE Composition", fontsize=16)
for ax in axs:
    ax.grid()
    ax.axhline(y=1, color="black", linewidth=4, alpha=1, label="BSE")
    # shade region red underneath y=0
    ax.fill_between(oxides, [0 for oxide in oxides], [-1e99 for oxide in oxides], alpha=0.2, color='red')
    ax.set_ylim(bottom=-1.0, top=4)
    # make all font size larger
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
for i, model in enumerate(all_models):
    to_index = 1
    min_max_theia_compositions_sel = min_max_theia_compositions['with recondensation']
    if "not_recondensed" in model[0]:
        to_index = 0
        min_max_theia_compositions_sel = min_max_theia_compositions['without recondensation']
    # shade the region between the min and max values
    axs[to_index].fill_between(oxides,
                               [min_max_theia_compositions_sel[oxide][0] / bse_composition[oxide] for oxide in oxides],
                               [min_max_theia_compositions_sel[oxide][1] / bse_composition[oxide] for oxide in oxides],
                               alpha=0.5, color='grey')

axs[0].plot(
    oxides,
    [theia_compositions["Canonical Model_O'Neill 1991_not_recondensed"][oxide] /
    bse_composition[oxide] for oxide in oxides],
    linewidth=2.0,
    color='blue',
    label="O'Neill 1991 Model"
)
axs[1].plot(
    oxides,
    [theia_compositions["Canonical Model_O'Neill 1991_recondensed"][oxide] /
    bse_composition[oxide] for oxide in oxides],
    linewidth=2.0,
    color='blue',
    label="O'Neill 1991 Model"
)
axs[1].legend(loc='upper right', fontsize=16)

plt.show()
