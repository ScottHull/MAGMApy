import copy

from monte_carlo.monte_carlo import run_monte_carlo_vapor_loss
from theia.theia import get_theia_composition, recondense_vapor
from theia.chondrites import plot_chondrites, get_enstatite_bulk_theia_core_si_pct
from monte_carlo.monte_carlo import theia_mixing, run_full_MAGMApy
from src.plots import collect_data, collect_metadata
from src.composition import normalize, get_molecular_mass, ConvertComposition
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
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor, as_completed

# use colorblind-friendly colors
plt.style.use('seaborn-colorblind')

RUN_NEW_SIMULATIONS = False
NUM_THREADS = 40
GATHER = False
# root_path = ""
# root_path = "C:/Users/Scott/OneDrive/Desktop/vaporize_theia/"
root_path = "/scratch/shull4/vaporize_theia/"

if RUN_NEW_SIMULATIONS:
    if not os.path.exists(root_path) and len(root_path) > 0:
        os.mkdir(root_path)

# ============================== Define Compositions ==============================

bse_composition = normalize({  # Visscher and Fegley (2013)
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
})

oxides = [i for i in bse_composition.keys() if i != "Fe2O3"]
lunar_bulk_compositions = pd.read_csv("data/lunar_bulk_compositions.csv", index_col="Oxide")
# normalize and replace the lunar bulk composition with the normalized values
favored_composition = [lunar_bulk_compositions["O'Neill 1991"].loc[oxide] for oxide in oxides]
mass_moon = 7.34767309e22  # kg, mass of the moon

annotate_models = [
    "Canonical (No Recondensation)",
    "Canonical (With Recondensation)",
    "Half-Earths (No Recondensation)",
    "Half-Earths (With Recondensation)"
]

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


def write_mass_distribution_file(melt_mass_at_vmf, bulk_vapor_mass_at_vmf, run_name,
                                 escaping_vapor_mass_at_vmf, retained_vapor_mass_at_vmf, to_path):
    if os.path.exists(f"mass_distribution.csv"):
        os.remove(f"mass_distribution.csv")
    tp_copy = to_path
    # if to_path doesn't end with a /, add one
    if tp_copy[-1] != "/":
        tp_copy += "/"
    with open(f"{tp_copy}mass_distribution.csv", "w") as f:
        header = "component," + ",".join([str(i) for i in melt_mass_at_vmf.keys()]) + "\n"
        f.write(header)
        f.write("melt mass," + ",".join([str(i) for i in melt_mass_at_vmf.values()]) + "\n")
        f.write("bulk vapor mass," + ",".join([str(i) for i in bulk_vapor_mass_at_vmf.values()]) + "\n")
        f.write("bulk system mass," + ",".join([str(i) for i in (np.array(list(melt_mass_at_vmf.values())) + np.array(
            list(bulk_vapor_mass_at_vmf.values()))).tolist()]) + "\n")
        f.write("escaping vapor mass," + ",".join([str(i) for i in escaping_vapor_mass_at_vmf.values()]) + "\n")
        f.write("retained vapor mass," + ",".join([str(i) for i in retained_vapor_mass_at_vmf.values()]) + "\n")
        f.write(
            "recondensed melt mass," + ",".join([str(i) for i in (np.array(list(melt_mass_at_vmf.values())) + np.array(
                list(retained_vapor_mass_at_vmf.values()))).tolist()]) + "\n")
    print(f"wrote file mass_distribution.csv")
    f.close()


def __run(run, bse_composition, lunar_bulk_composition, recondensed, run_name, run_path):
    if not os.path.exists(run_path):
        os.mkdir(run_path)
    ejecta_data = theia_mixing(
        guess_initial_composition=bse_composition, target_composition=lunar_bulk_composition,
        temperature=run["temperature"],
        vmf=run['vmf'] / 100, vapor_loss_fraction=run['vapor_loss_fraction'] / 100,
        full_report_path=run_path, target_melt_composition_type=recondensed, bse_composition=bse_composition
    )

    disk_mass_in_kg = run["disk_mass"] * mass_moon
    earth_mass_in_disk_in_kg = disk_mass_in_kg - (disk_mass_in_kg * (run['disk_theia_mass_fraction'] / 100.0))

    # write the ejecta data to a file
    run_full_MAGMApy(
        composition=ejecta_data['ejecta_composition'],
        target_composition=lbc,
        temperature=run["temperature"],
        to_vmf=90, to_dir=run_path
    )
    # get Theia's composition
    theia_data = get_theia_composition(starting_composition=ejecta_data['ejecta_composition'],
                                       earth_composition=bse_composition, disk_mass=disk_mass_in_kg,
                                       earth_mass=earth_mass_in_disk_in_kg)
    write_mass_distribution_file(
        melt_mass_at_vmf=ejecta_data['recondensed__original_melt_element_masses'],
        bulk_vapor_mass_at_vmf=ejecta_data['vapor_element_mass_at_vmf'],
        run_name=run_name,
        escaping_vapor_mass_at_vmf=ejecta_data['recondensed__lost_vapor_element_masses'],
        retained_vapor_mass_at_vmf=ejecta_data['recondensed__retained_vapor_element_masses'],
        to_path=run_path
    )
    # write the ejecta data (dictionary) to a file in text format
    with open(run_path + "/ejecta_composition.csv", "w") as f:
        f.write(str({k: v for k, v in ejecta_data.items() if k not in ['c', 'l', 'g', 't']}))
    # now, write the theia composition dictionary to a file in text format
    with open(run_path + "/theia_composition.csv", "w") as f:
        f.write(str({k: v for k, v in theia_data.items() if k not in ['c', 'l', 'g', 't']}))


def get_base_model_from_run_name(run_name):
    for run in runs:
        rn = run["run_name"]
        run_name_in_run = run_name.split("_")[0]
        if rn == run_name_in_run:
            return run


def format_species_string(species):
    """
    Splits by _ and converts all numbers to subscripts.
    :param species:
    :return:
    """
    formatted = species.split("_")[0]
    return rf"$\rm {formatted.replace('2', '_{2}').replace('3', '_{3}')}$"


def get_all_models(gather=False):
    all_models = []
    if not gather:
        for run in runs:
            run_name = run["run_name"]
            for model in list(lunar_bulk_compositions.keys()):
                for m in ['recondensed', 'not_recondensed']:
                    name = f"{run_name}_{model}_{m}"
                    path = f"{root_path}{run_name}_{model}_{m}"
                    all_models.append((name, path))
    else:  # the model names are the directory names in the root path
        for model in os.listdir(root_path):
            name = model
            path = f"{root_path}{model}"
            if "scratch" not in root_path:
                path += f"/{model}"
            all_models.append((name, path))
    return all_models


def format_compositions_for_latex(name: str, compositions: pd.DataFrame):
    """
    Formats a LaTeX table for the given compositions, which are to be given as a pandas DataFrame.
    The headers are the element/oxide names, and the index is the model name.
    Exports the table to a file.
    :param compositions:
    :return:
    """
    # get the model names
    model_names = list(compositions.index)
    # get the oxide names
    oxide_names = list(compositions.columns)
    # get the oxide names, but with the LaTeX formatting
    oxide_names_latex = [f"${oxide}$" for oxide in oxide_names]
    # get the compositions as a numpy array
    compositions_array = compositions.to_numpy()
    # convert all values to scientific notation with 2 decimal places
    compositions_array = np.array([["{:.2e}".format(value) for value in row] for row in compositions_array])
    # for values > 1e-2, convert to 2 decimal places number
    compositions_array = np.array([[float(value) if float(value) > 1e-2 else value for value in row] for row in
                                   compositions_array])
    # get the compositions as a list of lists
    compositions_list = compositions_array.tolist()
    # create the table
    table = tabulate(compositions_list, headers=oxide_names_latex, showindex=model_names, tablefmt="latex_raw")
    # write the table to a file
    with open(f"{name}_compositions.tex", "w") as f:
        f.write(table)


# ========================== PLOT THE LUNAR BULK COMPOSITIONS ==========================
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
colors = sns.color_palette('husl', n_colors=len(lunar_bulk_compositions.keys()))
for index, model in enumerate(lunar_bulk_compositions.keys()):
    ax.plot(
        lunar_bulk_compositions.index.tolist(),
        [lunar_bulk_compositions[model][oxide] / bse_composition[oxide] for oxide in lunar_bulk_compositions[model].keys() if oxide != "Fe2O3"],
        color=colors[list(lunar_bulk_compositions).index(model)], linewidth=2.0, label=model
    )
    if index + 1 == len(lunar_bulk_compositions.keys()):
        ax.set_xticklabels([format_species_string(oxide) for oxide in lunar_bulk_compositions[model].keys()], rotation=45)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_title("Lunar Bulk Composition", fontsize=16)
ax.set_ylabel("Lunar Bulk Composition / BSE Composition", fontsize=16)
ax.grid()
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig("lunar_bulk_compositions.png", format='png', dpi=300)

all_models = get_all_models(gather=GATHER)

# ============================== Run Simulations ==============================
if RUN_NEW_SIMULATIONS:
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {}
        for run in runs:
            for model in list(lunar_bulk_compositions.keys()):
                lbc = {
                    oxide: lunar_bulk_compositions[model].loc[oxide] for oxide in bse_composition.keys() if
                    oxide != "Fe2O3"
                }
                run_name = run["run_name"]
                for m in ['recondensed', 'not_recondensed']:
                    # __run(run, bse_composition, lbc, m, run_name, f"{root_path}{run_name}_{model}_{m}")
                    #             run_path = f"{root_path}{run_name}_{lbc}_{m}"
                    #             run_name = f"{run_name}_{lbc}_{m}"
                    if NUM_THREADS > 1:
                        futures.update({executor.submit(__run, run, bse_composition, lbc, m, run_name,
                                                        f"{root_path}{run_name}_{model}_{m}"): run_name})
                    else:
                        __run(run, bse_composition, lbc, m, run_name, f"{root_path}{run_name}_{model}_{m}")
        if NUM_THREADS > 1:
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

# export the ejecta and theia compositions to a pandas DataFrame, and then to a latex table
ejecta_compositions_df = pd.DataFrame(ejecta_compositions).transpose()
theia_compositions_df = pd.DataFrame(theia_compositions).transpose()
# we need ejecta and theia compositions split into the following groups:
# between the Canonical and Half-Earths models, and between models with and without recondensation
# so loop through each scenario and subset the DataFrame accordingly
for run in runs:
    run_name = run["run_name"]
    cols = [
               f"nr_{oxide}" for oxide in bse_composition.keys() if oxide != "Fe2O3"
           ] + [f"r_{oxide}" for oxide in bse_composition.keys() if oxide != "Fe2O3"]
    # create placeholder dataframes
    ejecta_compositions_df_subset = pd.DataFrame(columns=cols)
    theia_compositions_df_subset = pd.DataFrame(columns=cols)
    # loop through each model and subset the dataframes
    for model in list(lunar_bulk_compositions.keys()):
        # create a placeholder row
        ejecta_compositions_df_subset.loc[model] = [0 for _ in range(len(cols))]
        theia_compositions_df_subset.loc[model] = [0 for _ in range(len(cols))]
        for m in ['not_recondensed', 'recondensed']:
            m_prfix = "r"
            if m == 'not_recondensed':
                m_prfix = "nr"
            name = f"{run_name}_{model}_{m}"
            for oxide in [i for i in bse_composition.keys() if i != "Fe2O3"]:
                ejecta_compositions_df_subset[f"{m_prfix}_{oxide}"][model] = ejecta_compositions_df.loc[name][
                    oxide]
                theia_compositions_df_subset[f"{m_prfix}_{oxide}"][model] = theia_compositions_df.loc[name][
                    oxide]

    format_compositions_for_latex(f"bulk_ejecta_{run_name}", ejecta_compositions_df_subset)
    format_compositions_for_latex(f"bulk_theia_{run_name}", theia_compositions_df_subset)

    # export ejecta and theia as csv files
    ejecta_compositions_df_subset.to_csv(f"bulk_ejecta_{run_name}.csv")
    theia_compositions_df_subset.to_csv(f"bulk_theia_{run_name}.csv")

# get the min and max values for each oxide
min_max_ejecta_compositions = {'with recondensation': {oxide: [1e99, -1e99] for oxide in oxides},
                               'without recondensation': {oxide: [1e99, -1e99] for oxide in oxides}}
min_max_theia_compositions = {'with recondensation': {oxide: [1e99, -1e99] for oxide in oxides},
                              'without recondensation': {oxide: [1e99, -1e99] for oxide in oxides}}
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
fig, axs = plt.subplots(2, 2, figsize=(16, 9), sharex='all', sharey='all')
axs = axs.flatten()
# axs[0].set_title("Ejecta Bulk Composition (Without Recondensation)", fontsize=16)
# axs[1].set_title("Ejecta Bulk Composition (With Recondensation)", fontsize=16)
for index, ax in enumerate(axs):
    ax.grid()
    label = None
    if index == 0:
        label = "BSE"
    ax.axhline(y=1, color="black", linewidth=4, alpha=1, label=label)
for i, s in enumerate(ejecta_compositions.keys()):
    base_model = s.split("_")[1]
    to_index = 1
    label = None
    if "_not_recondensed" in s:
        to_index = 0
    if "Half-Earths" in s:
        to_index += 2
    if to_index == 0:
        label = base_model
    # shade the region between the min and max values
    # axs[to_index].fill_between(oxides,
    #                            [min_max_ejecta_compositions[s][oxide][0] / bse_composition[oxide] for oxide in oxides],
    #                            [min_max_ejecta_compositions[s][oxide][1] / bse_composition[oxide] for oxide in oxides],
    #                            alpha=0.5, color='grey')
    axs[to_index].plot(
        oxides, [ejecta_compositions[s][oxide] / bse_composition[oxide] for oxide in oxides],
        color=colors[list(lunar_bulk_compositions).index(base_model)], linewidth=2.0, label=label
    )

# axs[0].plot(
#     oxides,
#     [ejecta_compositions["Canonical Model_O'Neill 1991_not_recondensed"][oxide] /
#     bse_composition[oxide] for oxide in oxides],
#     linewidth=2.0,
#     color='blue',
#     label="O'Neill 1991 Model"
# )
# axs[1].plot(
#     oxides,
#     [ejecta_compositions["Canonical Model_O'Neill 1991_recondensed"][oxide] /
#     bse_composition[oxide] for oxide in oxides],
#     linewidth=2.0,
#     color='blue',
#     label="O'Neill 1991 Model"
# )

# set minimum plotted x value
letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs):
    # label each subplot with a letter in the upper-left corner
    ax.annotate(
        letters[index], xy=(0.05, 0.95), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontweight="bold", fontsize=20
    )
    ax.annotate(
        annotate_models[index], xy=(0.05, 0.85), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontsize=16
    )

fig.supylabel("Bulk Composition / BSE Composition", fontsize=16)
# replace the x-axis labels with the formatted oxide names
for ax in axs[-2:]:
    ax.set_xticklabels([format_species_string(oxide) for oxide in oxides], rotation=45)
# set the axis font size to be 16 for each subplot
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
fig.legend(loc=7)
fig.subplots_adjust(right=0.76)
# add legend to the right of the figure
plt.savefig("theia_mixing_ejecta_compositions.png", dpi=300)
plt.show()

# ========================== PLOT THE RANGE OF THEIA COMPOSITIONS ==========================
fig, axs = plt.subplots(2, 2, figsize=(16, 9), sharex='all', sharey='all')
axs = axs.flatten()
# axs[0].set_title("Theia Bulk Composition (Without Recondensation)", fontsize=16)
# axs[1].set_title("Theia Bulk Composition (With Recondensation)", fontsize=16)
fig.supylabel("Bulk Composition / BSE Composition", fontsize=16)
colors = sns.color_palette('husl', n_colors=len(lunar_bulk_compositions.keys()))
for index, ax in enumerate(axs):
    ax.grid()
    label = None
    if index == 0:
        label = "BSE"
    ax.axhline(y=1, color="black", linewidth=4, alpha=1, label=label)
    # shade region red underneath y=0
    ax.fill_between(oxides, [0 for oxide in oxides], [-1e99 for oxide in oxides], alpha=0.2, color='red')
    ax.set_ylim(bottom=-1.0, top=4.2)
for ax in axs[:-2]:
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
for i, s in enumerate(theia_compositions.keys()):
    base_model = s.split("_")[1]
    label = None
    to_index = 1
    mm = None
    if "_not_recondensed" in s:
        to_index = 0
    if "Half-Earths" in s:
        to_index += 2
    if to_index == 0:
        label = base_model
    # shade the region between the min and max values
    # axs[to_index].fill_between(oxides,
    #                            [min_max_theia_compositions[s][oxide][0] / bse_composition[oxide] for oxide in oxides],
    #                            [min_max_theia_compositions[s][oxide][1] / bse_composition[oxide] for oxide in oxides],
    #                            alpha=0.5, color='grey')
    axs[to_index].plot(
        oxides, [theia_compositions[s][oxide] / bse_composition[oxide] for oxide in oxides],
        color=colors[list(lunar_bulk_compositions).index(base_model)], linewidth=2.0, label=label
    )

# axs[0].plot(
#     oxides,
#     [theia_compositions["Canonical Model_O'Neill 1991_not_recondensed"][oxide] /
#     bse_composition[oxide] for oxide in oxides],
#     linewidth=2.0,
#     color='blue',
#     label="O'Neill 1991 Model"
# )
# axs[1].plot(
#     oxides,
#     [theia_compositions["Canonical Model_O'Neill 1991_recondensed"][oxide] /
#     bse_composition[oxide] for oxide in oxides],
#     linewidth=2.0,
#     color='blue',
#     label="O'Neill 1991 Model"
# )

# set minimum plotted x value
letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs):
    # label each subplot with a letter in the upper-left corner
    ax.annotate(
        letters[index], xy=(0.05, 0.95), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontweight="bold", fontsize=20
    )
    ax.annotate(
        annotate_models[index], xy=(0.05, 0.85), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontsize=16
    )

# replace the x-axis labels with the formatted oxide names
for ax in axs[-2:]:
    ax.set_xticklabels([format_species_string(oxide) for oxide in oxides], rotation=45)

# set the axis font size to be 16 for each subplot
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
fig.legend(loc=7)
fig.subplots_adjust(right=0.76)
plt.savefig("theia_mixing_theia_compositions.png", dpi=300)
plt.show()

# # ========================== PLOT THE RANGE OF EJECTA COMPOSITIONS DISTINCTLY ==========================
# fig, axs = plt.subplots(2, 2, figsize=(16, 9))
# axs = axs.flatten()
# axs[0].set_title("Without Recondensation")
# axs[1].set_title("With Recondensation")
# colors = sns.color_palette('husl', n_colors=len(lunar_bulk_compositions.keys()))
# for model in lunar_bulk_compositions.keys():
#     color = colors[list(lunar_bulk_compositions.keys()).index(model)]
#     axs[1].plot(
#         [], [], linewidth=2.0, color=color, label=model
#     )
# for ax in axs:
#     ax.grid()
#     ax.axhline(y=1, color="black", linewidth=4, alpha=1, label="BSE")
# for i, s in enumerate(min_max_ejecta_compositions.keys()):
#     to_index = 1
#     if "_not_recondensed" in s:
#         to_index = 0
#     if "Half-Earths" in s:
#         to_index += 2
#     # shade the region between the min and max values
#     # axs[to_index].fill_between(oxides,
#     #                            [min_max_ejecta_compositions[s][oxide][0] / bse_composition[oxide] for oxide in oxides],
#     #                            [min_max_ejecta_compositions[s][oxide][1] / bse_composition[oxide] for oxide in oxides],
#     #                            alpha=0.5, color='grey')
#
# for model in ejecta_compositions.keys():
#     lbc_model = model.split("_")[1]
#     color = colors[list(lunar_bulk_compositions.keys()).index(lbc_model)]
#     to_index = 1
#     if "not_recondensed" in model:
#         to_index = 0
#     if "Half-Earths" in model:
#         to_index += 2
#     axs[to_index].plot(
#     oxides,
#     [ejecta_compositions[model][oxide] /
#     bse_composition[oxide] for oxide in oxides],
#     linewidth=2.0,
#     color=color,
#     # label=model.split("_")[1]
# )
#
# axs[1].legend(loc='upper right', fontsize=16)
# plt.tight_layout()
# plt.savefig("theia_mixing_ejecta_compositions_distinct.png", dpi=300)
# plt.show()


# ========================== VERIFY THAT EJECTA COMPOSITION MAKES SENSE ==========================
# for model in all_models:
#     base_model = get_base_model_from_run_name(model[0])
#     vmf = base_model['vmf']
#     lunar_model = model[0].split("_")[1]
#     bulk_moon_composition = lunar_bulk_compositions[lunar_model]
#     ejecta_data = ejecta_data = eval(open(f'{model[1]}/ejecta_composition.csv', 'r').read())
#     melt_oxide_mass_fraction = collect_data(path=f"{model[1]}/magma_oxide_mass_fraction",
#                                             x_header='mass fraction vaporized')
#     fig, ax = plt.subplots(figsize=(16, 9))
#     ax.tick_params(axis='both', which='major', labelsize=20)
#     colors = sns.color_palette('husl', n_colors=len([i for i in bulk_moon_composition.keys() if i != 'Fe2O3']))
#     for index, oxide in enumerate(oxides):
#         ax.plot(np.array(list(melt_oxide_mass_fraction.keys())) * 100,
#                 [melt_oxide_mass_fraction[vmf][oxide] * 100 for vmf in melt_oxide_mass_fraction.keys()],
#                 linewidth=2.0,
#                 color=colors[index],
#                 label=oxide
#                 )
#         ax.axhline(y=bulk_moon_composition[oxide], color=colors[index], linestyle='--')
#         ax.scatter(
#             vmf,
#             ejecta_data['liquid_composition_at_vmf'][oxide],
#             color=colors[index],
#             marker='o'
#         )
#         ax.scatter(
#             min(melt_oxide_mass_fraction.keys()) * 100,
#             ejecta_data['ejecta_composition'][oxide],
#             color=colors[index],
#             marker='s'
#         )
#     ax.scatter(
#         [], [], color='black', marker='d', label="w/o recondensed vapor"
#     )
#     ax.scatter(
#         [], [], color='black', marker='o', label="w/ recondensed vapor"
#     )
#     ax.scatter(
#         [], [], color='black', marker='s', label="Bulk Ejecta"
#     )
#     ax.axvline(x=vmf, color='black', linestyle='--', label=f"VMF {vmf} %")
#     ax.set_title(f"{model[0]}", fontsize=20)
#     ax.set_xlabel("VMF (%)", fontsize=20)
#     ax.set_ylabel("Oxide Wt. %", fontsize=20)
#     ax.grid(alpha=0.4)
#     ax.legend(loc='lower right', fontsize=20)
#     plt.tight_layout()
#     # plt.show()

# ================================= Loss Fraction of From Each Model =================================
fig, axs = plt.subplots(2, 2, figsize=(16, 9), sharex='all', sharey='all')
axs = axs.flatten()
pct_50_cond_temps = pd.read_csv("data/50_pct_condensation_temperatures.csv", index_col="Element")
loss_frac_dict = {}
vmf_dict = {}
for i, s in enumerate(ejecta_compositions.keys()):
    ejecta_data = eval(open(f"{root_path}/{s}" + "/ejecta_composition.csv", 'r').read())
    base_model = s.split("_")[1]
    to_index = 1
    label = None
    if "_not_recondensed" in s:
        to_index = 0
    if "Half-Earths" in s:
        to_index += 2
    if to_index == 0:
        label = base_model
    prefix = "recondensed"
    if "_not_recondensed" in s:
        prefix = "not_recondensed"
    cations = list(ejecta_data[f'recondensed__lost_vapor_element_masses'].keys())
    cations = list(reversed(
        sorted(cations, key=lambda x: pct_50_cond_temps["50% Temperature"][x])))
    total_mass = {cation: ejecta_data[f'{prefix}__original_melt_element_masses'][cation] +
                          ejecta_data[f'{prefix}__lost_vapor_element_masses'][cation] +
                          ejecta_data[f'{prefix}__retained_vapor_element_masses'][cation] for cation in cations}
    total_vapor_mass = {cation: ejecta_data[f'{prefix}__lost_vapor_element_masses'][cation] +
                                ejecta_data[f'{prefix}__retained_vapor_element_masses'][cation] for cation in cations}
    loss_fraction_recondensed = {
        cation: ejecta_data[f'{prefix}__lost_vapor_element_masses'][cation] / total_mass[cation] * 100 for cation
        in cations}
    loss_fraction_not_recondensed = {
        cation: total_vapor_mass[cation] / total_mass[cation] * 100 for cation
        in cations}
    if "_not_recondensed" in s:
        axs[to_index].plot(
            [i for i in cations if i != "O"],
            [loss_fraction_not_recondensed[cation] for cation in cations if cation != "O"],
            color=colors[list(lunar_bulk_compositions).index(base_model)], linewidth=2.0, label=label
        )
    else:
        axs[to_index].plot(
            [i for i in cations if i != "O"],
            [loss_fraction_recondensed[cation] for cation in cations if cation != "O"],
            color=colors[list(lunar_bulk_compositions).index(base_model)], linewidth=2.0, label=label
        )

letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs):
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.set_yscale('log')
    ax.annotate(
        letters[index], xy=(0.05, 0.95), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontweight="bold", fontsize=20
    )
    ax.annotate(
        annotate_models[index], xy=(0.05, 0.85), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontsize=16
    )
# axs[0].set_title("Mass Loss Fraction (Without Recondensation)", fontsize=16)
# axs[1].set_title("Mass Loss Fraction (With Recondensation)", fontsize=16)
for index, ax in enumerate(axs):
    if index % 2 == 0:
        ax.set_ylabel("Mass Loss Fraction (%)", fontsize=20)
plt.tight_layout()
# axs[1].legend(loc='lower right', fontsize=14)
# add legend to the right of the figure
fig.legend(loc=7)
fig.subplots_adjust(right=0.76)
plt.savefig("theia_mixing_element_loss_fractions.png", dpi=300)
plt.show()

# =========================== EXPORT MASS LOSS FRACTION AND VMF SIDE-BY-SIDE FOR CANONICAL/HALF-EARTHS MODEL =========
cols = [i + f"_canonical" for i in cations] + [i + f"_half_earths" for i in cations]
df_loss_fraction_not_recondensed = pd.DataFrame(columns=cols)
df_vmf_not_recondensed = pd.DataFrame(columns=cols)
df_loss_fraction_recondensed = pd.DataFrame(columns=cols)
df_vmf_recondensed = pd.DataFrame(columns=cols)
for model in list(lunar_bulk_compositions.keys()):
    for i in [df_loss_fraction_not_recondensed, df_vmf_not_recondensed, df_loss_fraction_recondensed, df_vmf_recondensed]:
        i.loc[model] = [None for i in range(len(cols))]
for run in runs:
    run_name = run['run_name']
    run_prefix = "canonical"
    if "Half-Earths" in run_name:
        run_prefix = "half_earths"
    for i, s in enumerate(ejecta_compositions.keys()):
        prefix = "recondensed"
        if "not_recondensed" in s:
            prefix = "not_recondensed"
        if run_name in s:
            target_loss_fraction_df = df_loss_fraction_recondensed
            target_vmf_df = df_vmf_recondensed
            if "not_recondensed" in s:
                target_loss_fraction_df = df_loss_fraction_not_recondensed
                target_vmf_df = df_vmf_not_recondensed
            ejecta_data = eval(open(f"{root_path}/{s}" + "/ejecta_composition.csv", 'r').read())
            base_model = s.split("_")[1]
            total_mass = {cation: ejecta_data[f'{prefix}__original_melt_element_masses'][cation] +
                                  ejecta_data[f'{prefix}__lost_vapor_element_masses'][cation] +
                                  ejecta_data[f'{prefix}__retained_vapor_element_masses'][cation] for cation in cations}
            total_vapor_mass = {cation: ejecta_data[f'{prefix}__lost_vapor_element_masses'][cation] +
                                        ejecta_data[f'{prefix}__retained_vapor_element_masses'][cation] for cation in
                                cations}
            vmfs = {
                cation: total_vapor_mass[cation] / total_mass[cation] * 100 for cation in cations
            }
            loss_fraction_recondensed = {
                cation: ejecta_data[f'{prefix}__lost_vapor_element_masses'][cation] / total_mass[cation] * 100 for cation
                in cations}
            loss_fraction_not_recondensed = {
                cation: total_vapor_mass[cation] / total_mass[cation] * 100 for cation
                in cations
            }

            for c in cations:
                tdf2 = loss_fraction_recondensed
                if "not_recondensed" in s:
                    tdf2 = loss_fraction_not_recondensed
                target_loss_fraction_df.loc[base_model, c + f"_{run_prefix}"] = tdf2[c]
                target_vmf_df.loc[base_model, c + f"_{run_prefix}"] = vmfs[c]

for i, j in zip(
    [df_loss_fraction_not_recondensed, df_vmf_not_recondensed, df_loss_fraction_recondensed, df_vmf_recondensed],
    ['loss_fraction_not_recondensed', 'vmf_not_recondensed', 'loss_fraction_recondensed', 'vmf_recondensed']
    ):
    # loop through each cell with a number and round it to 2 decimal places
    # if the number if smaller than 0.01, set it to scientific notation
    for col in i.columns:
        for index, row in i.iterrows():
            if row[col] < 0.01:
                i.loc[index, col] = f"{row[col]:.2E}"
            else:
                i.loc[index, col] = f"{row[col]:.2f}"
    table = i.to_latex()
    if f"{j}.tex" in os.listdir(os.getcwd()):
        os.remove(f"{j}.tex")
    with open(f"{j}.tex", 'w') as f:
        f.write(table)


# ================================= Vapor Mass Fraction of From Each Model =================================
fig, axs = plt.subplots(2, 2, figsize=(20, 15), sharex='all', sharey='all')
axs = axs.flatten()
pct_50_cond_temps = pd.read_csv("data/50_pct_condensation_temperatures.csv", index_col="Element")
for index, s in enumerate(ejecta_compositions.keys()):
    to_index = 0
    label = None
    if "Half-Earths" in s:
        to_index += 2
    if not "_not_recondensed" in s:
        to_index += 1
    base_model = s.split("_")[1]
    if to_index == 0:
        label = base_model
    cations = list(ejecta_data[f'recondensed__lost_vapor_element_masses'].keys())
    cations = list(reversed(
        sorted(cations, key=lambda x: pct_50_cond_temps["50% Temperature"][x])))
    # read in the ejecta composition file
    mass_distribution = pd.read_csv(f"{root_path}/{s}" + "/mass_distribution.csv", index_col='component')
    # get the loss fraction of each element
    vapor_fraction = {element: mass_distribution.loc['bulk vapor mass', element] / (
                mass_distribution.loc['melt mass', element] + mass_distribution.loc['bulk vapor mass', element]) * 100.0
                      for element in cations}
    # sort cations by 50% condensation temperature
    cations = list(reversed(sorted(list(vapor_fraction.keys()), key=lambda x: pct_50_cond_temps["50% Temperature"][x])))
    # convert loss fraction to a LaTex table
    table = pd.DataFrame(vapor_fraction, index=['vapor mass fraction']).to_latex()
    # save the table to a file
    if f"{run_name}_vapor_mass_fraction.tex" in os.listdir(f"{root_path}/{s}"):
        os.remove(f"{root_path}/{s}/{run_name}_vapor_mass_fraction.tex")
    with open(f"{root_path}/{s}/{run_name}_vapor_mass_fraction.tex", 'w') as f:
        f.write(table)
    # remove O from the list of cations
    cations.remove('O')
    # get a unique color for each oxide
    axs[to_index].plot(
        cations, [vapor_fraction[cation] for cation in cations],
        linewidth=2,
        color=colors[list(lunar_bulk_compositions).index(base_model)],
        label=label
    )
    # # scatter the loss fraction on top of the line
    # axs[to_index].scatter(
    #     cations, [vapor_fraction[cation] for cation in cations],
    #     color=colors[list(lunar_bulk_compositions).index(base_model)],
    #     s=100,
    #     zorder=10
    # )

for ax in axs:
    # plot arrows at the bottom of the plot to indicate the range of volatility
    ax.arrow(
        -0.5, 10 ** -4.5, 3, 0, width=10 ** -5.8, head_width=10 ** -5, head_length=0.1, fc='k', ec='k', zorder=10,
        length_includes_head=True
    )
    ax.arrow(
        2.5, 10 ** -4.5, -3, 0, width=10 ** -5.8, head_width=10 ** -5, head_length=0.1, fc='k', ec='k', zorder=10,
        length_includes_head=True
    )
    # annotate in the center above the arrows
    ax.annotate(
        "Refractory", xy=(2 / 2, 10 ** -4.3), xycoords="data", horizontalalignment="center", verticalalignment="center",
        fontsize=14, fontweight="bold", backgroundcolor="w"
    )
    ax.arrow(
        2.5, 10 ** -4.5, 3, 0, width=10 ** -5.8, head_width=10 ** -5, head_length=0.1, fc='k', ec='k', zorder=10,
        length_includes_head=True
    )
    ax.arrow(
        5.5, 10 ** -4.5, -3, 0, width=10 ** -5.8, head_width=10 ** -5, head_length=0.1, fc='k', ec='k', zorder=10,
        length_includes_head=True
    )
    # annotate in the center above the arrows
    ax.annotate(
        "Transitional", xy=((5 - 2 / 2), 10 ** -4.3), xycoords="data", horizontalalignment="center",
        verticalalignment="center",
        fontsize=14, fontweight="bold", backgroundcolor="w"
    )
    ax.arrow(
        5.5, 10 ** -4.5, 3, 0, width=10 ** -5.8, head_width=10 ** -5, head_length=0.1, fc='k', ec='k', zorder=10,
        length_includes_head=True
    )
    ax.arrow(
        8, 10 ** -4.5, -2.5, 0, width=10 ** -5.8, head_width=10 ** -5, head_length=0.1, fc='k', ec='k', zorder=10,
        length_includes_head=True
    )
    # annotate in the center above the arrows
    ax.annotate(
        "Moderately Volatile", xy=((8.5 - 3 / 2)
                                   , 10 ** -4.3), xycoords="data", horizontalalignment="center",
        verticalalignment="center",
        fontsize=14, fontweight="bold", backgroundcolor="w"
    )

tmp_models = ["Canonical (No Recondensation)", "Canonical (Recondensed)", "Half-Earths (No Recondensation)",
                              "Half-Earths (Recondensed)"]
for index, ax in enumerate(axs):
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid()
    ax.set_yscale('log')
    ax.set_ylim(bottom=10 ** -5, top=10 ** 2.1)
    ax.annotate(
        letters[index], xy=(0.05, 0.95), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontweight="bold", fontsize=20
    )
    ax.annotate(
        tmp_models[index], xy=(0.05, 0.90), xycoords="axes fraction", horizontalalignment="left",
        verticalalignment="top",
        fontsize=16
    )
for ax in [axs[0], axs[2]]:
    ax.set_ylabel("Vapor Mass Fraction (%)", fontsize=20)
plt.tight_layout()
fig.legend(loc=7, fontsize=16)
fig.subplots_adjust(right=0.76)
plt.savefig("theia_vaporize_element_vapor_mass_fraction.png", dpi=300)
plt.show()


# ================================== THEIA MG/SI VS MG/AL ==================================
# https://www.lpi.usra.edu/books/MESSII/9039.pdf
# See Figure 6
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
# add chondrites
plot_chondrites(ax)
found_base_models = []
# generate a list of 4 different markers
markers = ['o', 's', 'D', '^']
bse_element_masses = ConvertComposition().oxide_wt_to_cation_wt(bse_composition)
bse_mg_si = bse_element_masses["Mg"] / bse_element_masses["Si"]
bse_al_si = bse_element_masses["Al"] / bse_element_masses["Si"]
ax.scatter(
    bse_al_si, bse_mg_si, color="k", s=300, marker="*"
)
# annotate the BSE
ax.annotate(
    "BSE", xy=(bse_al_si, bse_mg_si), xycoords="data", xytext=(bse_al_si + 0.005, bse_mg_si + 0.005), fontsize=14
)
# plot the Mg/Si vs Mg/Al for each of the modelled BST compositions
for index, s in enumerate(ejecta_compositions.keys()):
    base_model = s.split("_")[1]
    label = None
    marker = None
    if base_model not in found_base_models:
        label = base_model
        found_base_models.append(base_model)
        ax.scatter([], [], color=colors[list(lunar_bulk_compositions).index(base_model)], s=100, marker="s", label=label)
    if "not_recondensed" in s and "Canonical" in s:
        marker = markers[0]
    elif not "not_recondensed" in s and "Canonical" in s:
        marker = markers[1]
    elif "not_recondensed" in s and "Half-Earths" in s:
        marker = markers[2]
    elif not "not_recondensed" in s and "Half-Earths" in s:
        marker = markers[3]
    # read in the theia composition file
    theia_composition = eval(open(f"{root_path}{s}/theia_composition.csv", 'r').read())
    # get the mass of each bulk oxide from Theia
    theia_oxide_masses = theia_composition['theia_weights']
    # convert bulk oxide masses to bulk element masses
    theia_element_masses = ConvertComposition().oxide_wt_to_cation_wt(theia_oxide_masses)
    # get the Mg/Si and Mg/Al ratios
    mg_si = theia_element_masses['Mg'] / theia_element_masses['Si']
    al_si = theia_element_masses['Al'] / theia_element_masses['Si']
    # scatter the Mg/Si vs Al/Si
    ax.scatter(al_si, mg_si, color=colors[list(lunar_bulk_compositions).index(base_model)], s=100, marker=marker, edgecolor='k')
for m, model in zip(markers, ["Canonical (No Recondensation)", "Canonical (Recondensed)", "Half-Earths (No Recondensation)",
                              "Half-Earths (Recondensed)"]):
    ax.scatter([], [], color='k', s=100, marker=m, label=model)

ax.set_xlabel("Al/Si (mass ratio)", fontsize=20)
ax.set_ylabel("Mg/Si (mass ratio)", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig("theia_mg_si_vs_al_si.png", dpi=300)


# ================================== Lunar Models MG/SI VS MG/AL ==================================
def sort_lunar_models(models):
    """
    Takes a list of strings and sorts them based on the year at the end of the string.
    :param models:
    :return:
    """
    years = [int(model.replace("Fractional Model", "").replace("Equilibrium Model", "").split(" ")[-1]) for model in models]
    return [model for _, model in sorted(zip(years, models))]





# ==================================== PLOT MG/SI AND AL/SI FOR BULK THEIA ASSUMING ENSTATITE START ==================
fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey='all')
axs = axs.flatten()
# add chondrites
# plot_chondrites(ax)
found_base_models = []
# generate a list of 4 different markers
markers = ['o', 's', 'D', '^']
bse_element_masses = ConvertComposition().oxide_wt_to_cation_wt(bse_composition)
bse_mg_si = bse_element_masses["Mg"] / bse_element_masses["Si"]
bse_al_si = bse_element_masses["Al"] / bse_element_masses["Si"]
bulk_earth_mg = 15.4
bulk_earth_al = 1.59
bulk_earth_si = 16.1
ax.scatter(
    bse_al_si, bse_mg_si, color="k", s=300, marker="*"
)
# annotate the BSE and bulk Earth
ax.annotate(
    "BSE", xy=(bse_al_si, bse_mg_si), xycoords="data", xytext=(bse_al_si + 0.002, bse_mg_si + 0.002), fontsize=14
)
# plot the Mg/Si vs Mg/Al for each of the modelled BST compositions
for index, s in enumerate(ejecta_compositions.keys()):
    base_model = s.split("_")[1]
    label = None
    marker = None
    if base_model not in found_base_models:
        label = base_model
        found_base_models.append(base_model)
        axs[1].scatter([], [], color=colors[list(lunar_bulk_compositions).index(base_model)], s=100, marker="s", label=label)
    if "not_recondensed" in s and "Canonical" in s:
        marker = markers[0]
    elif not "not_recondensed" in s and "Canonical" in s:
        marker = markers[1]
    elif "not_recondensed" in s and "Half-Earths" in s:
        marker = markers[2]
    elif not "not_recondensed" in s and "Half-Earths" in s:
        marker = markers[3]
    # read in the theia composition file
    theia_composition = eval(open(f"{root_path}{s}/theia_composition.csv", 'r').read())
    # get the enstatite-based Theia Mg/Si and Mg/Al ratios as a function of Si core wt%
    shade = None
    if index == 0:
        shade = axs[1]
    pct_si_in_core, mg_si_bulk_theia, al_si_bulk_theia = get_enstatite_bulk_theia_core_si_pct(theia_composition['theia_weight_pct'], ax=shade)
    # scatter the Mg/Si vs Al/Si
    axs[0].scatter(mg_si_bulk_theia, pct_si_in_core, color=colors[list(lunar_bulk_compositions).index(base_model)], s=100, marker=marker, edgecolor='k')
    axs[1].scatter(al_si_bulk_theia, pct_si_in_core, color=colors[list(lunar_bulk_compositions).index(base_model)], s=100, marker=marker, edgecolor='k')
for m, model in zip(markers, ["Canonical (No Recondensation)", "Canonical (Recondensed)", "Half-Earths (No Recondensation)",
                              "Half-Earths (Recondensed)"]):
    axs[1].scatter([], [], color='k', s=100, marker=m, label=model)

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid()
axs[0].set_xlabel("Mg/Si (mass ratio)", fontsize=20)
axs[1].set_xlabel("Al/Si (mass ratio)", fontsize=20)
axs[0].set_ylabel("Si Core Mass Fraction (%)", fontsize=20)
axs[1].legend(fontsize=14)
plt.tight_layout()
plt.savefig("enstatite_theia_si_in_core.png", dpi=300)
