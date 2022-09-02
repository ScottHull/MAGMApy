from src.plots import collect_data

import pandas as pd
import numpy as np
from math import log, sqrt
from copy import copy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Bulk Silicate Moon (BSM) composition oxide wt%, Visccher & Fegley 2013
bsm_composition = {
    "SiO2": 44.60,
    'MgO': 35.10,
    'Al2O3': 3.90,
    'TiO2': 0.17,
    'Fe2O3': 0.00000,
    'FeO': 12.40,
    'CaO': 3.30,
    'Na2O': 0.050,
    'K2O': 0.004,
    'ZnO': 2.0e-4,
}

# BSE composition oxide wt%, Visccher & Fegley 2013
bse_composition = {
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

bse_mg_si = (bse_composition['MgO'] / 40.3044) / (bse_composition['SiO2'] / 60.08)  # mg/si at bse

runs = {
    # b = 0.73
    '5b073S': {
        "temperature": 6345.79,  # K
        "vmf": 35.73,  # %
    },
    '500b073S': {
        "temperature": 3664.25,
        "vmf": 19.21,
    },
    '1000b073S': {
        "temperature": 3465.15,
        "vmf": 10.75,
    },
    '2000b073S': {
        "temperature": 3444.84,
        "vmf": 8.57,
    },
    '5b073N': {
        "temperature": 6004.06,
        "vmf": 25.7,
    },
    '500b073N': {
        "temperature": 6637.25,
        "vmf": 27.34,
    },
    '1000b073N': {
        "temperature": 6280.91,
        "vmf": 29.53,
    },
    '2000b073N': {
        "temperature": 4342.08,
        "vmf": 10.61,
    },

    # b = 0.75
    '5b075S': {
        "temperature": 8536.22,
        "vmf": 67.98,
    },
    '500b075S': {
        "temperature": 6554.56,
        "vmf": 39.77,
    },
    '1000b075S': {
        "temperature": 6325.06,
        "vmf": 42.97,
    },
    '2000b075S': {
        "temperature": 4882.66,
        "vmf": 28.67,
    },
    '5b075N': {
        "temperature": 9504.66,
        "vmf": 78.25,
    },
    '500b075N': {
        "temperature": 6970.22,
        "vmf": 46.72,
    },
    '1000b075N': {
        "temperature": 6872.69,
        "vmf": 40.77,
    },
    '2000b075N': {
        "temperature": 6911.39,
        "vmf": 37.78,
    },
}


def return_vmf_and_element_lists(data):
    """
    Returns a list of VMF values and a dictionary of lists of element values.
    :param data:
    :return:
    """
    vmf_list = list(sorted(data.keys()))
    elements_at_vmf = {element: [data[vmf][element] for vmf in vmf_list] for element in data[vmf_list[0]].keys()}
    return vmf_list, elements_at_vmf


def renormalize_interpolated_elements(elements):
    """
    Normalizes the values of the elements dictionary to 1.
    :param elements:
    :return:
    """
    total = sum(elements.values())
    for element in elements.keys():
        elements[element] = elements[element] / total
    return elements


def interpolate_elements_at_vmf(vmf_list, elements, target_vmf):
    """
    Takes 2 VMF keys and uses the data dictionary to interplate all value elements.
    :param closet_vmf_1:
    :param cloest_vmf_2:
    :param data:
    :return:
    """
    interpolated_elements = {}
    for element in elements.keys():
        interp = interp1d(vmf_list, elements[element])
        interpolated_elements[element] = interp(target_vmf)
    return interpolated_elements


def find_best_fit_vmf(vmfs: list, composition: dict, target_composition: dict, restricted_composition=None):
    """
    Find the vmf with the lowest residual error between all composition.
    :param target_composition: given in wt% (assumes composition is normalized to 100)
    :param restricted_composition: Do not include these composition in the fit.
    :param vmfs:
    :param composition: wt% composition of the liquid, assumes normaalized to 1
    :return:
    """
    if restricted_composition is None:
        restricted_composition = []
    best_vmf = vmfs[0]
    best_error = np.inf
    for vmf in vmfs:
        error = 0
        for element in composition.keys():
            if element not in restricted_composition:
                error += ((target_composition[element] / 100) - composition[element][vmfs.index(vmf)]) ** 2
        if error < best_error:
            best_error = error
            best_vmf = vmf
    return best_vmf * 100


def get_element_abundances_as_function_of_vmf(data):
    """
    Uses the VMF keys in the data dictionary and the elements in the embedded dictionaries to build lists of element
    evolution as a function of VMF.
    :param data:
    :return:
    """
    vmfs = list(sorted(data.keys()))
    elements = data[vmfs[0]].keys()
    return vmfs, {element: [data[vmf][element] for vmf in vmfs] for element in elements}


def get_mg_si(vmfs: list, composition: dict, vmf: float):
    """
    Returns the Mg/Si ratio at the given vmf.
    :param vmfs:
    :param composition:
    :param vmf:
    :return:
    """
    interpolated = interpolate_elements_at_vmf(vmfs, composition, vmf)
    mgo_wt = interpolated['MgO']
    sio2_wt = interpolated['SiO2']
    # convert g to mol
    mgo_mol = mgo_wt / 40.3044
    sio2_mol = sio2_wt / 60.08
    return mgo_mol / sio2_mol

def force_mg_si(bulk_composition: dict, mg_si: float):
    """
    Takes the bulk composition dictionary and forces a molar Mg/Si ratio while retaining other
    molecular relative ratios.
    :param bulk_composition:
    :param mg_si: Desired molar ratio of Mg/Si
    :return:
    """
    molar_masses = pd.read_csv("data/periodic_table.csv", index_col="element")
    # keep sum of all other oxides constant so that we ONLY change Mg/Si
    safe_masses = {i: bulk_composition[i] for i in bulk_composition.keys() if i not in ['MgO', 'SiO2']}
    # these are the masses we want to change
    unsafe_masses = {i: bulk_composition[i] for i in bulk_composition.keys() if i in ['MgO', 'SiO2']}
    # get the molar abundances of the unsafe elements
    unsafe_molar_abundances = {i: unsafe_masses[i] / molar_masses.loc[i, 'atomic_mass'] for i in unsafe_masses.keys()}
    # molar weights of Mg/Si
    mg_mol_mass = molar_masses.loc['Mg', 'atomic_mass']
    si_mol_mass = molar_masses.loc['Si', 'atomic_mass']



fig, axs = plt.subplots(nrows=2, ncols=8, figsize=(32, 8), sharex="all", sharey="all")
axs = axs.flatten()
fig.tight_layout()
fig.supxlabel("VMF (%)")
fig.supylabel("Oxide Abundance (wt%)")
restricted_elements = ['SiO2', 'MgO']
for index, run in enumerate(runs.keys()):
    print("at run {}".format(run))
    ax = axs[index]
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax.set_title(run)
    data = collect_data(path="{}_reports/magma_oxide_mass_fraction".format(run), x_header='mass fraction vaporized')
    vmf_list, elements = return_vmf_and_element_lists(data)
    best_vmf = round(find_best_fit_vmf(vmf_list, elements, restricted_composition=restricted_elements,
                                       target_composition=bsm_composition), 2)
    best_fit_mg_si = get_mg_si(vmf_list, elements, best_vmf / 100)  # mg/si at best vmf
    ax.axvline(best_vmf, color='red', linestyle='--', label="Best VMF: {}%".format(best_vmf))
    interpolated_elements = interpolate_elements_at_vmf(vmf_list, elements, runs[run]["vmf"] / 100)

    for index2, oxide in enumerate(elements):
        color = color_cycle[index2]
        ax.plot(np.array(vmf_list) * 100, np.array(elements[oxide]) * 100, color=color)
        # ax.scatter(runs[run]['vmf'], interpolated_elements[oxide] * 100, color=color, s=200, marker='x')
        ax.axhline(bsm_composition[oxide], color=color, linewidth=2.0, linestyle='--')
        ax.scatter([], [], color=color, marker='s', label="{} (MAGMA)".format(oxide))
    ax.plot([], [], color='k', linestyle="--", label="Moon")
    ax.axvline(runs[run]['vmf'], linewidth=2.0, color='k', label="Predicted VMF")

axs[-1].legend(loc='upper right')


fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
run = "500b073N"
restricted_elements = ['SiO2', 'MgO']
ax.set_title(run)
ax.set_xlabel("VMF (%)")
ax.set_ylabel("Oxide Abundance (wt%)")
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
ax.set_title(run)
data = collect_data(path="{}_reports/magma_oxide_mass_fraction".format(run), x_header='mass fraction vaporized')
vmf_list, elements = return_vmf_and_element_lists(data)
best_vmf = round(find_best_fit_vmf(vmf_list, elements, restricted_composition=restricted_elements,
                                   target_composition=bsm_composition), 2)
best_fit_mg_si = get_mg_si(vmf_list, elements, best_vmf / 100)  # mg/si at best vmf
ax.axvline(best_vmf, color='red', linestyle='--', label="Best VMF: {}%".format(best_vmf))
interpolated_elements = interpolate_elements_at_vmf(vmf_list, elements, runs[run]["vmf"] / 100)
for index2, oxide in enumerate(elements):
    color = color_cycle[index2]
    ax.plot(np.array(vmf_list) * 100, np.array(elements[oxide]) * 100, color=color)
    # ax.scatter(runs[run]['vmf'], interpolated_elements[oxide] * 100, color=color, s=200, marker='x')
    ax.axhline(bsm_composition[oxide], color=color, linewidth=2.0, linestyle='--')
    ax.scatter([], [], color=color, marker='s', label="{} (MAGMA)".format(oxide))
ax.plot([], [], color='k', linestyle="--", label="Moon")
ax.axvline(runs[run]['vmf'], linewidth=2.0, color='k', label="Predicted VMF")

ax.legend(loc='upper right')

plt.show()
