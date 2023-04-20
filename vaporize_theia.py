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


RUN_NEW_SIMULATIONS = True
root_path = "/"
root_path = "/scratch/shull4/"


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

lunar_bulk_compositions = pd.read_csv("data/lunar_bulk_compositions.csv", index_col="Oxide")
favored_composition = [lunar_bulk_compositions["O'Neill 1991"].loc[oxide] for oxide in bse_composition.keys()
                       if oxide != "Fe2O3"]

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
        full_report_path=run_path, target_melt_composition=recondensed, bse_composition=bse_composition
    )
    # write the ejecta data to a file
    run_full_MAGMApy(
        composition=ejecta_data['ejecta_composition'],
        target_composition=lbc,
        temperature=run["temperature"],
        to_vmf=.9, to_dir=run_path
    )

# ============================== Run Simulations ==============================
if RUN_NEW_SIMULATIONS:
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {}
        for run in runs:
            for lbc in list(lunar_bulk_compositions.keys())[1:]:
                run_name = run["run_name"]
                for m in ['recondensed', 'not_recondensed']:
                    run_path = f"{root_path}{run_name}_{lbc}_{m}"
                    run_name = f"{run_name}_{lbc}_{m}"
                    futures.update({executor.submit(__run, run, bse_composition, lbc, m, run_name, run_path): run_name})
        for future in as_completed(futures):
            r = futures[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (r, exc))

                # run_path = f"{run_name}_{lbc}_{m}"
                # ejecta_data = test(
                #     guess_initial_composition=bse_composition, target_composition=lbc,
                #     temperature=run["temperature"],
                #     vmf=run['vmf'] / 100, vapor_loss_fraction=run['vapor_loss_fraction'] / 100,
                #     full_report_path=run_path, target_melt_composition=m
                # )
                # # write the ejecta data to a file
                # run_full_MAGMApy(
                #     composition=ejecta_data['ejecta_composition'],
                #     target_composition=lbc,
                #     temperature=run["temperature"],
                #     to_vmf=.9, to_dir=run_path
                # )
