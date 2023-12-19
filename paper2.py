from src.composition import Composition, ConvertComposition, normalize
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data, make_figure

from isotopes.rayleigh import FullSequenceRayleighDistillation_SingleReservior

import os
from math import log10
import re
import seaborn as sns
import pandas as pd
import string
from random import uniform
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import labellines

# use colorblind-friendly colors
plt.style.use('seaborn-colorblind')
# increase font size
plt.rcParams.update({"font.size": 12})

target_vmf = 3.0  # %

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
]

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
bulk_moon_composition = normalize({  # O'Neill 1991
    "SiO2": 44.37,
    'MgO': 34.90,
    'Al2O3': 3.90,
    'TiO2': 0.02,
    'Fe2O3': 0.0,
    'FeO': 13.54,
    'CaO': 3.27,
    'Na2O': 3.55e-3,
    'K2O': 3.78e-4,
    'ZnO': 2.39e-5,
})


def get_composition_at_vmf(d: dict, vmf_val: float):
    """
    Given a VMF, interpolate the composition of the d dictionary at that VMF.
    :param d:
    :param vmf_val:
    :return:
    """
    vmfs = list(d.keys())
    species = list(d[vmfs[0]].keys())
    interpolated_composition = {}
    for s in species:
        interpolated_composition[s] = interp1d(
            vmfs,
            [i[s] for i in d.values()]
        )(vmf_val)
    return interpolated_composition


melt_oxide_mass_fraction = collect_data(path=f"{runs[0]['run_name']}/magma_oxide_mass_fraction",
                                                x_header='mass fraction vaporized')
melt_at_target_vmf = get_composition_at_vmf(melt_oxide_mass_fraction, target_vmf / 100)
melt_at_target_vmf = {i: float(j) * 100 for i, j in melt_at_target_vmf.items()}
melt_at_vmf = get_composition_at_vmf(melt_oxide_mass_fraction, runs[0]['vmf'] / 100)
melt_at_vmf = {i: float(j) * 100 for i, j in melt_at_vmf.items()}
print(melt_at_vmf)
