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

runs = [
    {
        "run_name": "Canonical Model 2",
        "temperature": 2682.61,  # K
        "vmf": 0.96,  # %
        "disk_theia_mass_fraction": 66.78,  # %
        "disk_mass": 1.02,  # lunar masses
        "vapor_loss_fraction": 74.0,  # %
        "new_simulation": True,  # True to run a new simulation, False to load a previous simulation
    }
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


