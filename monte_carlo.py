from src.composition import Composition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report

"""
Takes the MAMGApy code and uses it to run a Monte Carlo search for the composition of Theia.
"""

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

earth_mass_fraction = 20  # %
theia_mass_fraction = 80  # %
disk_mass = 0.07346 * 10 ** 24  # kg, mass of the Moon
