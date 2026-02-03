from src.composition import Mixture
from pprint import pprint

# Define the BSE Composition (Visscher and Fegley, 2013)
bse = Mixture({
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

print("BSE Composition (wt%):")
pprint(bse.get_composition(), indent=1)
