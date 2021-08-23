from src.composition import Composition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure

composition = {
    "SiO2": 62.93000,
    'MgO': 3.79000,
    'Al2O3': 15.45000,
    'TiO2': 0.70000,
    'Fe2O3': 0.00000,
    'FeO': 5.78000,
    'CaO': 5.63000,
    'Na2O': 3.27000,
    'K2O': 2.45000
}

major_gas_species = [
    "SiO", "O2", "MgO", "Fe", "Ca", "Al", "Ti", "Na", "K"
]
gas_ion_species = [
    "Na+", "K+", "e-"
]


c = Composition(
    composition=composition
)

l = LiquidActivity(
    composition=c,
    complex_species="__all__"
)
l.calculate_activities(temperature=2500)

g = GasPressure(
    composition=c,
    major_gas_species=major_gas_species,
    minor_gas_species="__all__",
    ion_gas_species=gas_ion_species
)
g.calculate_pressures(temperature=2500, liquid_system=l)
