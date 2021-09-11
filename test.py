from src.composition import Composition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem

temperature = 2500

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

c = Composition(
    composition=composition
)
# print("Atoms Composition")
# print(c.atoms_composition)
# print("Oxide Mole Fraction (F) in Silicate")
# print(c.oxide_mole_fraction)
# print("RELATIVE ATOMIC ABUNDANCES OF METALS")
# print(c.cation_fraction)

l = LiquidActivity(
    composition=c,
    complex_species="__all__"
)
g = GasPressure(
    composition=c,
    major_gas_species=major_gas_species,
    minor_gas_species="__all__",
)
t = ThermoSystem(composition=c, gas_system=g, liquid_system=l)

count = 0
while count < 1000:
    print("[!] At count {}".format(count))
    l.calculate_activities(temperature=temperature)
    g.calculate_pressures(temperature=temperature, liquid_system=l)
    t.vaporize()
    print("Gas Total Mole Fraction", g.total_mole_fraction)
    print("Cation Fraction", c.cation_fraction)
    print("Oxide Mole Fraction", c.oxide_mole_fraction)
    print("Planetary", c.planetary_abundances)
    print("Activity Coefficients", l.activity_coefficients)
    count += 1
