from src.composition import Composition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report

import sys
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 18}
matplotlib.rc('font', **font)

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

g = GasPressure(
    composition=c,
    major_gas_species=major_gas_species,
    minor_gas_species="__all__",
)

l = LiquidActivity(
    composition=c,
    complex_species="__all__",
    gas_system=g
)

t = ThermoSystem(composition=c, gas_system=g, liquid_system=l)

reports = Report(composition=c, liquid_system=l, gas_system=g)

cation_fractions = {}
fraction_vaporized = []

count = 1
while count < 60:
    print("[!] At count {}".format(count))
    l.calculate_activities(temperature=temperature)
    g.calculate_pressures(temperature=temperature, liquid_system=l)
    if l.counter == 1:
        l.calculate_activities(temperature=temperature)
        g.calculate_pressures(temperature=temperature, liquid_system=l)
    t.vaporize()
    l.counter = 0  # reset Fe2O3 counter for next vaporizaiton step
    # print("Gas Total Mole Fraction", g.total_mole_fraction)
    # print("Cation Fraction", c.cation_fraction)
    # print("Oxide Mole Fraction", c.oxide_mole_fraction)
    # print("Planetary", c.planetary_abundances)
    # print("Activity Coefficients", l.activity_coefficients)
    # print("Activities", l.activities)
    count += 1
    for j in l.activities.keys():
        if j not in cation_fractions.keys():
            cation_fractions.update({j: []})
        cation_fractions[j].append(l.activities[j])
    fraction_vaporized.append(t.vaporized_magma_fraction * 100)

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
for i in cation_fractions.keys():
    ax.plot(fraction_vaporized, cation_fractions[i], label=i)
ax.set_xlabel("Magma Fraction Vaporized (%)")
ax.set_ylabel("Melt Species Activities")
ax.grid()
# ax.legend()
plt.show()
