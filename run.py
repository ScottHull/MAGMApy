import copy

from monte_carlo.monte_carlo import run_monte_carlo_vapor_loss
from theia.theia import get_theia_composition
from monte_carlo.monte_carlo import test

import os
import json
import matplotlib.pyplot as plt

# use colorblind-friendly colors
plt.style.use('seaborn-colorblind')

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

bsm_composition = {  # Visscher and Fegley (2013)
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

bulk_moon_composition = {  # including core Fe as FeO, my mass balance calculation
    "SiO2": 41.95862216,
    'MgO': 33.02124749,
    'Al2O3': 3.669027499,
    'TiO2': 0.159931968,
    'Fe2O3': 0.0,
    'FeO': 18.03561908,
    'CaO': 3.10456173,
    'Na2O': 0.047038814,
    'K2O': 0.003763105,
    'ZnO': 0.000188155,
}

# ============================== Define Input Parameters ==============================

run_name = "500b073S"
temperature = 2682.61  # K
vmf = 0.96  # %
disk_theia_mass_fraction = 66.78  # %
disk_mass = 1.02  # lunar masses
vapor_loss_fraction = 75.0  # %
run_new_simulation = True  # True to run a new simulation, False to load a previous simulation

# ============================== Define Constants ==============================

disk_earth_mass_fraction = 100 - disk_theia_mass_fraction  # %, fraction
mass_moon = 7.34767309e22  # kg, mass of the moon
disk_mass_kg = disk_mass * mass_moon  # kg, mass of the disk
earth_mass_in_disk_kg = disk_mass_kg * disk_earth_mass_fraction / 100  # kg, mass of the earth in the disk
theia_mass_in_disk = disk_mass_kg - earth_mass_in_disk_kg  # kg, mass of theia in the disk

# ============================== Do some file management ==============================
ejecta_file_path = f"{run_name}_ejecta_data.txt"
theia_file_path = f"{run_name}_theia_composition.txt"

# delete the output files if its a new simulation and they already exist
if run_new_simulation:
    for f in [ejecta_file_path, theia_file_path]:
        try:
            os.remove(f)
        except OSError:
            pass

# ============================== Calculate Bulk Ejecta Composition ==============================

# run the monte carlo simulation
# the initial guess is the BSE composition
# this will calculate the bulk ejecta composition that is required to reproduce the bulk moon composition
# (liquid + retained vapor that is assumed to recondense) at the given VMF and temperature
if run_new_simulation:
    ejecta_data = test(
        guess_initial_composition=bse_composition, target_composition=bulk_moon_composition, temperature=temperature,
        vmf=vmf, vapor_loss_fraction=vapor_loss_fraction
    )
else:
    # read in the data dictionary from the file
    ejecta_data = eval(open(ejecta_file_path, 'r').read())


# ============================== Calculate Bulk Silicate Theia (BST) Composition ==============================

if run_new_simulation:
    theia_data = get_theia_composition(starting_composition=ejecta_data['ejecta composition'],
                                       earth_composition=bse_composition, disk_mass=disk_mass_kg,
                                       earth_mass=earth_mass_in_disk_kg)
else:
    # read in the data dictionary from the file
    theia_data = eval(open(theia_file_path, 'r').read())

# ============================== Save Results  ==============================
# write the ejecta data (dictionary) to a file in text format
if run_new_simulation:
    with open(ejecta_file_path, "w") as f:
        f.write(str({k: v for k, v in ejecta_data.items() if k not in ['c', 'l', 'g', 't']}))
    # now, write the theia composition dictionary to a file in text format
    with open(theia_file_path, "w") as f:
        f.write(str({k: v for k, v in theia_data.items() if k not in ['c', 'l', 'g', 't']}))


# ============================== Plot Bulk Ejecta + BST Relative to BSE ==============================

# calculate the composition of bulk silicate Theia (BST)
fig = plt.figure(111, figsize=(16, 9))
ax = fig.add_subplot(111)
# increase the font size
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel('Oxide', fontsize=20)
ax.set_ylabel("Oxide Wt. % (Relative to BSE)", fontsize=20)
ax.grid()
for t, c in [
    ("BSM", bulk_moon_composition),
    ("Bulk Ejecta Composition", ejecta_data['ejecta composition']),
    ("Bulk Silicate Theia", theia_data['theia_weight_pct'])
]:
    ax.plot(
        [i for i in bse_composition.keys() if i != "Fe2O3"], [c[oxide] / bse_composition[oxide]
                                 for oxide in bse_composition.keys() if oxide != "Fe2O3"],
        linewidth=3, label=t
    )
# plot a horizontal line at 1.0
ax.plot([i for i in bse_composition.keys() if i != "Fe2O3"], [1.0 for _ in bse_composition.keys() if _ != "Fe2O3"],
        linewidth=3, linestyle="--", label="1:1 BSE")
# make x axis labels at 45 degree angle
plt.xticks(rotation=45)
# add a legend with large font
ax.legend(fontsize=20)
# set xlim to start at 0 and end at the length of the x axis
ax.set_xlim(0, len(bse_composition.keys()) - 2)
plt.tight_layout()
plt.show()

# ============================== Plot Vapor Composition As Function of VMF ==============================

