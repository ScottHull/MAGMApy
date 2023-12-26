from src.composition import Composition, ConvertComposition, normalize
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data

from isotopes.rayleigh import FullSequenceRayleighDistillation_SingleReservior

import os
import numpy as np
import pandas as pd
import copy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import labellines

# use colorblind-friendly colors
plt.style.use('seaborn-colorblind')
# increase font size
plt.rcParams.update({"font.size": 20})

runs = [
    {
        "run_name": "Canonical Model 2",
        "temperature": 2079.86,  # K
        "vmf": 3.65,  # %
        "0% VMF mass frac": 87.41,  # %
        "100% VMF mass frac": 0.66,  # %
        "disk_theia_mass_fraction": 66.78,  # %
        "disk_mass": 1.02,  # lunar masses
        "vapor_loss_fraction": 0.74,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    }
]

MASS_MOON = 7.34767309e22  # kg

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

bse_element_mass_fraction = normalize(ConvertComposition().oxide_wt_pct_to_cation_wt_pct(bse_composition, include_oxygen=True))

# read in the lunar bulk compositions
lunar_bulk_compositions = pd.read_csv("data/lunar_bulk_compositions.csv", index_col="Oxide")
# order composition by volatility
oxides_ordered = [
    "Al2O3", "TiO2", "CaO", "MgO", "FeO", "SiO2", "K2O", "Na2O", "ZnO"
]
cations_ordered = ["Al", "Ti", "Ca", "Mg", "Fe", "Si", "K", "Na", "Zn"]

def format_species_string(species):
    """
    Splits by _ and converts all numbers to subscripts.
    :param species:
    :return:
    """
    formatted = species.split("_")[0]
    return rf"$\rm {formatted.replace('2', '_{2}').replace('3', '_{3}')}$"

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

for run in runs:
    if run['new_simulation']:
        c = Composition(
            composition=bse_composition
        )

        g = GasPressure(
            composition=c,
            major_gas_species=["SiO", "O2", "MgO", "Fe", "Ca", "Al", "Ti", "Na", "K", "Zn"],
            minor_gas_species="__all__",
        )

        l = LiquidActivity(
            composition=c,
            complex_species="__all__",
            gas_system=g
        )

        t = ThermoSystem(composition=c, gas_system=g, liquid_system=l)

        reports = Report(composition=c, liquid_system=l, gas_system=g, thermosystem=t, to_dir=run['run_name'])

        count = 1
        while t.weight_fraction_vaporized < 0.2:
            l.calculate_activities(temperature=run['temperature'])
            g.calculate_pressures(temperature=run['temperature'], liquid_system=l)
            if l.counter == 1:
                l.calculate_activities(temperature=run['temperature'])
                g.calculate_pressures(temperature=run['temperature'], liquid_system=l)
            t.vaporize()
            l.counter = 0  # reset Fe2O3 counter for next vaporizaiton step
            print(
                "[~] At iteration: {} (Magma Fraction Vaporized: {} %)".format(
                    count, t.weight_fraction_vaporized * 100.0))
            if count % 50 == 0 or count == 1:
                reports.create_composition_report(iteration=count)
                reports.create_liquid_report(iteration=count)
                reports.create_gas_report(iteration=count)
            count += 1

for run in runs:
    # calculate the mass distribution between the desired populations (in kg)
    total_ejecta_mass = run['disk_mass'] * MASS_MOON  # define total ejecta mass, kg
    total_100_pct_vaporized_mass = total_ejecta_mass * run[
        '100% VMF mass frac'] / 100  # define total 100% vaporized mass
    intermediate_pct_vmf_mass = total_ejecta_mass * (100 - run['0% VMF mass frac'] - run['100% VMF mass frac']) / 100  # define intermediate pct VMF mass
    intermediate_pct_vmf_mass_vapor = intermediate_pct_vmf_mass * run['vmf'] / 100  # define intermediate pct VMF mass vapor
    intermediate_pct_vmf_mass_magma = intermediate_pct_vmf_mass * (100 - run['vmf']) / 100  # define intermediate pct VMF mass magma

    # read in the data
    melt_oxide_mass_fraction = collect_data(path=f"{run['run_name']}/magma_oxide_mass_fraction",
                                            x_header='mass fraction vaporized')
    magma_element_mass = collect_data(path=f"{run['run_name']}/magma_element_mass",
                                      x_header='mass fraction vaporized')
    vapor_element_mass = collect_data(path=f"{run['run_name']}/total_vapor_element_mass",
                                      x_header='mass fraction vaporized')

    # get the composition at the VMF
    melt_oxide_mass_fraction_at_vmf = get_composition_at_vmf(
        melt_oxide_mass_fraction,
        run['vmf'] / 100
    )
    magma_element_mass_at_vmf = get_composition_at_vmf(
        magma_element_mass,
        run['vmf'] / 100
    )
    vapor_element_mass_at_vmf = get_composition_at_vmf(
        vapor_element_mass,
        run['vmf'] / 100
    )
    vapor_element_mass_fraction_at_vmf = normalize(vapor_element_mass_at_vmf)
    magma_element_mass_fraction_at_vmf = normalize(magma_element_mass_at_vmf)

    # first, calculate the total ejecta mass for each element
    ejecta_mass = {element: total_ejecta_mass * val / 100 for element, val in bse_element_mass_fraction.items()}
    # first, calculate the mass of the 100% vaporized ejecta for each element
    vaporized_mass = {element: total_100_pct_vaporized_mass * val / 100 for element, val in bse_element_mass_fraction.items()}
    # remove the vaporized mass from the ejecta mass
    ejecta_mass = {element: ejecta_mass[element] - vaporized_mass[element] for element in ejecta_mass.keys()}
    # next, calculate remove the mass of the intermediate VMF vapor from the ejecta mass
    intermediate_pct_vmf_mass_vapor_element_mass = {element: intermediate_pct_vmf_mass_vapor * val / 100 for element, val in vapor_element_mass_fraction_at_vmf.items()}
    ejecta_mass = {element: ejecta_mass[element] - intermediate_pct_vmf_mass_vapor_element_mass[element] for element in ejecta_mass.keys()}

    # convert the ejecta mass back to oxide mass fraction
    ejecta_mass_fraction = normalize(ConvertComposition().cations_mass_to_oxides_weight_percent(ejecta_mass, oxides=bse_composition.keys()))

    run['ejecta_mass_fraction'] = ejecta_mass_fraction

fig, ax = plt.subplots(figsize=(10, 10))
# for each oxide, shade between the min and max value of each oxide in all of the lunar bulk composition models
ax.fill_between(
    [format_species_string(oxide) for oxide in oxides_ordered],
    np.array([min(lunar_bulk_compositions.loc[oxide]) / bse_composition[oxide] for oxide in lunar_bulk_compositions.index]),
    np.array([max(lunar_bulk_compositions.loc[oxide]) / bse_composition[oxide]  for oxide in lunar_bulk_compositions.index]),
    color='lightgrey',
    alpha=0.8,
)
ax.axhline(1, color='k', linestyle='--', linewidth=2)
for run in runs:
    ax.plot(
        [format_species_string(oxide) for oxide in oxides_ordered],
        [run['ejecta_mass_fraction'][oxide] / bse_composition[oxide] for oxide in oxides_ordered],
        label=run['run_name'],
        marker='o',
        # linestyle='--',
        linewidth=3,
        markersize=10,
    )

ax.set_ylabel("Disk / BSE (Oxide wt. %)")
ax.grid()
ax.legend()
# increase font size
plt.tight_layout()
plt.show()
