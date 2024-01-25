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
import string
import copy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as tck
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from random import uniform
import labellines

# use colorblind-friendly colors
plt.style.use('seaborn-colorblind')
# increase font size
plt.rcParams.update({"font.size": 20})

# get the color cycle as a list
prop_cycle = plt.rcParams['axes.prop_cycle']

runs = [
    {
        "run_name": "Canonical Model",
        "temperature": 2657.97,  # K
        "vmf": 3.80,  # %
        "0% VMF mass frac": 87.41,  # %
        "100% VMF mass frac": 0.66,  # %
        "disk_theia_mass_fraction": 66.78,  # %
        "disk_mass": 1.02,  # lunar masses
        "vapor_loss_fraction": 74.0,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    },
    {
        "run_name": "Half Earths Model",
        "temperature": 3514.15,  # K
        "vmf": 14.50,  # %
        "0% VMF mass frac": 81.3,  # %
        "100% VMF mass frac": 1.9,  # %
        "disk_theia_mass_fraction": 51.97,  # %
        "disk_mass": 1.70,  # lunar masses
        "vapor_loss_fraction": 16.0,  # %
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

bse_element_mass_fraction = normalize(
    ConvertComposition().oxide_wt_pct_to_cation_wt_pct(bse_composition, include_oxygen=True))

# read in the lunar bulk compositions
lunar_bulk_compositions = pd.read_csv("data/lunar_bulk_compositions.csv", index_col="Oxide")
# order composition by volatility
oxides_ordered = [
    "Al2O3", "TiO2", "CaO", "MgO", "FeO", "SiO2", "K2O", "Na2O", "ZnO"
]
cations_ordered = ["Al", "Ti", "Ca", "Mg", "Fe", "Si", "K", "Na", "Zn"]

global_disk_compositions_no_recondensation = {'run_name': [i['run_name'] for i in runs]}
global_disk_compisitions_with_recondensation = {'run_name': [i['run_name'] for i in runs]}
for i in [global_disk_compositions_no_recondensation, global_disk_compisitions_with_recondensation]:
    i.update({oxide: [] for oxide in oxides_ordered})
global_element_vmf = {'run_name': [i['run_name'] for i in runs]}
global_element_loss_fraction = {'run_name': [i['run_name'] for i in runs]}
for i in [global_element_vmf, global_element_loss_fraction]:
    i.update({cation: [] for cation in cations_ordered})


def format_species_string(species):
    """
    Splits by _ and converts all numbers to subscripts.
    :param species:
    :return:
    """
    formatted = species.split("_")[0]
    return rf"$\rm {formatted.replace('2', '_{2}').replace('3', '_{3}')}$"


def append_to_global(global_dict, key, val):
    # append the value to the global dictionary as a string
    # if the value is a float, round to 2 decimal places and if it is < 0.01, then use scientific notation
    if val < 0.1:
        global_dict[key].append("{:.2e}".format(val))
    else:
        global_dict[key].append("{:.2f}".format(val))


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
        while t.weight_fraction_vaporized < 0.8:
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
            if count % 25 == 0 or count == 1:
                reports.create_composition_report(iteration=count)
                reports.create_liquid_report(iteration=count)
                reports.create_gas_report(iteration=count)
            count += 1

for run in runs:
    # calculate the mass distribution between the desired populations (in kg)
    total_ejecta_mass = run['disk_mass'] * MASS_MOON  # define total ejecta mass, kg
    total_100_pct_vaporized_mass = total_ejecta_mass * run[
        '100% VMF mass frac'] / 100  # define total 100% vaporized mass
    intermediate_pct_vmf_mass = total_ejecta_mass * (
            100 - run['0% VMF mass frac'] - run['100% VMF mass frac']) / 100  # define intermediate pct VMF mass
    no_vaporization_mass = total_ejecta_mass * run['0% VMF mass frac'] / 100  # define no vaporization mass
    intermediate_pct_vmf_mass_vapor = intermediate_pct_vmf_mass * run[
        'vmf'] / 100  # define intermediate pct VMF mass vapor
    intermediate_pct_vmf_mass_magma = intermediate_pct_vmf_mass * (
            100 - run['vmf']) / 100  # define intermediate pct VMF mass magma

    assert np.isclose(intermediate_pct_vmf_mass, intermediate_pct_vmf_mass_vapor + intermediate_pct_vmf_mass_magma)

    assert np.isclose(
        total_ejecta_mass,
        total_100_pct_vaporized_mass + intermediate_pct_vmf_mass_vapor + intermediate_pct_vmf_mass_magma + no_vaporization_mass
    )

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
    initial_element_masses = copy.deepcopy(ejecta_mass)

    # first, calculate the mass of the 100% vaporized ejecta for each element
    vaporized_mass = {element: total_100_pct_vaporized_mass * val / 100 for element, val in
                      bse_element_mass_fraction.items()}
    # remove the vaporized mass from the ejecta mass
    ejecta_mass = {element: ejecta_mass[element] - vaporized_mass[element] for element in ejecta_mass.keys()}
    # next, calculate remove the mass of the intermediate VMF vapor from the ejecta mass
    intermediate_pct_vmf_mass_vapor_element_mass = {element: intermediate_pct_vmf_mass_vapor * val / 100 for
                                                    element, val in vapor_element_mass_fraction_at_vmf.items()}
    ejecta_mass = {element: ejecta_mass[element] - intermediate_pct_vmf_mass_vapor_element_mass[element] for element in
                   ejecta_mass.keys()}
    total_melt_mass = copy.copy(ejecta_mass)

    # calculate the total vapor mass for each element
    total_vapor_element_mass = {element: vaporized_mass[element] + intermediate_pct_vmf_mass_vapor_element_mass[element]
                                for element in vaporized_mass.keys()}
    # calculate the lost and retained mass for each element
    lost_mass = {element: total_vapor_element_mass[element] * run['vapor_loss_fraction'] / 100 for element in
                 ejecta_mass.keys()}
    retained_mass = {element: total_vapor_element_mass[element] - lost_mass[element] for element in ejecta_mass.keys()}
    fully_recondensed_ejecta_mass = {element: ejecta_mass[element] + retained_mass[element] for element in
                                     ejecta_mass.keys()}

    element_vmf = {element: total_vapor_element_mass[element] / initial_element_masses[element] * 100 for element in
                   ejecta_mass.keys()}
    lost_vapor_mass_fraction = {element: lost_mass[element] / initial_element_masses[element] * 100 for element in
                                ejecta_mass.keys()}

    assert np.isclose(sum(initial_element_masses.values()),
                      sum(ejecta_mass.values()) + sum(total_vapor_element_mass.values()),
                      total_ejecta_mass,
                      total_100_pct_vaporized_mass + intermediate_pct_vmf_mass_vapor + intermediate_pct_vmf_mass_magma + no_vaporization_mass
                      )

    # convert the ejecta mass back to oxide mass fraction
    ejecta_mass_fraction = normalize(
        ConvertComposition().cations_mass_to_oxides_weight_percent(ejecta_mass, oxides=bse_composition.keys()))
    recondensed_ejecta_mass_fraction = normalize(
        ConvertComposition().cations_mass_to_oxides_weight_percent(fully_recondensed_ejecta_mass,
                                                                   oxides=bse_composition.keys()))

    run['ejecta_mass_fraction'] = ejecta_mass_fraction
    run['recondensed_ejecta_mass_fraction'] = recondensed_ejecta_mass_fraction
    run['element_vmf'] = element_vmf
    run['lost_vapor_mass_fraction'] = lost_vapor_mass_fraction

    for oxide in oxides_ordered:
        append_to_global(global_disk_compositions_no_recondensation, oxide, ejecta_mass_fraction[oxide])
        append_to_global(global_disk_compisitions_with_recondensation, oxide, recondensed_ejecta_mass_fraction[oxide])
    for element in cations_ordered:
        append_to_global(global_element_vmf, element, element_vmf[element])
        append_to_global(global_element_loss_fraction, element, lost_vapor_mass_fraction[element])

fig, ax = plt.subplots(figsize=(10, 10))
# for each oxide, shade between the min and max value of each oxide in all of the lunar bulk composition models
ax.fill_between(
    [format_species_string(oxide) for oxide in oxides_ordered],
    np.array([min(lunar_bulk_compositions.loc[oxide]) / lunar_bulk_compositions["O'Neill 1991"].loc[oxide] for oxide in
              oxides_ordered]),
    np.array([max(lunar_bulk_compositions.loc[oxide]) / lunar_bulk_compositions["O'Neill 1991"].loc[oxide] for oxide in
              oxides_ordered]),
    color='lightgrey',
    alpha=0.8,
)
ax.axhline(1, color='k', linestyle='--', linewidth=2)
for index, run in enumerate(runs):
    # plot the disk composition (no recondensation)
    ax.plot(
        [format_species_string(oxide) for oxide in oxides_ordered],
        [run['ejecta_mass_fraction'][oxide] / lunar_bulk_compositions["O'Neill 1991"].loc[oxide] for oxide in
         oxides_ordered],
        label=run['run_name'],
        marker='o',
        # linestyle='--',
        color=prop_cycle.by_key()['color'][index],
        linewidth=3,
        markersize=10,
    )
    # plot the disk composition (with recondensation)
    ax.plot(
        [format_species_string(oxide) for oxide in oxides_ordered],
        [run['recondensed_ejecta_mass_fraction'][oxide] / lunar_bulk_compositions["O'Neill 1991"].loc[oxide] for oxide
         in oxides_ordered],
        # label=run['run_name'],
        marker='o',
        linestyle='--',
        color=prop_cycle.by_key()['color'][index],
        linewidth=3,
        markersize=10,
    )

# plot the BSE composition
ax.plot(
    [format_species_string(oxide) for oxide in oxides_ordered],
    [bse_composition[oxide] / lunar_bulk_compositions["O'Neill 1991"].loc[oxide] for oxide in oxides_ordered],
    marker='o',
    linewidth=3,
    markersize=10,
    color=prop_cycle.by_key()['color'][2],
    label="BSE",
)

ax.set_ylabel("Disk / BSE (Oxide wt. %)")
ax.set_yscale('log')
ax.grid()
ax.legend()
# increase font size
plt.tight_layout()
# plt.show()
plt.savefig("bse_melt_spider_plot.png", format='png', dpi=200)

# now we will plot the mass fraction of each element in the vapor and magma at the VMF
fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharex='all')
axs = axs.flatten()

for index, run in enumerate(runs):
    axs[0].plot(
        [format_species_string(cation) for cation in cations_ordered],
        [run['element_vmf'][cation] for cation in cations_ordered],
        label=run['run_name'],
        marker='o',
        color=prop_cycle.by_key()['color'][index],
        linewidth=3,
        markersize=10,
    )
    axs[1].plot(
        [format_species_string(cation) for cation in cations_ordered],
        [run['lost_vapor_mass_fraction'][cation] for cation in cations_ordered],
        label=run['run_name'],
        marker='o',
        color=prop_cycle.by_key()['color'][index],
        linewidth=3,
        markersize=10,
    )

axs[0].set_ylabel("Element VMF (%)")
axs[1].set_ylabel("Element Loss Fraction (%)")

for ax in axs:
    ax.grid()
    # ax.set_yscale('log')
    # turn on minor ticks on the y-axis
    # ax.yaxis.set_minor_locator(tck.LogLocator(numticks=999, subs="auto"))
    # # make the ticks larger
    # ax.tick_params(which='minor', width=2, length=4)

axs[0].legend()
plt.tight_layout()
# plt.show()
plt.savefig("bse_vaporize_element_vapor_mass_fraction.png", format='png', dpi=200)

# output the disk composition (no recondensation) to a LaTex table
disk_composition_no_recondensation = pd.DataFrame(global_disk_compositions_no_recondensation).to_latex(index=False)
if "bse_disk_comp_no_recondensation.tex" in os.listdir():
    os.remove("bse_disk_comp_no_recondensation.tex")
with open("bse_disk_comp_no_recondensation.tex", "w") as f:
    f.write(disk_composition_no_recondensation)
disk_composition_with_recondensation = pd.DataFrame(global_disk_compisitions_with_recondensation).to_latex(index=False)
if "bse_disk_comp_with_recondensation.tex" in os.listdir():
    os.remove("bse_disk_comp_with_recondensation.tex")
with open("bse_disk_comp_with_recondensation.tex", "w") as f:
    f.write(disk_composition_with_recondensation)
element_vmf = pd.DataFrame(global_element_vmf).to_latex(index=False)
if "bse_element_vmf.tex" in os.listdir():
    os.remove("bse_element_vmf.tex")
with open("bse_element_vmf.tex", "w") as f:
    f.write(element_vmf)
element_loss_fraction = pd.DataFrame(global_element_loss_fraction).to_latex(index=False)
if "bse_element_loss_fraction.tex" in os.listdir():
    os.remove("bse_element_loss_fraction.tex")
with open("bse_element_loss_fraction.tex", "w") as f:
    f.write(element_loss_fraction)

# increase font size
plt.rcParams.update({"font.size": 20})
# make a 2x2 figure to plot the melt and fractional vapor mass fraction for each element
fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex='all')

for index, run in enumerate(runs):
    melt_composition = collect_data(path=f"{run['run_name']}/magma_oxide_mass_fraction",
                                    x_header='mass fraction vaporized')
    # vapor_composition = collect_data(path=f"{run['run_name']}/partial_pressures",
    #                                     x_header='mass fraction vaporized')
    vapor_composition = collect_data(path=f"{run['run_name']}/total_vapor_species_mass_fraction",
                                     x_header='mass fraction vaporized')
    total_vapor_pressure = {vmf: sum(vapor_composition[vmf].values()) for vmf in vapor_composition.keys()}

    # plot the melt composition on the left column
    melt_species = list(melt_composition[list(melt_composition.keys())[0]].keys())
    vapor_species = list(vapor_composition[list(vapor_composition.keys())[0]].keys())
    # generate a color spectrum for each species
    melt_colors = cm.get_cmap('tab10', len(melt_species))
    vapor_colors = cm.get_cmap('tab10', len(vapor_species))
    for i, species in enumerate(melt_species):
        axs[index, 0].plot(
            [j * 100 for j in melt_composition.keys()],
            [melt_composition[j][species] * 100 for j in melt_composition.keys()],
            label=format_species_string(species),
            color=melt_colors(i),
            linewidth=2,
        )
    for i, species in enumerate(vapor_species):
        axs[index, 1].plot(
            [j * 100 for j in vapor_composition.keys()],
            [vapor_composition[j][species] * 100 for j in vapor_composition.keys()],
            label=format_species_string(species),
            color=vapor_colors(i),
            linewidth=2,
        )
    # axs[index, 1].plot(
    #     np.array(list(total_vapor_pressure.keys())) * 100,
    #     total_vapor_pressure.values(),
    #     label="Total Pressure",
    #     color='k',
    #     linewidth=2,
    # )

    axs[index, 0].set_ylabel("Melt Composition (wt. %)")
    axs[index, 1].set_ylabel("Vapor Composition (wt. %)")
    axs[index, 0].set_ylim(10 ** -2, 100)
    axs[index, 1].set_ylim(10 ** -2, 100)
    axs[index, 0].axvline(run['vmf'], color='k', linestyle='--', linewidth=2)
    axs[index, 1].axvline(run['vmf'], color='k', linestyle='--', linewidth=2)

letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs.flatten()):
    ax.grid()
    labellines.labelLines(ax.get_lines(), zorder=2.5, align=True,
                          xvals=[uniform(10 ** -4, 10 ** 0) for i in ax.get_lines()], fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10 ** -2, 75)
    # annotate a letter in the upper right hand corner of each subplot
    ax.annotate(
        f"{letters[index]}",
        xy=(0.95, 0.95),
        xycoords='axes fraction',
        fontsize=20,
        fontweight='bold',
        horizontalalignment='right',
        verticalalignment='top',
    )

for ax in axs.flatten()[-2:]:
    ax.set_xlabel("VMF (%)")

plt.tight_layout()
plt.savefig("bse_melt_vapor_composition.png", format='png', dpi=200)
