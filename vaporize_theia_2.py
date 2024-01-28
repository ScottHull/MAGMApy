from src.composition import Composition, ConvertComposition, normalize
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data

from theia.chondrites import plot_chondrites, get_enstatite_bulk_theia_core_si_pct
from theia.theia import get_theia_composition, recondense_vapor

from isotopes.rayleigh import FullSequenceRayleighDistillation_SingleReservior

import os
import string
from ast import literal_eval
import numpy as np
import warnings
import pandas as pd
import copy
from math import sqrt
import seaborn as sns
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import labellines
from concurrent.futures import ThreadPoolExecutor, as_completed

# use colorblind-friendly colors
plt.style.use('seaborn-colorblind')
# increase font size
# plt.rcParams.update({"font.size": 20})
# turn off all double scaling warnings

runs = [
    {
        "run_name": "Canonical Model",
        "temperature": 2657.97,  # K
        "vmf": 3.80,  # %
        "0% VMF mass frac": 87.41,  # %
        "100% VMF mass frac": 0.66,  # %
        "disk_theia_mass_fraction": 66.78,  # %
        "disk_mass": 1.02,  # lunar masses
        "vapor_loss_fraction": 74,  # %
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
elements_ordered = ["Al", "Ti", "Ca", "Mg", "Fe", "Si", "K", "Na", "Zn", "O"]
colors = sns.color_palette('husl', n_colors=len(lunar_bulk_compositions.keys()))


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

annotate_models = [
    "Canonical (No Recondensation)",
    "Canonical (With Recondensation)",
    "Half-Earths (No Recondensation)",
    "Half-Earths (With Recondensation)"
]

bulk_ejecta_oxide_outfile = f"ejecta_bulk_oxide_compositions.csv"
bulk_ejecta_elements_outfile = f"ejecta_bulk_element_compositions.csv"
bulk_theia_oxide_outfile = f"theia_bulk_oxide_compositions.csv"
bulk_theia_elements_outfile = f"theia_bulk_element_compositions.csv"
for i in [bulk_ejecta_oxide_outfile, bulk_theia_oxide_outfile]:
    # write the header
    with open(i, "w") as f:
        f.write(f"Run Name,Lunar Model,Recondensation Model,{','.join([oxide for oxide in oxides_ordered])}\n")
for i in [bulk_ejecta_elements_outfile, bulk_theia_elements_outfile]:
    # write the header
    with open(i, "w") as f:
        f.write(f"Run Name,Lunar Model,Recondensation Model,{','.join([element for element in elements_ordered])}\n")


# for run in runs:
#     for lunar_bulk_model in lunar_bulk_compositions.columns:
def __run_model(run, lunar_bulk_model):
    print(f"Target bulk composition: {lunar_bulk_model}")
    for recondense in ['no_recondensation', 'full_recondensation']:
        run_name = f"{run['run_name']}_{lunar_bulk_model}_{recondense}"
        if os.path.exists(run_name):
            return  # skip if the run already exists
        error = 1e99
        residuals = {oxide: 0.0 for oxide in oxides_ordered}
        bulk_ejecta_composition = copy.copy(lunar_bulk_compositions[lunar_bulk_model].to_dict())
        print(f"Running {run['run_name']} with {lunar_bulk_model} lunar bulk composition")
        solution_count = 1
        while (error > 0.15 and run['new_simulation']):
            bulk_ejecta_composition = {oxide: bulk_ejecta_composition[oxide] + residuals[oxide] if (
                    bulk_ejecta_composition[oxide] + residuals[oxide] > 0) else bulk_ejecta_composition[oxide] for
                                       oxide in oxides_ordered}
            bulk_ejecta_composition['Fe2O3'] = 0.0
            bulk_ejecta_composition = normalize(
                {oxide: bulk_ejecta_composition[oxide] for oxide in bse_composition.keys()})
            if run['new_simulation']:
                print(f"Running composition: {bulk_ejecta_composition}")
                # ======================================= RUN MAGMApy IMULATION ====================================
                c = Composition(
                    composition=bulk_ejecta_composition
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

                reports = Report(composition=c, liquid_system=l, gas_system=g, thermosystem=t, to_dir=run_name)

                print(f"Starting MAGMApy loop")
                count = 1
                while t.weight_fraction_vaporized < 0.80:
                    # print("Running MAGMApy iteration", count)
                    l.calculate_activities(temperature=run['temperature'])
                    g.calculate_pressures(temperature=run['temperature'], liquid_system=l)
                    if l.counter == 1:
                        l.calculate_activities(temperature=run['temperature'])
                        g.calculate_pressures(temperature=run['temperature'], liquid_system=l)
                    t.vaporize()
                    l.counter = 0  # reset Fe2O3 counter for next vaporization step
                    if (count % 20 == 0 or count == 1):
                        reports.create_composition_report(iteration=count)
                        reports.create_liquid_report(iteration=count)
                        reports.create_gas_report(iteration=count)
                    count += 1
                print("MAGMApy simulation complete")

                # ============================== ASSESS MODEL ERROR =================================

                # calculate the mass distribution between the desired populations (in kg)
                total_ejecta_mass = run['disk_mass'] * MASS_MOON  # define total ejecta mass, kg
                unvaporized_ejecta_mass = total_ejecta_mass * run['0% VMF mass frac'] / 100  # define unvaporized mass
                total_100_pct_vaporized_mass = total_ejecta_mass * run[
                    '100% VMF mass frac'] / 100  # define total 100% vaporized mass
                intermediate_pct_vmf_mass = total_ejecta_mass * (100 - run['0% VMF mass frac'] - run[
                    '100% VMF mass frac']) / 100  # define intermediate pct VMF mass
                intermediate_pct_vmf_mass_vapor = intermediate_pct_vmf_mass * run[
                    'vmf'] / 100  # define intermediate pct VMF mass vapor
                intermediate_pct_vmf_mass_magma = intermediate_pct_vmf_mass * (
                        100 - run['vmf']) / 100  # define intermediate pct VMF mass magma
                total_bse_sourced_mass = total_ejecta_mass * (1 - run['disk_theia_mass_fraction'] / 100)
                total_theia_sourced_mass = total_ejecta_mass * run['disk_theia_mass_fraction'] / 100

                # make sure the total mass is conserved
                assert np.isclose(total_ejecta_mass,
                                 unvaporized_ejecta_mass + total_100_pct_vaporized_mass + intermediate_pct_vmf_mass_vapor + intermediate_pct_vmf_mass_magma
                                 )

                # read in the data
                melt_oxide_mass_fraction = collect_data(path=f"{run_name}/magma_oxide_mass_fraction",
                                                        x_header='mass fraction vaporized')
                magma_element_mass = collect_data(path=f"{run_name}/magma_element_mass",
                                                  x_header='mass fraction vaporized')
                vapor_element_mass = collect_data(path=f"{run_name}/total_vapor_element_mass",
                                                  x_header='mass fraction vaporized')

                # get the composition at the VMF
                melt_oxide_mass_fraction_at_vmf = normalize(get_composition_at_vmf(
                    melt_oxide_mass_fraction,
                    run['vmf'] / 100
                ))
                magma_element_mass_fraction_at_vmf = normalize(get_composition_at_vmf(
                    magma_element_mass,
                    run['vmf'] / 100
                ))
                vapor_element_mass_fraction_at_vmf = normalize(get_composition_at_vmf(
                    vapor_element_mass,
                    run['vmf'] / 100
                ))

                run[run_name] = {}
                run[run_name].update({'bulk_ejecta_composition': bulk_ejecta_composition})
                run[run_name].update(
                    {'target_lunar_bulk_composition': lunar_bulk_compositions[lunar_bulk_model].to_dict()})

                # store the bulk masses
                run[run_name].update(
                    {'total_ejecta_mass': total_ejecta_mass, 'bse_sourced_mass': total_bse_sourced_mass,
                     'theia_sourced_mass': total_theia_sourced_mass,
                     'total_100_pct_vaporized_mass': total_100_pct_vaporized_mass,
                     'intermediate_pct_vmf_mass': intermediate_pct_vmf_mass,
                     'intermediate_pct_vmf_mass_vapor': intermediate_pct_vmf_mass_vapor,
                     'intermediate_pct_vmf_mass_magma': intermediate_pct_vmf_mass_magma,
                     'unvaporized_ejecta_mass': unvaporized_ejecta_mass})

                # first, calculate the total ejecta mass for each element
                ejecta_mass = {element: total_ejecta_mass * val / 100 for element, val in
                               ConvertComposition().oxide_wt_pct_to_cation_wt_pct(bulk_ejecta_composition,
                                                                                  include_oxygen=True).items()}

                bse_element_mass = {element: total_bse_sourced_mass * val / 100 for element, val in
                                    bse_element_mass_fraction.items()}
                theia_element_mass = {element: ejecta_mass[element] - bse_element_mass[element] for element in
                                      ejecta_mass.keys()}
                theia_composition = normalize(
                    ConvertComposition().cations_mass_to_oxides_weight_percent(theia_element_mass,
                                                                               oxides=oxides_ordered))

                run[run_name].update({'bse_element_mass': bse_element_mass,
                                      'theia_element_mass': theia_element_mass,
                                      'theia_composition': theia_composition})

                run[run_name].update({'total_ejecta_element_mass_before_vaporization': copy.copy(ejecta_mass)})

                # next, calculate the total vapor mass from the 100% and intermediate vaporized pools
                fully_vaporized_element_vapor_mass = {element: total_100_pct_vaporized_mass * val / 100 for element, val
                                                      in
                                                      ConvertComposition().oxide_wt_pct_to_cation_wt_pct(
                                                          bulk_ejecta_composition, include_oxygen=True).items()}
                intermediate_vaporized_element_vapor_mass = {element: intermediate_pct_vmf_mass_vapor * val / 100 for
                                                             element, val in
                                                             vapor_element_mass_fraction_at_vmf.items()}
                total_vapor_mass = {
                    element: fully_vaporized_element_vapor_mass[element] + intermediate_vaporized_element_vapor_mass[
                        element] for element in fully_vaporized_element_vapor_mass.keys()}

                run[run_name].update({'fully_vaporized_element_vapor_mass': fully_vaporized_element_vapor_mass,
                                      'intermediate_vaporized_element_vapor_mass': intermediate_vaporized_element_vapor_mass,
                                      'total_vapor_mass': total_vapor_mass})

                # next, remove the vapor mass from the ejecta mass
                ejecta_mass = {element: ejecta_mass[element] - total_vapor_mass[element] for element in
                               ejecta_mass.keys()}

                run[run_name].update({'total_ejecta_mass_after_vapor_removal_without_recondensation': ejecta_mass})

                # calculate the lost and retained vapor mass following hydrodynamic escape
                lost_vapor_mass = {element: total_vapor_mass[element] * run['vapor_loss_fraction'] / 100 for element in
                                   ejecta_mass.keys()}
                retained_vapor_mass = {element: total_vapor_mass[element] - lost_vapor_mass[element] for element in
                                       ejecta_mass.keys()}

                run[run_name].update({'lost_vapor_mass': lost_vapor_mass, 'retained_vapor_mass': retained_vapor_mass})

                if "full_recondensation" in run_name:
                    # add back in the retained vapor mass to the ejecta mass
                    ejecta_mass = {element: ejecta_mass[element] + retained_vapor_mass[element] for element in
                                   ejecta_mass.keys()}
                    run[run_name].update({'total_ejecta_mass_after_vapor_removal_with_recondensation': ejecta_mass})

                # convert the ejecta mass back to oxide mass fraction
                ejecta_mass_fraction = normalize(ConvertComposition().cations_mass_to_oxides_weight_percent(ejecta_mass,
                                                                                                            oxides=bse_composition.keys()))

                fname = f"{run_name}_theia_mixing_model.csv"
                if os.path.exists(fname):
                    os.remove(fname)
                # the data is a dictionary of dictionaries.  output as a csv, where the first key is the left column and the inner keys are the headers
                with open(fname, "w") as f:
                    f.write(str(run[run_name]))
                f.close()

                # calculate error residuals
                residuals = {
                    oxide: lunar_bulk_compositions[lunar_bulk_model].loc[oxide] - ejecta_mass_fraction[oxide]
                    for oxide in oxides_ordered}
                error = sum([abs(i) for i in residuals.values()])
                run[run_name].update({'error': error, 'residuals': residuals})
                print(f"Error: {error}, residuals: {residuals}")
                solution_count += 1

            else:
                print(f"No solution found, trying again... ({solution_count}, error: {error})")

        print(f"Solution found! {solution_count} iterations")


# num_workers = len(lunar_bulk_compositions.columns)
# num_workers = 1
# for run in runs:
#     if num_workers > 1:
#         with ThreadPoolExecutor(max_workers=num_workers) as executor:
#             futures = {}
#             for lbc in lunar_bulk_compositions.columns:
#                 futures.update({executor.submit(__run_model, run, lbc): run['run_name'] + "_" + str(lbc)})
#
#             for future in as_completed(futures):
#                 r = futures[future]
#                 try:
#                     data = future.result()
#                 except Exception as exc:
#                     print('%r generated an exception: %s' % (r, exc))
#     else:
#         for lbc in lunar_bulk_compositions.columns:
#             __run_model(run, lbc)


# ================================== THEIA MG/SI VS MG/AL ==================================
# https://www.lpi.usra.edu/books/MESSII/9039.pdf
# See Figure 6
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
# add chondrites
plot_chondrites(ax)
found_base_models = []
# generate a list of 4 different markers
markers = ['o', 's', 'D', '^']
bse_element_masses = ConvertComposition().oxide_wt_to_cation_wt(bse_composition)
bse_mg_si = bse_element_masses["Mg"] / bse_element_masses["Si"]
bse_al_si = bse_element_masses["Al"] / bse_element_masses["Si"]
ax.scatter(
    bse_al_si, bse_mg_si, color="k", s=300, marker="*"
)
# annotate the BSE
ax.annotate(
    "BSE", xy=(bse_al_si, bse_mg_si), xycoords="data", xytext=(bse_al_si + 0.005, bse_mg_si + 0.005), fontsize=14
)
# plot the Mg/Si vs Mg/Al for each of the modelled BST compositions
found_models = []
for index, s in enumerate(lunar_bulk_compositions.keys()):
    for run in runs:
        for recondense in ['no_recondensation', 'full_recondensation']:
            fname = f"{run['run_name']}_{s}_{recondense}_theia_mixing_model.csv"
            # read in the data as a dictionary
            data = literal_eval(open(fname, 'r').read())
            label = None
            marker = None
            if s not in found_base_models:
                label = s
                found_base_models.append(s)
                ax.scatter([], [], color=colors[list(lunar_bulk_compositions).index(s)], s=100, marker="s", label=label)
            if "no_recondensation" in fname and "Canonical" in fname:
                marker = markers[0]
            elif "full_recondensation" in fname and "Canonical" in fname:
                marker = markers[1]
            elif "no_recondensation" in fname and "Half Earths" in fname:
                marker = markers[2]
            elif "full_recondensation" in fname and "Half Earths" in fname:
                marker = markers[3]
            # read in the theia composition file
            theia_composition = data['theia_composition']
            # convert bulk oxide masses to bulk element masses
            theia_element_masses = ConvertComposition().oxide_wt_to_cation_wt(theia_composition)
            # get the Mg/Si and Mg/Al ratios
            mg_si = theia_element_masses['Mg'] / theia_element_masses['Si']
            al_si = theia_element_masses['Al'] / theia_element_masses['Si']
            # scatter the Mg/Si vs Al/Si
            ax.scatter(al_si, mg_si, color=colors[list(lunar_bulk_compositions).index(s)], s=100, marker=marker,
                       edgecolor='k')
for m, model in zip(markers,
                    ["Canonical (No Recondensation)", "Canonical (Recondensed)", "Half-Earths (No Recondensation)",
                     "Half-Earths (Recondensed)"]):
    ax.scatter([], [], color='k', s=100, marker=m, label=model)

ax.set_xlabel("Al/Si (mass ratio)", fontsize=20)
ax.set_ylabel("Mg/Si (mass ratio)", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlim(.018, 0.24)
ax.grid()
ax.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.savefig("theia_mg_si_vs_al_si.png", dpi=300)

# ==================================== PLOT MG/SI AND AL/SI FOR BULK THEIA ASSUMING ENSTATITE START ==================
fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey='all')
axs = axs.flatten()
# add chondrites
# plot_chondrites(ax)
found_models = []
# generate a list of 4 different markers
markers = ['o', 's', 'D', '^']
bse_element_masses = ConvertComposition().oxide_wt_to_cation_wt(bse_composition)
bse_mg_si = bse_element_masses["Mg"] / bse_element_masses["Si"]
bse_al_si = bse_element_masses["Al"] / bse_element_masses["Si"]
bulk_earth_mg = 15.4
bulk_earth_al = 1.59
bulk_earth_si = 16.1
ax.scatter(
    bse_al_si, bse_mg_si, color="k", s=300, marker="*"
)
# annotate the BSE and bulk Earth
ax.annotate(
    "BSE", xy=(bse_al_si, bse_mg_si), xycoords="data", xytext=(bse_al_si + 0.002, bse_mg_si + 0.002), fontsize=14
)
# plot the Mg/Si vs Mg/Al for each of the modelled BST compositions
for index, s in enumerate(lunar_bulk_compositions.keys()):
    for run in runs:
        for recondense in ['no_recondensation', 'full_recondensation']:
            fname = f"{run['run_name']}_{s}_{recondense}_theia_mixing_model.csv"
            data = literal_eval(open(fname, 'r').read())
            label = None
            marker = None
            if s not in found_base_models:
                label = s
                found_base_models.append(s)
                axs[1].scatter([], [], color=colors[list(lunar_bulk_compositions).index(s)], s=100, marker="s", label=label)
            if "no_recondensation" in fname and "Canonical" in fname:
                marker = markers[0]
            elif "full_recondensation" in fname and "Canonical" in fname:
                marker = markers[1]
            elif "no_recondensation" in fname and "Half Earths" in fname:
                marker = markers[2]
            elif "full_recondensation" in fname and "Half Earths" in fname:
                marker = markers[3]
            # read in the theia composition file
            theia_composition = data['theia_composition']
            # get the enstatite-based Theia Mg/Si and Mg/Al ratios as a function of Si core wt%
            shade = None
            if index == 0:
                shade = axs[1]
            pct_si_in_core, mg_si_bulk_theia, al_si_bulk_theia = get_enstatite_bulk_theia_core_si_pct(theia_composition,
                                                                                                      ax=shade)
            # scatter the Mg/Si vs Al/Si
            axs[0].scatter(mg_si_bulk_theia, pct_si_in_core, color=colors[list(lunar_bulk_compositions).index(s)], s=100,
                           marker=marker, edgecolor='k')
            axs[1].scatter(al_si_bulk_theia, pct_si_in_core, color=colors[list(lunar_bulk_compositions).index(s)], s=100,
                           marker=marker, edgecolor='k')
for m, model in zip(markers,
                    ["Canonical (No Recondensation)", "Canonical (Recondensed)", "Half-Earths (No Recondensation)",
                     "Half-Earths (Recondensed)"]):
    axs[1].scatter([], [], color='k', s=100, marker=m, label=model)

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid()
axs[0].set_xlabel("Mg/Si (mass ratio)", fontsize=20)
axs[1].set_xlabel("Al/Si (mass ratio)", fontsize=20)
axs[0].set_ylabel("Si Core Mass Fraction (%)", fontsize=20)
axs[1].legend(fontsize=14)
plt.tight_layout()
plt.savefig("enstatite_theia_si_in_core.png", dpi=300)



# ======================= PLOT BULK EJECTA COMPOSITIONS =======================
fig, axs = plt.subplots(2, 2, figsize=(25, 15), sharex='all', sharey='all')
axs = axs.flatten()
# axs[0].set_title("Ejecta Bulk Composition (Without Recondensation)", fontsize=18)
# axs[1].set_title("Ejecta Bulk Composition (With Recondensation)", fontsize=18)
for index, ax in enumerate(axs):
    ax.grid()
    label = None
    if index == 0:
        label = "BSE"
    ax.axhline(y=1, color="black", linewidth=4, alpha=1, label=label)
for i, s in enumerate(lunar_bulk_compositions.keys()):
    for run in runs:
        for recondense in ['no_recondensation', 'full_recondensation']:
            fname = f"{run['run_name']}_{s}_{recondense}_theia_mixing_model.csv"
            data = literal_eval(open(fname, 'r').read())
            ejecta_composition = data['bulk_ejecta_composition']
            to_index = 1
            label = None
            if "no_recondensation" in fname:
                to_index = 0
            if "Half Earths" in fname:
                to_index += 2
            if to_index == 0:
                label = s
            axs[to_index].plot(
                oxides_ordered, [ejecta_composition[oxide] / bse_composition[oxide] for oxide in oxides_ordered],
                color=colors[list(lunar_bulk_compositions).index(s)], marker='o', markersize=8,
                linewidth=2.0, label=label
            )

# set minimum plotted x value
letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs):
    # label each subplot with a letter in the upper-left corner
    ax.annotate(
        letters[index], xy=(0.05, 0.95), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontweight="bold", fontsize=20
    )
    ax.annotate(
        annotate_models[index], xy=(0.05, 0.90), xycoords="axes fraction", horizontalalignment="left",
        verticalalignment="top",
        fontsize=18
    )

# fig.supylabel("Bulk Composition / BSE Composition", fontsize=18)
for ax in [axs[0], axs[2]]:
    ax.set_ylabel("Bulk Composition / BSE Composition", fontsize=18)
# replace the x-axis labels with the formatted oxide names
for ax in axs[-2:]:
    ax.set_xticklabels([format_species_string(oxide) for oxide in oxides_ordered], rotation=45)
# set the axis font size to be 16 for each subplot
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
legend = fig.legend(loc=7, fontsize=17)
for line in legend.get_lines():
    line.set_linewidth(4.0)
fig.subplots_adjust(right=0.76)
# add legend to the right of the figure
plt.savefig("theia_mixing_ejecta_compositions.png", dpi=300)

# ======================= PLOT BULK THEIA COMPOSITIONS =======================
fig, axs = plt.subplots(2, 2, figsize=(25, 15), sharex='all', sharey='all')
axs = axs.flatten()
# axs[0].set_title("Ejecta Bulk Composition (Without Recondensation)", fontsize=18)
# axs[1].set_title("Ejecta Bulk Composition (With Recondensation)", fontsize=18)
for index, ax in enumerate(axs):
    ax.grid()
    label = None
    if index == 0:
        label = "BSE"
    ax.axhline(y=1, color="black", linewidth=4, alpha=1, label=label)
for i, s in enumerate(lunar_bulk_compositions.keys()):
    for run in runs:
        for recondense in ['no_recondensation', 'full_recondensation']:
            fname = f"{run['run_name']}_{s}_{recondense}_theia_mixing_model.csv"
            data = literal_eval(open(fname, 'r').read())
            theia_composition = data['theia_composition']
            to_index = 1
            label = None
            if "no_recondensation" in fname:
                to_index = 0
            if "Half Earths" in fname:
                to_index += 2
            if to_index == 0:
                label = s
            axs[to_index].plot(
                oxides_ordered, [theia_composition[oxide] / bse_composition[oxide] for oxide in oxides_ordered],
                color=colors[list(lunar_bulk_compositions).index(s)], marker='o', markersize=8,
                linewidth=2.0, label=label
            )

for ax in axs:
    ax.fill_between(oxides_ordered, [0 for oxide in oxides_ordered], [-1e99 for oxide in oxides_ordered],
                    alpha=0.2, color='red')

# set minimum plotted x value
letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs):
    # label each subplot with a letter in the upper-left corner
    ax.annotate(
        letters[index], xy=(0.05, 0.95), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontweight="bold", fontsize=20
    )
    ax.annotate(
        annotate_models[index], xy=(0.05, 0.90), xycoords="axes fraction", horizontalalignment="left",
        verticalalignment="top",
        fontsize=18
    )

# fig.supylabel("Bulk Composition / BSE Composition", fontsize=18)
for ax in [axs[0], axs[2]]:
    ax.set_ylabel("Bulk Composition / BSE Composition", fontsize=18)
# replace the x-axis labels with the formatted oxide names
for ax in axs[-2:]:
    ax.set_xticklabels([format_species_string(oxide) for oxide in oxides_ordered], rotation=45)
# set the axis font size to be 16 for each subplot
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_ylim(bottom=-1.0, top=4.2)

plt.tight_layout()
legend = fig.legend(loc=7, fontsize=17)
for line in legend.get_lines():
    line.set_linewidth(4.0)
fig.subplots_adjust(right=0.76)
# add legend to the right of the figure
plt.savefig("theia_mixing_theia_compositions.png", dpi=300)



# ======================= PLOT THEIA EJECTA VMF AND LOSS FRACTION =======================
# make a 2 column 4 row figure
fig, axs = plt.subplots(4, 2, figsize=(16, 20), sharex='all', sharey='all')
axs = axs.flatten()
for i, s in enumerate(lunar_bulk_compositions.keys()):
    label = s
    if "Fractional" or "Equilibrium" in s:
        # add a newline in the space preceding the word "Equilibrium" or "Fractional"
        label = s.replace("Fractional", "\nFractional").replace("Equilibrium", "\nEquilibrium")
    axs[0].plot([], [], color=colors[i], marker='o', markersize=8, linewidth=4.0, label=label)
    for run in runs:
        for recondense in ['no_recondensation', 'full_recondensation']:
            to_index = 0
            fname = f"{run['run_name']}_{s}_{recondense}_theia_mixing_model.csv"
            if "Half Earths" in fname:
                to_index += 1
            if "full_recondensation" in fname:
                to_index += 2
            data = literal_eval(open(fname, 'r').read())
            ejecta_element_mass = {element: val / 100 * data['total_ejecta_mass'] for element, val in
                                   normalize(ConvertComposition().oxide_wt_pct_to_cation_wt_pct(
                                       data['bulk_ejecta_composition'], include_oxygen=True)).items()}
            elemental_vmf = {element: data['total_vapor_mass'][element] / ejecta_element_mass[element] * 100 for element
                             in ejecta_element_mass.keys()}

            elemental_hydrodynamic_loss_fraction = {
                element: (data['total_vapor_mass'][element] * run['vapor_loss_fraction'] / 100) / ejecta_element_mass[
                    element] * 100 for element in ejecta_element_mass.keys()}
            axs[to_index].plot(
                elements_ordered[:-1], [elemental_vmf[element] for element in elements_ordered[:-1]],
                color=colors[list(lunar_bulk_compositions).index(s)], marker='o', markersize=8,
                linewidth=2.0
            )
            axs[to_index + 4].plot(
                elements_ordered[:-1], [elemental_hydrodynamic_loss_fraction[element] for element in
                                        elements_ordered[:-1]],
                color=colors[list(lunar_bulk_compositions).index(s)], marker='o', markersize=8,
                linewidth=2.0
            )

for index, ax in enumerate(axs[0:4]):
    if index % 2 == 0:
        ax.set_ylabel("VMF (%)", fontsize=20)
for index, ax in enumerate(axs[4:]):
    if index % 2 == 0:
        ax.set_ylabel("Vapor Mass Loss Fraction (%)", fontsize=20)
letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs):
    ax.grid()
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=20)
    # add minor ticks to the y axis
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(LogLocator(numticks=999, subs="auto"))
    # make the ticks larger
    ax.tick_params(axis='y', which='both', width=2, length=6)
    # annotate each subplot with a letter in the upper-left corner
    ax.annotate(
        letters[index], xy=(0.05, 0.95), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontweight="bold", fontsize=20
    )

for index, i in enumerate([
    "Canonical (No Recondensation)", "Canonical (Recondensed)", "Half-Earths (No Recondensation)",
    "Half-Earths (Recondensed)"]):
    axs[index].set_title(i, fontsize=20)
    axs[index + 4].set_title(i, fontsize=20)

plt.tight_layout()
legend = fig.legend(loc=7, fontsize=16)
for line in legend.get_lines():
    line.set_linewidth(4.0)
fig.subplots_adjust(right=0.74)
plt.savefig("theia_mixing_vmf_and_loss_fraction.png", dpi=200)


# ======================= OUTPUT THEIA AND EJECTA BULK COMPOSITIONS TO LATEX TABLE =======================
global_ejecta_no_recondensation_canonical = {}
global_ejecta_full_recondensation_canonical = {}
global_theia_no_recondensation_canonical = {}
global_theia_full_recondensation_canonical = {}
global_ejecta_no_recondensation_half_earths = {}
global_ejecta_full_recondensation_half_earths = {}
global_theia_no_recondensation_half_earths = {}
global_theia_full_recondensation_half_earths = {}
global_element_vmf_canonical_no_reconensation = {}
global_element_vmf_half_earths_no_reconensation = {}
global_element_loss_fraction_canonical_no_reconensation = {}
global_element_loss_fraction_half_earth_no_reconensations = {}
global_element_vmf_canonical_full_reconensation = {}
global_element_vmf_half_earths_full_reconensation = {}
global_element_loss_fraction_canonical_full_reconensation = {}
global_element_loss_fraction_half_earth_full_reconensations = {}
for i in [global_ejecta_no_recondensation_canonical, global_ejecta_full_recondensation_canonical,
            global_theia_no_recondensation_canonical, global_theia_full_recondensation_canonical,
            global_ejecta_no_recondensation_half_earths, global_ejecta_full_recondensation_half_earths,
            global_theia_no_recondensation_half_earths, global_theia_full_recondensation_half_earths]:
    i.update({'run_name': [s for s in lunar_bulk_compositions.keys()]})
    i.update({oxide: [] for oxide in oxides_ordered})
for i in [global_element_vmf_canonical_no_reconensation, global_element_vmf_half_earths_no_reconensation,
            global_element_loss_fraction_canonical_no_reconensation, global_element_loss_fraction_half_earth_no_reconensations,
            global_element_vmf_canonical_full_reconensation, global_element_vmf_half_earths_full_reconensation,
            global_element_loss_fraction_canonical_full_reconensation, global_element_loss_fraction_half_earth_full_reconensations]:
    i.update({'run_name': [s for s in lunar_bulk_compositions.keys()]})
    i.update({element: [] for element in elements_ordered[:-1]})
    
def format_val(val):
    """
    Take the value and return as a string.  If val < 0.1, then return as scientific notation.  Otherwise, return as
    decimal.
    :param val: 
    :return: 
    """
    if val < 0.1:
        return f"{val:.2e}"
    else:
        return f"{val:.2f}"
    
for run in runs:
    for recondense in ['no_recondensation', 'full_recondensation']:
        for s in lunar_bulk_compositions.keys():
            fname = f"{run['run_name']}_{s}_{recondense}_theia_mixing_model.csv"
            data = literal_eval(open(fname, 'r').read())
            ejecta_composition = data['bulk_ejecta_composition']
            theia_composition = data['theia_composition']
            total_ejecta_mass = data['total_ejecta_mass']
            bse_source_mass = data['bse_sourced_mass']
            theia_source_mass = data['theia_sourced_mass']

            fully_vaporized_element_mass = data['fully_vaporized_element_vapor_mass']
            intermediate_vaporized_element_mass = data['intermediate_vaporized_element_vapor_mass']

            # do some mass balance checks
            # first, make sure that bukl ejecta = bse source mass + theia source mass
            assert np.isclose(
                total_ejecta_mass, bse_source_mass + theia_source_mass
            )
            total_vapor_mass = sum(data['total_vapor_mass'].values())
            total_melt_mass_no_recondensation = sum(data['total_ejecta_mass_after_vapor_removal_without_recondensation'].values())
            # make sure that the total ejecta mass after vapor removal is equal to the total vapor mass plus the total melt mass
            assert np.isclose(
                total_ejecta_mass, total_vapor_mass + total_melt_mass_no_recondensation
            )


            # next, we are going to calculate the VMF and hydrodynamic loss fraction of each element
            ejecta_element_mass = {element: val / 100 * data['total_ejecta_mass'] for element, val in
                                   normalize(ConvertComposition().oxide_wt_pct_to_cation_wt_pct(data['bulk_ejecta_composition'], include_oxygen=True)).items()}
            elemental_vmf = {element: data['total_vapor_mass'][element] / ejecta_element_mass[element] * 100 for element in ejecta_element_mass.keys()}

            elemental_hydrodynamic_loss_fraction = {element: (data['total_vapor_mass'][element] * run['vapor_loss_fraction'] / 100) / ejecta_element_mass[element] * 100 for element in ejecta_element_mass.keys()}

            if "no_recondensation" in fname:
                if "Canonical" in fname:
                    for oxide in oxides_ordered:
                        global_ejecta_no_recondensation_canonical[oxide].append(format_val(ejecta_composition[oxide]))
                        global_theia_no_recondensation_canonical[oxide].append(format_val(theia_composition[oxide]))
                    for cation in elements_ordered[:-1]:
                        global_element_vmf_canonical_no_reconensation[cation].append(format_val(elemental_vmf[cation]))
                        global_element_loss_fraction_canonical_no_reconensation[cation].append(format_val(elemental_hydrodynamic_loss_fraction[cation]))
                elif "Half Earths" in fname:
                    for oxide in oxides_ordered:
                        global_ejecta_no_recondensation_half_earths[oxide].append(format_val(ejecta_composition[oxide]))
                        global_theia_no_recondensation_half_earths[oxide].append(format_val(theia_composition[oxide]))
                    for cation in elements_ordered[:-1]:
                        global_element_vmf_half_earths_no_reconensation[cation].append(format_val(elemental_vmf[cation]))
                        global_element_loss_fraction_half_earth_no_reconensations[cation].append(format_val(elemental_hydrodynamic_loss_fraction[cation]))
            elif "full_recondensation" in fname:
                if "Canonical" in fname:
                    for oxide in oxides_ordered:
                        global_ejecta_full_recondensation_canonical[oxide].append(format_val(ejecta_composition[oxide]))
                        global_theia_full_recondensation_canonical[oxide].append(format_val(theia_composition[oxide]))
                    for cation in elements_ordered[:-1]:
                        global_element_vmf_canonical_full_reconensation[cation].append(format_val(elemental_vmf[cation]))
                        global_element_loss_fraction_canonical_full_reconensation[cation].append(format_val(elemental_hydrodynamic_loss_fraction[cation]))
                elif "Half Earths" in fname:
                    for oxide in oxides_ordered:
                        global_ejecta_full_recondensation_half_earths[oxide].append(format_val(ejecta_composition[oxide]))
                        global_theia_full_recondensation_half_earths[oxide].append(format_val(theia_composition[oxide]))
                    for cation in elements_ordered[:-1]:
                        global_element_vmf_half_earths_full_reconensation[cation].append(format_val(elemental_vmf[cation]))
                        global_element_loss_fraction_half_earth_full_reconensations[cation].append(format_val(elemental_hydrodynamic_loss_fraction[cation]))


global_ejecta_no_recondensation_canonical_df = pd.DataFrame(global_ejecta_no_recondensation_canonical).to_latex(index=False)
if "global_ejecta_no_recondensation_canonical.tex" in os.listdir():
    os.remove("global_ejecta_no_recondensation_canonical.tex")
with open("global_ejecta_no_recondensation_canonical.tex", "w") as f:
    f.write(global_ejecta_no_recondensation_canonical_df)
f.close()
global_ejecta_full_recondensation_canonical_df = pd.DataFrame(global_ejecta_full_recondensation_canonical).to_latex(index=False)
if "global_ejecta_full_recondensation_canonical.tex" in os.listdir():
    os.remove("global_ejecta_full_recondensation_canonical.tex")
with open("global_ejecta_full_recondensation_canonical.tex", "w") as f:
    f.write(global_ejecta_full_recondensation_canonical_df)
f.close()
global_theia_no_recondensation_canonical_df = pd.DataFrame(global_theia_no_recondensation_canonical).to_latex(index=False)
if "global_theia_no_recondensation_canonical.tex" in os.listdir():
    os.remove("global_theia_no_recondensation_canonical.tex")
with open("global_theia_no_recondensation_canonical.tex", "w") as f:
    f.write(global_theia_no_recondensation_canonical_df)
f.close()
global_theia_full_recondensation_canonical_df = pd.DataFrame(global_theia_full_recondensation_canonical).to_latex(index=False)
if "global_theia_full_recondensation_canonical.tex" in os.listdir():
    os.remove("global_theia_full_recondensation_canonical.tex")
with open("global_theia_full_recondensation_canonical.tex", "w") as f:
    f.write(global_theia_full_recondensation_canonical_df)
f.close()
global_ejecta_no_recondensation_half_earths_df = pd.DataFrame(global_ejecta_no_recondensation_half_earths).to_latex(index=False)
if "global_ejecta_no_recondensation_half_earths.tex" in os.listdir():
    os.remove("global_ejecta_no_recondensation_half_earths.tex")
with open("global_ejecta_no_recondensation_half_earths.tex", "w") as f:
    f.write(global_ejecta_no_recondensation_half_earths_df)
f.close()
global_ejecta_full_recondensation_half_earths_df = pd.DataFrame(global_ejecta_full_recondensation_half_earths).to_latex(index=False)
if "global_ejecta_full_recondensation_half_earths.tex" in os.listdir():
    os.remove("global_ejecta_full_recondensation_half_earths.tex")
with open("global_ejecta_full_recondensation_half_earths.tex", "w") as f:
    f.write(global_ejecta_full_recondensation_half_earths_df)
f.close()
global_theia_no_recondensation_half_earths_df = pd.DataFrame(global_theia_no_recondensation_half_earths).to_latex(index=False)
if "global_theia_no_recondensation_half_earths.tex" in os.listdir():
    os.remove("global_theia_no_recondensation_half_earths.tex")
with open("global_theia_no_recondensation_half_earths.tex", "w") as f:
    f.write(global_theia_no_recondensation_half_earths_df)
f.close()
global_theia_full_recondensation_half_earths_df = pd.DataFrame(global_theia_full_recondensation_half_earths).to_latex(index=False)
if "global_theia_full_recondensation_half_earths.tex" in os.listdir():
    os.remove("global_theia_full_recondensation_half_earths.tex")
with open("global_theia_full_recondensation_half_earths.tex", "w") as f:
    f.write(global_theia_full_recondensation_canonical_df)
f.close()
global_element_vmf_canonical_no_reconensation_df = pd.DataFrame(global_element_vmf_canonical_no_reconensation).to_latex(index=False)
if "global_element_vmf_canonical_no_reconensation.tex" in os.listdir():
    os.remove("global_element_vmf_canonical_no_reconensation.tex")
with open("global_element_vmf_canonical_no_reconensation.tex", "w") as f:
    f.write(global_element_vmf_canonical_no_reconensation_df)
f.close()
global_element_vmf_half_earths_no_reconensation_df = pd.DataFrame(global_element_vmf_half_earths_no_reconensation).to_latex(index=False)
if "global_element_vmf_half_earths_no_reconensation.tex" in os.listdir():
    os.remove("global_element_vmf_half_earths_no_reconensation.tex")
with open("global_element_vmf_half_earths_no_reconensation.tex", "w") as f:
    f.write(global_element_vmf_half_earths_no_reconensation_df)
f.close()
global_element_loss_fraction_canonical_no_reconensation_df = pd.DataFrame(global_element_loss_fraction_canonical_no_reconensation).to_latex(index=False)
if "global_element_loss_fraction_canonical_no_reconensation.tex" in os.listdir():
    os.remove("global_element_loss_fraction_canonical_no_reconensation.tex")
with open("global_element_loss_fraction_canonical_no_reconensation.tex", "w") as f:
    f.write(global_element_loss_fraction_canonical_no_reconensation_df)
f.close()
global_element_loss_fraction_half_earth_no_reconensations_df = pd.DataFrame(global_element_loss_fraction_half_earth_no_reconensations).to_latex(index=False)
if "global_element_loss_fraction_half_earth_no_reconensations.tex" in os.listdir():
    os.remove("global_element_loss_fraction_half_earth_no_reconensations.tex")
with open("global_element_loss_fraction_half_earth_no_reconensations.tex", "w") as f:
    f.write(global_element_loss_fraction_half_earth_no_reconensations_df)
f.close()
global_element_vmf_canonical_full_reconensation_df = pd.DataFrame(global_element_vmf_canonical_full_reconensation).to_latex(index=False)
if "global_element_vmf_canonical_full_reconensation.tex" in os.listdir():
    os.remove("global_element_vmf_canonical_full_reconensation.tex")
with open("global_element_vmf_canonical_full_reconensation.tex", "w") as f:
    f.write(global_element_vmf_canonical_full_reconensation_df)
f.close()
global_element_vmf_half_earths_full_reconensation_df = pd.DataFrame(global_element_vmf_half_earths_full_reconensation).to_latex(index=False)
if "global_element_vmf_half_earths_full_reconensation.tex" in os.listdir():
    os.remove("global_element_vmf_half_earths_full_reconensation.tex")
with open("global_element_vmf_half_earths_full_reconensation.tex", "w") as f:
    f.write(global_element_vmf_half_earths_full_reconensation_df)
f.close()
global_element_loss_fraction_canonical_full_reconensation_df = pd.DataFrame(global_element_loss_fraction_canonical_full_reconensation).to_latex(index=False)
if "global_element_loss_fraction_canonical_full_reconensation.tex" in os.listdir():
    os.remove("global_element_loss_fraction_canonical_full_reconensation.tex")
with open("global_element_loss_fraction_canonical_full_reconensation.tex", "w") as f:
    f.write(global_element_loss_fraction_canonical_full_reconensation_df)
f.close()
global_element_loss_fraction_half_earth_full_reconensations_df = pd.DataFrame(global_element_loss_fraction_half_earth_full_reconensations).to_latex(index=False)
if "global_element_loss_fraction_half_earth_full_reconensations.tex" in os.listdir():
    os.remove("global_element_loss_fraction_half_earth_full_reconensations.tex")
with open("global_element_loss_fraction_half_earth_full_reconensations.tex", "w") as f:
    f.write(global_element_loss_fraction_half_earth_full_reconensations_df)
f.close()