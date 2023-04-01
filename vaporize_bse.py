from src.composition import Composition, ConvertComposition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data, make_figure

from math import log10
import re
import seaborn as sns
import pandas as pd
import string
from random import uniform
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import labellines

# use colorblind-friendly colors
plt.style.use('seaborn-colorblind')
# increase font size
plt.rcParams.update({"font.size": 12})

runs = [
    {
        "run_name": "Canonical Model",
        "temperature": 2682.61,  # K
        "vmf": 0.96,  # %
        "disk_theia_mass_fraction": 66.78,  # %
        "disk_mass": 1.02,  # lunar masses
        "vapor_loss_fraction": 74.0,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    },
    {
        "run_name": "Half-Earths Model",
        "temperature": 3517.83,  # K
        "vmf": 4.17,  # %
        "disk_theia_mass_fraction": 51.97,  # %
        "disk_mass": 1.70,  # lunar masses
        "vapor_loss_fraction": 16.0,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    }
]

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
bulk_moon_composition = {  # O'Neill 1991
    "SiO2": 44.37,
    'MgO': 34.90,
    'Al2O3': 3.90,
    'TiO2': 0.02,
    'Fe2O3': 0.0,
    'FeO': 13.54,
    'CaO': 3.27,
    'Na2O': 3.55e-3,
    'K2O': 3.78e-4,
    'ZnO': 2.39e-5,
}

major_gas_species = [
    "SiO", "O2", "MgO", "Fe", "Ca", "Al", "Ti", "Na", "K", "Zn"
]

for run in runs:
    # assign dictionary values to variables
    run_name = run["run_name"]
    temperature = run["temperature"]
    vmf = run["vmf"]
    disk_theia_mass_fraction = run["disk_theia_mass_fraction"]
    disk_mass = run["disk_mass"]
    vapor_loss_fraction = run["vapor_loss_fraction"]
    new_simulation = run["new_simulation"]

    if new_simulation:
        c = Composition(
            composition=bse_composition
        )

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

        reports = Report(composition=c, liquid_system=l, gas_system=g, thermosystem=t, to_dir=run_name)

        count = 1
        while t.weight_fraction_vaporized < 0.9:
            l.calculate_activities(temperature=temperature)
            g.calculate_pressures(temperature=temperature, liquid_system=l)
            if l.counter == 1:
                l.calculate_activities(temperature=temperature)
                g.calculate_pressures(temperature=temperature, liquid_system=l)
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
        )(vmf_val / 100.0)
    return interpolated_composition

def get_all_data_for_runs():
    data = {}
    for r in runs:
        run = r["run_name"]
        data[run] = r
        # get the data
        melt_oxide_mass_fraction = collect_data(path=f"{run}/magma_oxide_mass_fraction",
                                                x_header='mass fraction vaporized')
        magma_element_mass = collect_data(path=f"{run}/magma_element_mass",
                                          x_header='mass fraction vaporized')
        vapor_species_mass = collect_data(path=f"{run}/total_vapor_species_mass",
                                                   x_header='mass fraction vaporized')
        vapor_element_mass = collect_data(path=f"{run}/total_vapor_element_mass",
                                          x_header='mass fraction vaporized')
        vapor_species_mass_fraction = collect_data(path=f"{run}/total_vapor_species_mass_fraction",
                                                    x_header='mass fraction vaporized')
        vapor_element_mass_fraction = collect_data(path=f"{run}/total_vapor_element_mass_fraction",
                                                    x_header='mass fraction vaporized')
        melt_oxide_mass_fraction_at_vmf = get_composition_at_vmf(
            d=melt_oxide_mass_fraction,
            vmf_val=r["vmf"]
        )
        magma_element_mass_at_vmf = get_composition_at_vmf(
            d=magma_element_mass,
            vmf_val=r["vmf"]
        )
        vapor_species_mass_at_vmf = get_composition_at_vmf(
            d=vapor_species_mass,
            vmf_val=r["vmf"]
        )
        vapor_element_mass_at_vmf = get_composition_at_vmf(
            d=vapor_element_mass,
            vmf_val=r["vmf"]
        )
        vapor_species_mass_fraction_at_vmf = get_composition_at_vmf(
            d=vapor_species_mass_fraction,
            vmf_val=r["vmf"]
        )
        vapor_element_mass_fraction_at_vmf = get_composition_at_vmf(
            d=vapor_element_mass_fraction,
            vmf_val=r["vmf"]
        )


        # add each data set to the dictionary
        data[run]["melt_oxide_mass_fraction"] = melt_oxide_mass_fraction
        data[run]["magma_element_mass"] = magma_element_mass
        data[run]["vapor_species_mass"] = vapor_species_mass
        data[run]["vapor_element_mass"] = vapor_element_mass
        data[run]["vapor_species_mass_fraction"] = vapor_species_mass_fraction
        data[run]["vapor_element_mass_fraction"] = vapor_element_mass_fraction
        data[run]["melt_oxide_mass_fraction_at_vmf"] = melt_oxide_mass_fraction_at_vmf
        data[run]["magma_element_mass_at_vmf"] = magma_element_mass_at_vmf
        data[run]["vapor_species_mass_at_vmf"] = vapor_species_mass_at_vmf
        data[run]["vapor_element_mass_at_vmf"] = vapor_element_mass_at_vmf
        data[run]["vapor_species_mass_fraction_at_vmf"] = vapor_species_mass_fraction_at_vmf
        data[run]["vapor_element_mass_fraction_at_vmf"] = vapor_element_mass_fraction_at_vmf
    return data


def format_species_string(species):
    """
    Splits by _ and converts all numbers to subscripts.
    :param species:
    :return:
    """
    formatted = species.split("_")[0]
    return rf"$\rm {formatted.replace('2', '_{2}').replace('3', '_{3}')}$"
    # sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    # species = species.split("_")[0]
    # species = species.translate(sub)
    # return "".join(species)


# get run data
data = get_all_data_for_runs()

# ========================= MELT/VAPOR SPECIES MASS FRACTION =========================

# make a plot tracking liquid and vapor composition as a function of VMF
# first, create a 2x2 grid of plots
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.flatten()
# set a shared x and y axis label
fig.supxlabel("VMF (%)", fontsize=20)
fig.supylabel("Mass Fraction (%)", fontsize=20)
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(alpha=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1e-1, right=1e2)
    ax.set_ylim(bottom=1e-2, top=1e2)
# in the upper row, plot the liquid composition for each run
to_plot = 0
for run in data.keys():
    magma_plot_index = to_plot
    vapor_plot_index = to_plot + 1
    axs[magma_plot_index].set_title(f"{run} - Liquid Composition")
    axs[vapor_plot_index].set_title(f"{run} - Vapor Composition")
    for ax in [axs[magma_plot_index], axs[vapor_plot_index]]:
        # set a vertical line at the VMF
        ax.axvline(x=data[run]["vmf"], color="black", linestyle="--", alpha=1)
    melt = data[run]["melt_oxide_mass_fraction"]
    vapor = data[run]["vapor_species_mass_fraction"]
    # get the first item in the dictionary to get the species
    magma_species = list(melt[list(melt.keys())[0]].keys())
    vapor_species = list(vapor[list(vapor.keys())[0]].keys())
    # get a unique color for each oxide
    melt_colors = plt.cm.jet(np.linspace(0, 1, len(magma_species)))
    vapor_colors = plt.cm.jet(np.linspace(0, 1, len(vapor_species)))
    for index, species in enumerate(magma_species):
        axs[magma_plot_index].plot(
            np.array(list(melt.keys())) * 100,
            np.array([melt[i][species] for i in melt.keys()]) * 100,
            linewidth=2,
            color=melt_colors[index],
            label=format_species_string(species),
        )
    for index, species in enumerate(vapor_species):
        axs[vapor_plot_index].plot(
            np.array(list(vapor.keys())) * 100,
            np.array([vapor[i][species] for i in vapor.keys()]) * 100,
            linewidth=2,
            color=vapor_colors[index],
            label=format_species_string(species),
        )
    for ax in [axs[magma_plot_index], axs[vapor_plot_index]]:
        # labellines.labelLines(ax.get_lines(), zorder=2.5, align=True, fontsize=12)
        labellines.labelLines(ax.get_lines(), zorder=2.5, align=True,
                              xvals=[uniform(1e-2, 15) for i in ax.get_lines()], fontsize=12)
    to_plot += 2
# label the subplots
letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs):
    # label each subplot with a letter in the upper-left corner
    ax.annotate(
        letters[index], xy=(0.90, 0.98), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontweight="bold", fontsize=20
    )
fig.tight_layout()
plt.savefig("melt_vapor_species_mass_fraction.png", format='png', dpi=300)
plt.show()


# ========================= MELT SPIDER PLOT VS. BULK MOON =========================
pct_50_cond_temps = pd.read_csv("data/50_pct_condensation_temperatures.csv", index_col="Element")
lunar_bulk_compositions = pd.read_csv("data/lunar_bulk_compositions.csv", index_col="Oxide")
volatility_scale = pd.read_csv("data/volatility_scale.csv", index_col="Category")
normalized_bulk_compositions = {}  # normalize the bulk compositions to the preferred model
for oxide in lunar_bulk_compositions.index:
    for model in lunar_bulk_compositions.columns:
        if oxide not in normalized_bulk_compositions.keys():
            normalized_bulk_compositions[oxide] = {}
        if lunar_bulk_compositions[model][oxide] > 0:
            normalized_bulk_compositions[oxide][model] = lunar_bulk_compositions[model][oxide] / bulk_moon_composition[oxide]
# get the min and max values for each oxide across all models
min_values = {}
max_values = {}
for oxide in lunar_bulk_compositions.index:
    min_values[oxide] = min([normalized_bulk_compositions[oxide][model] for model in lunar_bulk_compositions.columns if
                             model in normalized_bulk_compositions[oxide].keys()])
    max_values[oxide] = max([normalized_bulk_compositions[oxide][model] for model in lunar_bulk_compositions.columns if
                             model in normalized_bulk_compositions[oxide].keys()])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
magma_species = []
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
for index, run in enumerate(runs):
    run_name = run["run_name"]
    melt = data[run_name]["melt_oxide_mass_fraction_at_vmf"]
    # get the first item in the dictionary to get the species
    magma_species = list(melt.keys())
    # get a list of the cations of each species, by first splitting by any numbers, then splitting by any capital letters
    cations = [re.split(r"\d+", species)[0] for species in magma_species]
    # split by capital letters
    cations = [re.split('(?<=.)(?=[A-Z])', cation)[0] for cation in cations]
    # order cations by their 50% condensation temperature
    cations = list(reversed(sorted(cations, key=lambda x: pct_50_cond_temps["50% Temperature"][x])))
    # sort the magma species by their cation
    magma_species2 = []
    for i in cations:
        for j in magma_species:
            if i in j:
                magma_species2.append(j)
    magma_species = magma_species2
    # plot arrows at the bottom of the plot to indicate the range of volatility
    ax.arrow(
        0, 10 ** -0.9, 3, 0, head_width=0.02, head_length=0.1, fc='k', ec='k', zorder=10, length_includes_head=True
    )
    ax.arrow(
        3, 10 ** -0.9, -3, 0, head_width=0.02, head_length=0.1, fc='k', ec='k', zorder=10, length_includes_head=True
    )
    # annotate in the center above the arrows
    ax.annotate(
        "Refractory", xy=(3 / 2, 10 ** -0.8), xycoords="data", horizontalalignment="center", verticalalignment="center",
        fontsize=14, fontweight="bold"
    )
    ax.arrow(
        3, 10 ** -0.9, 2, 0, head_width=0.02, head_length=0.1, fc='k', ec='k', zorder=10, length_includes_head=True
    )
    ax.arrow(
        5, 10 ** -0.9, -2, 0, head_width=0.02, head_length=0.1, fc='k', ec='k', zorder=10, length_includes_head=True
    )
    # annotate in the center above the arrows
    ax.annotate(
        "Transitional", xy=((5 - 2 / 2), 10 ** -0.8), xycoords="data", horizontalalignment="center", verticalalignment="center",
        fontsize=14, fontweight="bold"
    )
    ax.arrow(
        5, 10 ** -0.9, 3, 0, head_width=0.02, head_length=0.1, fc='k', ec='k', zorder=10, length_includes_head=True
    )
    ax.arrow(
        8, 10 ** -0.9, -3, 0, head_width=0.02, head_length=0.1, fc='k', ec='k', zorder=10, length_includes_head=True
    )
    # annotate in the center above the arrows
    ax.annotate(
        "Moderately Volatile", xy=((8 - 3 / 2)
                                   , 10 ** -0.8), xycoords="data", horizontalalignment="center", verticalalignment="center",
        fontsize=14, fontweight="bold"
    )
    # get a unique color for each oxide
    ax.plot(
        magma_species,
        np.array([melt[species] for species in magma_species]) * 100 / [bulk_moon_composition[species]
                                                                        for species in magma_species],
        linewidth=4,
        color=color_cycle[index],
        label=run['run_name']
    )
    # scatter the melt abundance on top of the line
    ax.scatter(
        magma_species,
        np.array([melt[species] for species in magma_species]) * 100 / [bulk_moon_composition[species]
                                                                        for species in magma_species],
        color=color_cycle[index],
        s=100,
        zorder=10
    )
# shade the range of normalized lunar bulk compositions
ax.fill_between(
    magma_species, [min_values[i] for i in magma_species],
    [max_values[i] for i in magma_species], color='grey', alpha=0.5
)
# plot the BSE composition
ax.plot(
    magma_species,
    [bse_composition[species] / bulk_moon_composition[species] for species in magma_species],
    linewidth=4,
    linestyle="--",
    color=color_cycle[len(runs)],
    label="BSE"
)
# scatter the BSE melt abundance on top of the line
ax.scatter(
    magma_species,
    [bse_composition[species] / bulk_moon_composition[species] for species in magma_species],
    color=color_cycle[len(runs)],
    s=100,
    zorder=10
)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xticklabels([format_species_string(species) for species in magma_species], rotation=45)
ax.axhline(y=1, color="black", linewidth=4, alpha=1, label="Bulk Moon")
ax.set_ylabel("Melt Species Mass (Relative to Bulk Moon)", fontsize=20)
ax.grid()
ax.set_ylim(bottom=10 ** -1, top=10 ** 2.5)
ax.set_yscale("log")
ax.legend(fontsize=18)
plt.tight_layout()
plt.savefig("melt_spider_plot.png", format='png', dpi=300)
plt.show()


# ========================= VAPOR ELEMENTS AS A FUNCTION OF VOLATILITY =========================
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
# generate a list of 3 different markers
markers = ["o", "s", "D"]
marker_size = 200
colors = list(sns.color_palette("colorblind", len(pct_50_cond_temps.index)))
colors_2 = list(sns.color_palette("colorblind", len(volatility_scale.index)))
# plot the volatility scale by shading between the upper and lower bounds
for index, c in enumerate(volatility_scale.index):
    ax.axvspan(
        volatility_scale.loc[c, "Lower Bound"],
        volatility_scale.loc[c, "Upper Bound"],
        alpha=0.5,
        color=colors_2[index],
        zorder=5,
        label=c
    )
# plot the 50% condensation temperatures
for index, run in enumerate(runs):
    run_name = run["run_name"]
    marker = markers[index]
    vapor_element_mass_frac = data[run_name]["vapor_element_mass_fraction_at_vmf"]
    elements = list(vapor_element_mass_frac.keys())
    abundances = np.array([vapor_element_mass_frac[element] for element in elements])
    # get the 50% condensation temperature for each element
    cond_temps = [pct_50_cond_temps.loc[element, "50% Temperature"] for element in elements]
    # plot the vapor element mass fraction as a function of the 50% condensation temperature
    ax.scatter(
        cond_temps,
        abundances * 100,
        color='black',
        s=marker_size,
        marker=marker,
        zorder=10,
        alpha=1
    )
    run_name = run["run_name"]
    ax.scatter(
        [],
        [],
        color='black',
        s=marker_size,
        marker=marker,
        label=run_name,
    )
    # annotate the name of the element at the point
    for index, element in enumerate(elements):
        ax.annotate(
            format_species_string(element),
            xy=(cond_temps[index] + 70, abundances[index] * 100),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=18,
            fontweight="bold",
        )

ax.set_xlabel("50% Condensation Temperature (K)", fontsize=20)
ax.set_ylabel("Vapor Element Mass Fraction (%)", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_yscale("log")
ax.grid()
ax.set_ylim(bottom=10 ** -2, top=10 ** 2)
ax.set_xlim(0, 1800)
ax.legend(fontsize=18)
plt.tight_layout()
plt.show()
plt.savefig("vapor_elements_vs_volatility.png", format='png', dpi=300)


# ========================= VAPOR ELEMENTS AS A FUNCTION OF VOLATILITY (SPIDER) =========================
# do the same as above but with bars
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)



# # collect data
# melt_oxide_mass_fraction = collect_data(path=f"{run_name}/magma_oxide_mass_fraction",
#                                         x_header='mass fraction vaporized')
# magma_element_mass = collect_data(path=f"{run_name}/magma_element_mass",
#                                   x_header='mass fraction vaporized')
# vapor_species_mass_fraction = collect_data(path=f"{run_name}/total_vapor_species_mass",
#                                            x_header='mass fraction vaporized')
# vapor_element_mass = collect_data(path=f"{run_name}/total_vapor_element_mass",
#                                   x_header='mass fraction vaporized')
#
# # collect some metadata about the simulation
# vmfs = list(melt_oxide_mass_fraction.keys())
# oxides = [i for i in bse_composition.keys() if i != "Fe2O3"]
# vapor_species = [i for i in vapor_species_mass_fraction[vmfs[0]].keys()]
# elements = [i for i in vapor_element_mass[vmfs[0]].keys()]
#
#
# # plot the melt and vapor composition as a function of mass fraction vaporized
# # make a figure with 2 columns and 1 row
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# # plot the liquid composition in the first column
# for oxide in oxides:
#     ax[0].plot(
#         list(melt_oxide_mass_fraction.keys()),
#         [i[oxide] for i in melt_oxide_mass_fraction.values()],
#         label=oxide
#     )
# ax[0].set_xlabel("Mass Fraction Vaporized (%)")
#
# def iterpolate_at_vmf(vmf_val):
#     """
#     Interpolate the melt and vapor composition at the specified vmf.
#     :param vmf_val:
#     :return:
#     """
#     # interpolate the melt and vapor composition at the specified vmf
#     magma_oxide_mass_fraction_at_vmf = {}
#     magma_element_mass_at_vmf = {}
#     vapor_species_mass_fraction_at_vmf = {}
#     vapor_element_mass_at_vmf = {}
#
#     # interpolate each at the given vmf
#     for oxide in oxides:
#         magma_oxide_mass_fraction_at_vmf[oxide] = interp1d(
#             list(melt_oxide_mass_fraction.keys()),
#             [i[oxide] for i in melt_oxide_mass_fraction.values()]
#         )(vmf_val / 100.0)
#     for element in elements:
#         magma_element_mass_at_vmf[element] = interp1d(
#             list(magma_element_mass.keys()),
#             [i[element] for i in magma_element_mass.values()]
#         )(vmf_val / 100.0)
#         vapor_element_mass_at_vmf[element] = interp1d(
#             list(vapor_element_mass.keys()),
#             [i[element] for i in vapor_element_mass.values()]
#         )(vmf_val / 100.0)
#     for species in vapor_species:
#         vapor_species_mass_fraction_at_vmf[species] = interp1d(
#             list(vapor_species_mass_fraction.keys()),
#             [i[species] for i in vapor_species_mass_fraction.values()]
#         )(vmf_val / 100.0)
#
#     return magma_oxide_mass_fraction_at_vmf, magma_element_mass_at_vmf, \
#             vapor_species_mass_fraction_at_vmf, vapor_element_mass_at_vmf
#
# def recondense_vapor(vapor_element_mass_at_vmf_dict: dict, magma_element_mass_at_vmf_dict: dict,
#                      vapor_loss_fraction_val: float):
#     """
#     Recondense the vapor at the specified vapor loss fraction.
#     :param vapor_element_mass_at_vmf_dict:
#     :param magma_element_mass_at_vmf_dict:
#     :param vapor_loss_fraction_val:
#     :return:
#     """
#     # calculate the elemental mass of the vapor after hydrodynamic vapor loss
#     vapor_element_mass_after_hydrodynamic_vapor_loss = {}
#     for element in elements:
#         vapor_element_mass_after_hydrodynamic_vapor_loss[element] = vapor_element_mass_at_vmf_dict[element] * \
#                                                                     (1 - (vapor_loss_fraction_val / 100.0))
#     # add this back to the magma assuming full elemental recondensation
#     magma_element_mass_after_recondensation = {}
#     for element in elements:
#         magma_element_mass_after_recondensation[element] = magma_element_mass_at_vmf_dict[element] + \
#                                                            vapor_element_mass_after_hydrodynamic_vapor_loss[element]
#     # renormalize the magma elemental mass after recondensation to oxide weight percent
#     magma_oxide_mass_fraction_after_recondensation = ConvertComposition().cations_mass_to_oxides_weight_percent(
#         magma_element_mass_after_recondensation, oxides=oxides
#     )
#     return magma_oxide_mass_fraction_after_recondensation
#
#
# magma_oxide_mass_fraction_at_vmf, magma_element_mass_at_vmf, \
#             vapor_species_mass_fraction_at_vmf, vapor_element_mass_at_vmf = iterpolate_at_vmf(vmf_val=vmf)
# magma_oxide_mass_fraction_after_recondensation = recondense_vapor(
#     vapor_element_mass_at_vmf_dict=vapor_element_mass_at_vmf,
#     magma_element_mass_at_vmf_dict=magma_element_mass_at_vmf,
#     vapor_loss_fraction_val=vapor_loss_fraction
# )
#
# # plot the bulk oxide composition of the BSE as it vaporizes
# # and compare it to the bulk oxide composition of the Moon
# fig = plt.figure(figsize=(16, 9))
# ax = fig.add_subplot(111)
# # get a unique color for each oxide
# colors = plt.cm.jet(np.linspace(0, 1, len(oxides)))
# for index, oxide in enumerate(oxides):
#     ax.plot(
#         np.array(list(melt_oxide_mass_fraction.keys())) * 100,
#         np.array([i[oxide] for i in melt_oxide_mass_fraction.values()]) * 100,
#         color=colors[index],
#         linewidth=2,
#         label=oxide
#     )
# ax.axvline(
#     x=vmf,
#     color="black",
#     linestyle="--",
#     linewidth=2,
#     # label=f"VMF: {round(vmf, 2)}%"
# )
# # scatter the interpolated bulk oxide composition of the BSE at the given vmf
# for oxide in oxides:
#     ax.scatter(
#         vmf,
#         magma_oxide_mass_fraction_at_vmf[oxide] * 100,
#         color=colors[oxides.index(oxide)],
#         s=100,
#         zorder=10
#     )
# ax.set_xlabel("Magma Fraction Vaporized (%)", fontsize=16)
# ax.set_ylabel("Bulk Oxide Composition (wt. %)", fontsize=16)
# ax.set_title(
#     "Bulk Oxide Composition of the BSE as it Vaporizes",
# )
# ax.grid()
# ax.set_yscale("log")
# # set a lower y limit of 10 ** -4
# ax.set_ylim(bottom=10 ** -4)
# # add a legend to the right of the plot
# ax.legend(
#     loc="center left",
#     bbox_to_anchor=(1, 0.5),
#     fontsize=16,
#     frameon=False
# )
# plt.show()
#
# # make a spider plot showing the composition of the recondensed BSE melt relative to the Moon
# fig = plt.figure(figsize=(16, 9))
# ax = fig.add_subplot(111)
# # get a unique color for each oxide
# ax.plot(
#     oxides,
#     np.array([magma_oxide_mass_fraction_after_recondensation[oxide] / bulk_moon_composition[oxide] for oxide in oxides]),
#     color='black',
#     linewidth=2,
# )
# # label the x-axis with the oxide names
# # ax.set_xticklabels(oxides)
# ax.set_title(
#     "Recondensed BSE Melt Composition Relative to the Moon",
#     fontsize=16
# )
# ax.axhline(
#     y=1,
#     color="red",
#     linestyle="--",
#     linewidth=2,
#     label="1:1 with Bulk Moon"
# )
# # annotate the VMF, the temperature, and the vapor loss fraction
# ax.annotate(
#     f"VMF: {round(vmf, 2)}%\nVapor Loss Fraction: {round(vapor_loss_fraction, 2)}%\nTemperature: {round(temperature, 2)} K",
#     xy=(0.5, 0.90),
#     xycoords="axes fraction",
#     fontsize=16,
#     horizontalalignment="center",
#     verticalalignment="center"
# )
# ax.set_yscale("log")
# ax.grid()
# ax.legend()
# plt.show()
#
# # make the same figure, but do it over a range of vmfs
# fig = plt.figure(figsize=(16, 9))
# ax = fig.add_subplot(111)
#
# for vmf_val in np.linspace(0.1, 85, 15):
#     magma_oxide_mass_fraction_at_vmf, magma_element_mass_at_vmf, \
#         vapor_species_mass_fraction_at_vmf, vapor_element_mass_at_vmf = iterpolate_at_vmf(vmf_val=vmf_val)
#     magma_oxide_mass_fraction_after_recondensation = recondense_vapor(
#         vapor_element_mass_at_vmf_dict=vapor_element_mass_at_vmf,
#         magma_element_mass_at_vmf_dict=magma_element_mass_at_vmf,
#         vapor_loss_fraction_val=vapor_loss_fraction
#     )
#     ax.plot(
#         oxides,
#         np.array([magma_oxide_mass_fraction_after_recondensation[oxide] / bulk_moon_composition[oxide] for oxide in oxides]),
#         linewidth=2,
#         label=f"VMF: {round(vmf_val, 2)}%"
#     )
# # label the x-axis with the oxide names
# ax.set_title(
#     "Recondensed BSE Melt Composition Relative to the Moon",
#     fontsize=16
# )
# ax.axhline(
#     y=1,
#     color="red",
#     linestyle="--",
#     linewidth=2,
#     label="1:1 with Bulk Moon"
# )
# # annotate the VMF, the temperature, and the vapor loss fraction
# ax.annotate(
#     f"VMF: {round(vmf, 2)}%\nVapor Loss Fraction: {round(vapor_loss_fraction, 2)}%\nTemperature: {round(temperature, 2)} K",
#     xy=(0.5, 0.90),
#     xycoords="axes fraction",
#     fontsize=16,
#     horizontalalignment="center",
#     verticalalignment="center"
# )
# ax.set_yscale("log")
# ax.grid()
# ax.legend()
# plt.show()