from src.composition import Composition, ConvertComposition, normalize
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data, make_figure

from isotopes.rayleigh import FullSequenceRayleighDistillation_SingleReservior

import os
from math import log10
import re
import seaborn as sns
import pandas as pd
import string
from random import uniform
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
        "new_simulation": True,  # True to run a new simulation, False to load a previous simulation
    },
    {
        "run_name": "Half-Earths Model",
        "temperature": 3517.83,  # K
        "vmf": 4.17,  # %
        "disk_theia_mass_fraction": 51.97,  # %
        "disk_mass": 1.70,  # lunar masses
        "vapor_loss_fraction": 16.0,  # %
        "new_simulation": True,  # True to run a new simulation, False to load a previous simulation
    }
]

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
bulk_moon_composition = normalize({  # O'Neill 1991
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
})

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


def write_mass_distribution_file(melt_mass_at_vmf, bulk_vapor_mass_at_vmf, run_name,
                                 escaping_vapor_mass_at_vmf, retained_vapor_mass_at_vmf):
    if os.path.exists(f"{run_name}_mass_distribution.csv"):
        os.remove(f"{run_name}_mass_distribution.csv")
    with open(f"{run_name}_mass_distribution.csv", "w") as f:
        header = "component," + ",".join([str(i) for i in melt_mass_at_vmf.keys()]) + "\n"
        f.write(header)
        f.write("melt mass," + ",".join([str(i) for i in melt_mass_at_vmf.values()]) + "\n")
        f.write("bulk vapor mass," + ",".join([str(i) for i in bulk_vapor_mass_at_vmf.values()]) + "\n")
        f.write("bulk system mass," + ",".join([str(i) for i in (np.array(list(melt_mass_at_vmf.values())) + np.array(
            list(bulk_vapor_mass_at_vmf.values()))).tolist()]) + "\n")
        f.write("escaping vapor mass," + ",".join([str(i) for i in escaping_vapor_mass_at_vmf.values()]) + "\n")
        f.write("retained vapor mass," + ",".join([str(i) for i in retained_vapor_mass_at_vmf.values()]) + "\n")
        f.write("recondensed melt mass," + ",".join([str(i) for i in (np.array(list(melt_mass_at_vmf.values())) + np.array(
            list(retained_vapor_mass_at_vmf.values()))).tolist()]) + "\n")
    print(f"wrote file {run_name}_mass_distribution.csv")
    f.close()


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


def recondense_vapor(melt_absolute_cation_masses: dict, vapor_absolute_cation_mass: dict, vapor_loss_fraction: float):
    """
    Recondenses retained vapor into the melt.
    :param vapor_absolute_mass:
    :param vapor_loss_fraction:
    :return:
    """
    lost_vapor_mass = {
        k: v * (vapor_loss_fraction / 100) for k, v in vapor_absolute_cation_mass.items()
    }
    retained_vapor_mass = {
        k: v - lost_vapor_mass[k] for k, v in vapor_absolute_cation_mass.items()
    }
    recondensed_melt_mass = {
        k: v + retained_vapor_mass[k] for k, v in melt_absolute_cation_masses.items()
    }
    # convert to oxide mass fractions
    c = ConvertComposition().cations_mass_to_oxides_weight_percent(
        cations=recondensed_melt_mass, oxides=list(bse_composition.keys())
    )
    # divide by 100 to get mass fraction
    return {
        "recondensed_melt_oxide_mass_fraction": {k: v / 100 for k, v in c.items()},
        "lost_vapor_mass": lost_vapor_mass,
        "retained_vapor_mass": retained_vapor_mass,
        "recondensed_melt_mass": recondensed_melt_mass
    }


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
        vapor_element_mole_fraction = collect_data(path=f"{run}/atmosphere_total_mole_fraction",
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

        recondensed = recondense_vapor(
            melt_absolute_cation_masses=magma_element_mass_at_vmf,
            vapor_absolute_cation_mass=vapor_element_mass_at_vmf,
            vapor_loss_fraction=r["vapor_loss_fraction"]
        )

        recondensed_melt_oxide_mass_fraction = recondensed["recondensed_melt_oxide_mass_fraction"]
        escaping_vapor_mass = recondensed["lost_vapor_mass"]
        retained_vapor_mass = recondensed["retained_vapor_mass"]
        recondensed_melt_mass = recondensed["recondensed_melt_mass"]

        vapor_element_mole_fraction_at_vmf = get_composition_at_vmf(
            d=vapor_element_mole_fraction,
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
        data[run]["recondensed_melt_oxide_mass_fraction"] = recondensed_melt_oxide_mass_fraction
        data[run]["vapor_element_mole_fraction"] = vapor_element_mole_fraction
        data[run]["vapor_element_mole_fraction_at_vmf"] = vapor_element_mole_fraction_at_vmf

        # write the mass distribution file
        write_mass_distribution_file(
            melt_mass_at_vmf=magma_element_mass_at_vmf, bulk_vapor_mass_at_vmf=vapor_element_mass_at_vmf,
            run_name=run,
            escaping_vapor_mass_at_vmf=escaping_vapor_mass, retained_vapor_mass_at_vmf=retained_vapor_mass
        )

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
# reformat all oxide wt% compositions to a pandas dataframe so it can be output to a csv easily
df = pd.DataFrame({"oxide": [i for i in list(bse_composition.keys()) if i != "Fe2O3"]})
for run in data.keys():
    for key in ["melt_oxide_mass_fraction_at_vmf", "recondensed_melt_oxide_mass_fraction"]:
        df[f"{run}_{key}"] = np.array([data[run][key][oxide] for oxide in
                                       bse_composition.keys() if oxide != "Fe2O3"]) * 100
df.to_csv("vaporize_bse.csv", index=False)

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
    # # set the bottom and upper limts of the x axis to match the min and max VMF values
    # ax.set_xlim(
    #     left=min(next(iter(data.values()))["melt_oxide_mass_fraction"].keys()) * 100,
    #     right=max(next(iter(data.values()))["melt_oxide_mass_fraction"].keys()) * 100
    # )
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
                              xvals=[uniform(1e-2, 2) for i in ax.get_lines()], fontsize=12)
    to_plot += 2
# label the subplots
letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs):
    # label each subplot with a letter in the upper-left corner
    ax.annotate(
        letters[index], xy=(0.90, 0.98), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
        fontweight="bold", fontsize=20
    )
plt.tight_layout()
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
            normalized_bulk_compositions[oxide][model] = lunar_bulk_compositions[model][oxide] / bulk_moon_composition[
                oxide]
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
    recondensed_melt = data[run_name]["recondensed_melt_oxide_mass_fraction"]
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
        fontsize=14, fontweight="bold", backgroundcolor="w"
    )
    ax.arrow(
        3, 10 ** -0.9, 2, 0, head_width=0.02, head_length=0.1, fc='k', ec='k', zorder=10, length_includes_head=True
    )
    ax.arrow(
        5, 10 ** -0.9, -2, 0, head_width=0.02, head_length=0.1, fc='k', ec='k', zorder=10, length_includes_head=True
    )
    # annotate in the center above the arrows
    ax.annotate(
        "Transitional", xy=((5 - 2 / 2), 10 ** -0.8), xycoords="data", horizontalalignment="center",
        verticalalignment="center",
        fontsize=14, fontweight="bold", backgroundcolor="w"
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
                                   , 10 ** -0.8), xycoords="data", horizontalalignment="center",
        verticalalignment="center",
        fontsize=14, fontweight="bold", backgroundcolor="w"
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
    # plot the same for the recondensed melt
    ax.plot(
        magma_species,
        np.array([recondensed_melt[species] for species in magma_species]) * 100 / [bulk_moon_composition[species]
                                                                                    for species in magma_species],
        linewidth=4,
        color=color_cycle[index],
        linestyle="--",
        # label=run['run_name']
    )
    # scatter the melt abundance on top of the line
    ax.scatter(
        magma_species,
        np.array([recondensed_melt[species] for species in magma_species]) * 100 / [bulk_moon_composition[species]
                                                                                    for species in magma_species],
        color=color_cycle[index],
        s=100,
        marker='d',
        zorder=10
    )
ax.plot(
    [], [], linewidth=4, linestyle="--", color='black', label="Including Retained\nVapor Recondensation"
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
    # linestyle="--",
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

# ====================================== MVE ISOTOPE FRACTIONATION ======================================
# define constants
# define the observed isotope ratios
delta_K_Lunar_BSE = 0.415  # mean of 41K/39K isotope ratios from lunar samples
delta_K_Lunar_BSE_std_error = 0.05  # standard error of 41K/39K isotope ratios from lunar samples
delta_Zn_Lunar_BSE = 1.12  # mean of 66Zn/64Zn isotope ratios from lunar samples
delta_Zn_Lunar_BSE_std_error = 0.55  # standard error of 66Zn/64Zn isotope ratios from lunar samples
delta_K_BSE = -0.479  # mean of 41K/39K isotope ratios from BSE
delta_K_BSE_std_error = 0.027  # standard error of 41K/39K isotope ratios from BSE
delta_Zn_BSE = 0.28  # mean of 66Zn/64Zn isotope ratios from BSE
delta_Zn_BSE_std_error = 0.05  # standard error of 66Zn/64Zn isotope ratios from BSE

# define ranges for the isotope ratios
delta_k_theia_range = np.arange(-0.6, 0.6, 0.1)  # range of 41K/39K isotope ratios to test for Theia
# delta_k_theia_range = np.arange(-500, 500, 20)  # range of 41K/39K isotope ratios to test for Theia
# delta_Zn_theia_range = np.arange(-410, -380, 5)  # range of 66Zn/64Zn isotope ratios to test for Theia
delta_Zn_theia_range = np.arange(-900, 900, 20)  # range of 66Zn/64Zn isotope ratios to test for Theia

# make a subplot with 2 columns and 3 rows
fig, ax = plt.subplots(1, 1, figsize=(10, 10),  sharex='all', sharey='all')
# increase the font size
# axs = ax.flatten()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# generate a list of markers that is as long as the runs list
markers = ["o", "D"]
to_plot = 0
k_fractionation_data = {'run': []}
ax.tick_params(axis='both', which='major', labelsize=18)
ax.grid(alpha=0.5)
ebar_index = 1
for run_index, run in enumerate(runs):
    run_name = run['run_name']
    mass_distribution = pd.read_csv(f"{run_name}_mass_distribution.csv", index_col='component')
    k_isotopes = FullSequenceRayleighDistillation_SingleReservior(
        heavy_z=41, light_z=39, vapor_escape_fraction=run['vapor_loss_fraction'],
        system_element_mass=mass_distribution['K']['bulk system mass'],
        melt_element_mass=mass_distribution['K']['melt mass'],
        vapor_element_mass=mass_distribution['K']['bulk vapor mass'], earth_isotope_composition=delta_K_BSE,
        theia_ejecta_fraction=0,
        total_melt_mass=sum([mass_distribution[i]['melt mass'] for i in mass_distribution.keys() if len(i) < 3]),
        total_vapor_mass=sum(
            [mass_distribution[i]['bulk vapor mass'] for i in mass_distribution.keys() if len(i) < 3]),
    )
    # k_isotopes_starting_earth_isotope_composition = k_isotopes.run_3_stage_fractionation()  # assumes ejecta is fully Earth-like
    k_data = k_isotopes.fractionate(
        reservoir_delta=delta_K_BSE
    )
    k_lower_data = k_isotopes.fractionate(
        reservoir_delta=delta_K_BSE - delta_K_BSE_std_error
    )
    k_upper_data = k_isotopes.fractionate(
        reservoir_delta=delta_K_BSE + delta_K_BSE_std_error
    )  # assumes ejecta is a mix of Earth and Theia

    # do the same, but assume no physical kinetic fractionation
    k_isotopes_no_phys = FullSequenceRayleighDistillation_SingleReservior(
        heavy_z=41, light_z=39, vapor_escape_fraction=run['vapor_loss_fraction'],
        system_element_mass=mass_distribution['K']['bulk system mass'],
        melt_element_mass=mass_distribution['K']['melt mass'],
        vapor_element_mass=mass_distribution['K']['bulk vapor mass'], earth_isotope_composition=delta_K_BSE,
        theia_ejecta_fraction=0,
        total_melt_mass=sum([mass_distribution[i]['melt mass'] for i in mass_distribution.keys() if len(i) < 3]),
        total_vapor_mass=sum(
            [mass_distribution[i]['bulk vapor mass'] for i in mass_distribution.keys() if len(i) < 3]),
        alpha_phys=1
    )

    # k_isotopes_starting_earth_isotope_composition = k_isotopes.run_3_stage_fractionation()  # assumes ejecta is fully Earth-like
    k_data_no_phys = k_isotopes_no_phys.fractionate(
        reservoir_delta=delta_K_BSE
    )
    k_lower_data_no_phys = k_isotopes_no_phys.fractionate(
        reservoir_delta=delta_K_BSE - delta_K_BSE_std_error
    )
    k_upper_data_no_phys = k_isotopes_no_phys.fractionate(
        reservoir_delta=delta_K_BSE + delta_K_BSE_std_error
    )  # assumes ejecta is a mix of Earth and Theia

    label = None
    if run_index == 0:
        label = "Observed"
    ax.axvline(
        delta_K_Lunar_BSE,
        color='grey',
        linestyle='--',
        alpha=1,
        label=label,
    )
    ax.axvspan(
        delta_K_Lunar_BSE - delta_K_Lunar_BSE_std_error,
        delta_K_Lunar_BSE + delta_K_Lunar_BSE_std_error,
        alpha=0.2,
        color='grey',
        # label="Observed",
    )

    isotope_runs = [
        # [delta_K_Lunar_BSE, delta_K_Lunar_BSE_std_error, delta_K_Lunar_BSE_std_error, "Observed"],
        [k_data['delta_moon_earth_no_recondensation'], k_data['delta_moon_earth_no_recondensation'] - k_lower_data['delta_moon_earth_no_recondensation'], k_upper_data['delta_moon_earth_no_recondensation'] - k_data['delta_moon_earth_no_recondensation'], "No Recondensation"],
        [k_data['delta_moon_earth'], k_data['delta_moon_earth'] - k_lower_data['delta_moon_earth'], k_upper_data['delta_moon_earth'] - k_data['delta_moon_earth'], "With Recondensation"],

        # [k_data_no_phys['delta_moon_earth_no_recondensation'],
         # k_data_no_phys['delta_moon_earth_no_recondensation'] - k_lower_data_no_phys['delta_moon_earth_no_recondensation'],
         # k_upper_data_no_phys['delta_moon_earth_no_recondensation'] - k_data_no_phys['delta_moon_earth_no_recondensation'],
         # "No Phys., No Recondensation"],
        [k_data_no_phys['delta_moon_earth'], k_data_no_phys['delta_moon_earth'] - k_lower_data_no_phys['delta_moon_earth'],
         k_upper_data_no_phys['delta_moon_earth'] - k_data_no_phys['delta_moon_earth'], "No Physical Frac.\nWith Recondensation"],
    ]
    for index, dataset in enumerate(isotope_runs):
        label = None
        if to_plot == 0:
            label = dataset[3]
        ax.errorbar(
            dataset[0], ebar_index,
            xerr=[[dataset[1]], [dataset[2]]],
            fmt=markers[run_index],
            markersize=10,
            elinewidth=2,
            capsize=5,
            capthick=3,
            color=colors[index],
            # label=label,
        )
        ebar_index += 1

    for index, dataset in enumerate(isotope_runs):
        if run_index == 0:
            ax.scatter(
                [], [], marker='s', color=colors[index], s=100, label=dataset[3]
            )

    ax.scatter(
        [], [], marker=markers[run_index], color='k', s=100, label=runs[run_index]['run_name']
    )


    k_fractionation_data['run'].append(run_name)
    for key in k_data.keys():
        if key not in k_fractionation_data.keys():
            k_fractionation_data[key] = []
        k_fractionation_data[key].append(k_data[key])

    # to_plot += 1

# annotate a letter in the upper left corner
# letters = list(string.ascii_lowercase)
# for index, ax in enumerate(ax):
#     # label each subplot with a letter in the upper-left corner
#     ax.annotate(
#         letters[index], xy=(0.05, 0.95), xycoords="axes fraction", horizontalalignment="left", verticalalignment="top",
#         fontweight="bold", fontsize=20
#     )

for ax, t in [(ax, r"$\delta \rm ^{41}K_{Theia}$")]:
    # ax.set_xlabel(r"$\Delta_{\rm Lunar-BSE}$ " + f"({t})", fontsize=20)
    ax.set_xlabel(r"$\delta \rm ^{41}K_{Lunar-BSE}$", fontsize=20)
# turn of y-axis labels for all subplots
ax.set_yticklabels([])
ax.set_yticks([])
# ax.set_ylim(len(isotope_runs) / 2 , 0.5 + 2)

ax.set_title(r"$\rm ^{41/39}K$", fontsize=20)
ax.legend(loc='lower left', fontsize=18)
plt.tight_layout()
# fig.subplots_adjust(right=0.84)

pd.DataFrame(k_fractionation_data).to_csv("bse_k_isotope_fractionation.csv", index=False)
plt.savefig("bse_isotope_fractionation.png", dpi=300)
plt.show()




# ============================= Plot K/K0 and Na/Na0 as function of VMF =============================
# collect atmosphere mass fraction data
d_v = collect_data(path=f"{run_name}/total_vapor_element_mass", x_header='mass fraction vaporized')
d_l = collect_data(path=f"{run_name}/magma_element_mass", x_header='mass fraction vaporized')
d_t = {v: {element: d_v[v][element] + d_l[v][element] for element in d_v[v].keys()} for v in d_v.keys()}
d = {}
# bse_cation_wt_pct = Composition(composition=bse_composition).oxide_wt_pct_to_cation_wt_pct(composition=bse_composition)
# drop O from each sub-dictionary and re-normalize
for key in d_v.keys():
    d[key] = {element: mass / d_t[key][element] for element, mass in d_v[key].items()}

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
normalizer = Normalize(min(d.keys()) * 100, max(d.keys()) * 100)
cmap = cm.get_cmap('jet')
ax.scatter(
    [d[key]['K'] for key in d.keys()],
    [d[key]['Na'] for key in d.keys()],
    color=[cmap(normalizer(key * 100)) for key in d.keys()],
    s=100,
)
ax.set_xlabel(r"K/K$_0$", fontsize=20)
ax.set_ylabel(r"Na/Na$_0$", fontsize=20)
ax.set_title("Mass Ratios of K and Na relative to Liquid + Vapor (Total) Mass", fontsize=16)
ax.grid()

sm = cm.ScalarMappable(norm=normalizer, cmap=cmap)
sm.set_array([])
cbaxes = inset_axes(ax, width="30%", height="3%", loc='lower right', borderpad=1.8)
cbar = plt.colorbar(sm, cax=cbaxes, orientation='horizontal')
cbar.ax.tick_params(labelsize=12)
# cbar.ax.set_title("Entropy", fontsize=6)
cbar.ax.set_title("VMF (%)", fontsize=12)

# make both axes log scale
ax.set_xscale('log')
ax.set_yscale('log')

plt.show()
plt.savefig("bse_k_na_vapor_comp.png", dpi=300)


# ================== Plot the mass fraction of each element lost relative to initial ==================
for index, run in enumerate(runs):
    run_name = run['run_name']
    vapor_loss_fraction = run['vapor_loss_fraction']
    # read in the ejecta composition file
    mass_distribution = pd.read_csv(f"{run_name}_mass_distribution.csv", index_col='component')
    # get the loss fraction of each element
    loss_fraction = {element: mass_distribution.loc[element, 'escaping vapor mass'] / (mass_distribution.loc[element, 'melt mass'] + mass_distribution.loc[element, 'bulk vapor mass']) * 100.0 for element in elements}
    # sort cations by 50% condensation temperature
    cations = list(reversed(sorted(list(loss_fraction.keys()), key=lambda x: pct_50_cond_temps["50% Temperature"][x])))
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
        fontsize=14, fontweight="bold", backgroundcolor="w"
    )
    ax.arrow(
        3, 10 ** -0.9, 2, 0, head_width=0.02, head_length=0.1, fc='k', ec='k', zorder=10, length_includes_head=True
    )
    ax.arrow(
        5, 10 ** -0.9, -2, 0, head_width=0.02, head_length=0.1, fc='k', ec='k', zorder=10, length_includes_head=True
    )
    # annotate in the center above the arrows
    ax.annotate(
        "Transitional", xy=((5 - 2 / 2), 10 ** -0.8), xycoords="data", horizontalalignment="center",
        verticalalignment="center",
        fontsize=14, fontweight="bold", backgroundcolor="w"
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
                                   , 10 ** -0.8), xycoords="data", horizontalalignment="center",
        verticalalignment="center",
        fontsize=14, fontweight="bold", backgroundcolor="w"
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
    # scatter the loss fraction on top of the line
    ax.scatter(
        magma_species,
        np.array([melt[species] for species in magma_species]) * 100 / [bulk_moon_composition[species]
                                                                        for species in magma_species],
        color=color_cycle[index],
        s=100,
        zorder=10
    )
    # scatter the melt abundance on top of the line
    ax.scatter(
        magma_species,
        np.array([recondensed_melt[species] for species in magma_species]) * 100 / [bulk_moon_composition[species]
                                                                                    for species in magma_species],
        color=color_cycle[index],
        s=100,
        marker='d',
        zorder=10
    )
ax.plot(
    [], [], linewidth=4, linestyle="--", color='black', label="Including Retained\nVapor Recondensation"
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









#
# # ========================= PLOT ELEMENTAL VAPOR MASS FRACTION AS FUNCTION OF VMF =========================
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# ax.set_xlabel("Element", fontsize=20)
# ax.set_ylabel("Vapor Element Mass Fraction (%)", fontsize=20)
# linestyles = ["-", "--", "-."]
# for index, run in enumerate(runs):
#     vapor_masses = data[run["run_name"]]["vapor_element_mass_at_vmf"]
#     total_vapor_mass = sum(vapor_masses.values())
#     elements = list(vapor_masses.keys())
#     color_cycle = list(sns.color_palette("colorblind", len(elements)))
#     ax.plot(
#         elements,
#         [vapor_masses[element] / total_vapor_mass * 100 for element in elements],
#         linewidth=4,
#         linestyle=linestyles[index],
#         label=run["run_name"],
#     )
# ax.grid()
# ax.set_yscale("log")
# ax.legend(fontsize=18)
# plt.tight_layout()
# plt.show()
#
#
# # ========================= PLOT ELEMENTAL VAPOR MOLE FRACTION AS FUNCTION OF VMF =========================
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# ax.set_xlabel("Element", fontsize=20)
# ax.set_ylabel("Vapor Element Mole Fraction (%)", fontsize=20)
# linestyles = ["-", "--", "-."]
# for index, run in enumerate(runs):
#     vapor_masses = data[run["run_name"]]["vapor_element_mole_fraction_at_vmf"]
#     total_vapor_mass = sum(vapor_masses.values())
#     elements = list(vapor_masses.keys())
#     color_cycle = list(sns.color_palette("colorblind", len(elements)))
#     ax.plot(
#         elements,
#         [vapor_masses[element] / total_vapor_mass * 100 for element in elements],
#         linewidth=4,
#         linestyle=linestyles[index],
#         label=run["run_name"],
#     )
# ax.grid()
# ax.set_yscale("log")
# ax.set_ylim(bottom=10 ** -4)
# ax.legend(fontsize=18)
# plt.tight_layout()
# plt.show()


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
