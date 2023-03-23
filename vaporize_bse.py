from src.composition import Composition, ConvertComposition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data, make_figure

from math import log10
from random import uniform
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import labellines

# use colorblind-friendly colors
plt.style.use('seaborn-colorblind')

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

        # add each data set to the dictionary
        data[run]["melt_oxide_mass_fraction"] = melt_oxide_mass_fraction
        data[run]["magma_element_mass"] = magma_element_mass
        data[run]["vapor_species_mass"] = vapor_species_mass
        data[run]["vapor_element_mass"] = vapor_element_mass
        data[run]["vapor_species_mass_fraction"] = vapor_species_mass_fraction
        data[run]["vapor_element_mass_fraction"] = vapor_element_mass_fraction
    return data


# get run data
data = get_all_data_for_runs()

# make a plot tracking liquid and vapor composition as a function of VMF
# first, create a 2x2 grid of plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()
# set a shared x and y axis label
fig.supxlabel("VMF (%)", fontsize=14)
fig.supylabel("Mass Fraction (%)", fontsize=14)
for ax in axs:
    ax.grid(alpha=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-2)
# in the upper row, plot the liquid composition for each run
to_plot = 0
for run in data.keys():
    magma_plot_index = to_plot
    vapor_plot_index = to_plot + 1
    axs[magma_plot_index].set_title(f"{run} - Liquid")
    axs[vapor_plot_index].set_title(f"{run} - Vapor")
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
            label=species,
        )
    for index, species in enumerate(vapor_species):
        axs[vapor_plot_index].plot(
            np.array(list(vapor.keys())) * 100,
            np.array([vapor[i][species] for i in vapor.keys()]) * 100,
            linewidth=2,
            color=vapor_colors[index],
            label=species,
        )
    for ax in [axs[magma_plot_index], axs[vapor_plot_index]]:
        labellines.labelLines(ax.get_lines(), zorder=2.5, align=True, xvals=[uniform(1e-2, 30) for i in ax.get_lines()])
    to_plot += 2
fig.tight_layout()
plt.show()











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
