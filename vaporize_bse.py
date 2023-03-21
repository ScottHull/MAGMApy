from src.composition import Composition, ConvertComposition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data, make_figure

from math import log10
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# use colorblind-friendly colors
plt.style.use('seaborn-colorblind')

run_name = "BSE_starting_composition"
temperature = 2682.61  # K
vmf = 19.3  # %
vapor_loss_fraction = 100  # %
new_simulation = False

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
            "[~] At iteration: {} (Magma Fraction Vaporized: {} %)".format(count, t.weight_fraction_vaporized * 100.0))
        if count % 50 == 0 or count == 1:
            reports.create_composition_report(iteration=count)
            reports.create_liquid_report(iteration=count)
            reports.create_gas_report(iteration=count)
        count += 1

# collect data
melt_oxide_mass_fraction = collect_data(path=f"{run_name}/magma_oxide_mass_fraction",
                                        x_header='mass fraction vaporized')
magma_element_mass = collect_data(path=f"{run_name}/magma_element_mass",
                                  x_header='mass fraction vaporized')
vapor_species_mass_fraction = collect_data(path=f"{run_name}/total_vapor_species_mass",
                                           x_header='mass fraction vaporized')
vapor_element_mass = collect_data(path=f"{run_name}/total_vapor_element_mass",
                                  x_header='mass fraction vaporized')

# collect some metadata about the simulation
vmfs = list(melt_oxide_mass_fraction.keys())
oxides = [i for i in bse_composition.keys() if i != "Fe2O3"]
vapor_species = [i for i in vapor_species_mass_fraction[vmfs[0]].keys()]
elements = [i for i in vapor_element_mass[vmfs[0]].keys()]

def iterpolate_at_vmf(vmf_val):
    """
    Interpolate the melt and vapor composition at the specified vmf.
    :param vmf_val:
    :return:
    """
    # interpolate the melt and vapor composition at the specified vmf
    magma_oxide_mass_fraction_at_vmf = {}
    magma_element_mass_at_vmf = {}
    vapor_species_mass_fraction_at_vmf = {}
    vapor_element_mass_at_vmf = {}
    
    # interpolate each at the given vmf
    for oxide in oxides:
        magma_oxide_mass_fraction_at_vmf[oxide] = interp1d(
            list(melt_oxide_mass_fraction.keys()),
            [i[oxide] for i in melt_oxide_mass_fraction.values()]
        )(vmf_val / 100.0)
    for element in elements:
        magma_element_mass_at_vmf[element] = interp1d(
            list(magma_element_mass.keys()),
            [i[element] for i in magma_element_mass.values()]
        )(vmf_val / 100.0)
        vapor_element_mass_at_vmf[element] = interp1d(
            list(vapor_element_mass.keys()),
            [i[element] for i in vapor_element_mass.values()]
        )(vmf_val / 100.0)
    for species in vapor_species:
        vapor_species_mass_fraction_at_vmf[species] = interp1d(
            list(vapor_species_mass_fraction.keys()),
            [i[species] for i in vapor_species_mass_fraction.values()]
        )(vmf_val / 100.0)

    return magma_oxide_mass_fraction_at_vmf, magma_element_mass_at_vmf, \
            vapor_species_mass_fraction_at_vmf, vapor_element_mass_at_vmf

def recondense_vapor(vapor_element_mass_at_vmf_dict: dict, magma_element_mass_at_vmf_dict: dict,
                     vapor_loss_fraction_val: float):
    """
    Recondense the vapor at the specified vapor loss fraction.
    :param vapor_element_mass_at_vmf_dict:
    :param magma_element_mass_at_vmf_dict:
    :param vapor_loss_fraction_val:
    :return:
    """
    # calculate the elemental mass of the vapor after hydrodynamic vapor loss
    vapor_element_mass_after_hydrodynamic_vapor_loss = {}
    for element in elements:
        vapor_element_mass_after_hydrodynamic_vapor_loss[element] = vapor_element_mass_at_vmf_dict[element] * \
                                                                    (1 - (vapor_loss_fraction_val / 100.0))
    # add this back to the magma assuming full elemental recondensation
    magma_element_mass_after_recondensation = {}
    for element in elements:
        magma_element_mass_after_recondensation[element] = magma_element_mass_at_vmf_dict[element] + \
                                                           vapor_element_mass_after_hydrodynamic_vapor_loss[element]
    # renormalize the magma elemental mass after recondensation to oxide weight percent
    magma_oxide_mass_fraction_after_recondensation = ConvertComposition().cations_mass_to_oxides_weight_percent(
        magma_element_mass_after_recondensation, oxides=oxides
    )
    return magma_oxide_mass_fraction_after_recondensation


magma_oxide_mass_fraction_at_vmf, magma_element_mass_at_vmf, \
            vapor_species_mass_fraction_at_vmf, vapor_element_mass_at_vmf = iterpolate_at_vmf(vmf_val=vmf)
magma_oxide_mass_fraction_after_recondensation = recondense_vapor(
    vapor_element_mass_at_vmf_dict=vapor_element_mass_at_vmf,
    magma_element_mass_at_vmf_dict=magma_element_mass_at_vmf,
    vapor_loss_fraction_val=vapor_loss_fraction
)

# plot the bulk oxide composition of the BSE as it vaporizes
# and compare it to the bulk oxide composition of the Moon
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
# get a unique color for each oxide
colors = plt.cm.jet(np.linspace(0, 1, len(oxides)))
for index, oxide in enumerate(oxides):
    ax.plot(
        np.array(list(melt_oxide_mass_fraction.keys())) * 100,
        np.array([i[oxide] for i in melt_oxide_mass_fraction.values()]) * 100,
        color=colors[index],
        linewidth=2,
        label=oxide
    )
ax.axvline(
    x=vmf,
    color="black",
    linestyle="--",
    linewidth=2,
    # label=f"VMF: {round(vmf, 2)}%"
)
# scatter the interpolated bulk oxide composition of the BSE at the given vmf
for oxide in oxides:
    ax.scatter(
        vmf,
        magma_oxide_mass_fraction_at_vmf[oxide] * 100,
        color=colors[oxides.index(oxide)],
        s=100,
        zorder=10
    )
ax.set_xlabel("Magma Fraction Vaporized (%)", fontsize=16)
ax.set_ylabel("Bulk Oxide Composition (wt. %)", fontsize=16)
ax.set_title(
    "Bulk Oxide Composition of the BSE as it Vaporizes",
)
ax.grid()
ax.set_yscale("log")
# set a lower y limit of 10 ** -4
ax.set_ylim(bottom=10 ** -4)
# add a legend to the right of the plot
ax.legend(
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=16,
    frameon=False
)
plt.show()

# make a spider plot showing the composition of the recondensed BSE melt relative to the Moon
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
# get a unique color for each oxide
ax.plot(
    oxides,
    np.array([magma_oxide_mass_fraction_after_recondensation[oxide] / bulk_moon_composition[oxide] for oxide in oxides]),
    color='black',
    linewidth=2,
)
# label the x-axis with the oxide names
# ax.set_xticklabels(oxides)
ax.set_title(
    "Recondensed BSE Melt Composition Relative to the Moon",
    fontsize=16
)
ax.axhline(
    y=1,
    color="red",
    linestyle="--",
    linewidth=2,
    label="1:1 with Bulk Moon"
)
# annotate the VMF, the temperature, and the vapor loss fraction
ax.annotate(
    f"VMF: {round(vmf, 2)}%\nVapor Loss Fraction: {round(vapor_loss_fraction, 2)}%\nTemperature: {round(temperature, 2)} K",
    xy=(0.5, 0.90),
    xycoords="axes fraction",
    fontsize=16,
    horizontalalignment="center",
    verticalalignment="center"
)
ax.set_yscale("log")
ax.grid()
ax.legend()
plt.show()

# make the same figure, but do it over a range of vmfs
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)

for vmf_val in np.linspace(0.1, 85, 15):
    magma_oxide_mass_fraction_at_vmf, magma_element_mass_at_vmf, \
        vapor_species_mass_fraction_at_vmf, vapor_element_mass_at_vmf = iterpolate_at_vmf(vmf_val=vmf_val)
    magma_oxide_mass_fraction_after_recondensation = recondense_vapor(
        vapor_element_mass_at_vmf_dict=vapor_element_mass_at_vmf,
        magma_element_mass_at_vmf_dict=magma_element_mass_at_vmf,
        vapor_loss_fraction_val=vapor_loss_fraction
    )
    ax.plot(
        oxides,
        np.array([magma_oxide_mass_fraction_after_recondensation[oxide] / bulk_moon_composition[oxide] for oxide in oxides]),
        linewidth=2,
        label=f"VMF: {round(vmf_val, 2)}%"
    )
# label the x-axis with the oxide names
ax.set_title(
    "Recondensed BSE Melt Composition Relative to the Moon",
    fontsize=16
)
ax.axhline(
    y=1,
    color="red",
    linestyle="--",
    linewidth=2,
    label="1:1 with Bulk Moon"
)
# annotate the VMF, the temperature, and the vapor loss fraction
ax.annotate(
    f"VMF: {round(vmf, 2)}%\nVapor Loss Fraction: {round(vapor_loss_fraction, 2)}%\nTemperature: {round(temperature, 2)} K",
    xy=(0.5, 0.90),
    xycoords="axes fraction",
    fontsize=16,
    horizontalalignment="center",
    verticalalignment="center"
)
ax.set_yscale("log")
ax.grid()
ax.legend()
plt.show()
