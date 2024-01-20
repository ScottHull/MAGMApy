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
from matplotlib.ticker import MaxNLocator

import matplotlib.cm as cm
import matplotlib.ticker as tck
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import labellines

# use colorblind-friendly colors
plt.style.use('seaborn-colorblind')
# increase font size
plt.rcParams.update({"font.size": 16})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

runs = [
    {
        "run_name": "G",
        "temperature": 2201.89,  # K
        "vmf": 0.7785655850744196,  # %
        "impactor%": 70.758 / 100,  # %
        "vapor_loss_fraction": 92 / 100,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    },
    {
        "run_name": "H",
        "temperature": 2271.61,  # K
        "vmf": 0.44644432484432994,  # %
        "impactor%": 65.81413641 / 100,  # %
        "vapor_loss_fraction": 88 / 100,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    },
    {
        "run_name": "K",
        "temperature": 2213.514879,  # K
        "vmf": 0.2747740008124389,  # %
        "impactor%": 16.6080938 / 100,  # %
        "vapor_loss_fraction": 89 / 100,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    },
    {
        "run_name": "L",
        "temperature": 2103.03,  # K
        "vmf": 0.42,  # %
        "impactor%": 71.23136432 / 100,  # %
        "vapor_loss_fraction": 85 / 100,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    },
]

# define the reservoirs
mars_composition = normalize({
    "SiO2": 44.33,
    'MgO': 31.01,
    'Al2O3': 3.13,
    'TiO2': 0.14,
    'Fe2O3': 0.00000,
    'FeO': 18.34,
    'CaO': 2.46,
    'Na2O': 0.55,
    'K2O': 0.04,
    'ZnO': 2.38e-3,
})

d_type_asteroid_composition = normalize({
    "SiO2": 34.26,
    'MgO': 25.16,
    'Al2O3': 2.63,
    'TiO2': 0.12,
    'Fe2O3': 0.00000,
    'FeO': 34.88,
    'CaO': 1.95,
    'Na2O': 0.84,
    'K2O': 0.11,
    'ZnO': 0.04,
})
oxides_ordered = [
    "Al2O3", "TiO2", "CaO", "MgO", "FeO", "SiO2", "K2O", "Na2O", "ZnO"
]
cations_ordered = ["Al", "Ti", "Ca", "Mg", "Fe", "Si", "K", "Na", "Zn"]

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


def recondense_vapor(melt_absolute_cation_masses: dict, vapor_absolute_cation_mass: dict, vapor_loss_fraction: float, bulk_composition: dict):
    """
    Recondenses retained vapor into the melt.
    :param vapor_absolute_mass:
    :param vapor_loss_fraction:
    :return:
    """
    lost_vapor_mass = {
        k: v * vapor_loss_fraction for k, v in vapor_absolute_cation_mass.items()
    }
    retained_vapor_mass = {
        k: v - lost_vapor_mass[k] for k, v in vapor_absolute_cation_mass.items()
    }
    recondensed_melt_mass = {
        k: v + retained_vapor_mass[k] for k, v in melt_absolute_cation_masses.items()
    }
    # convert to oxide mass fractions
    c = ConvertComposition().cations_mass_to_oxides_weight_percent(
        cations=recondensed_melt_mass, oxides=list(bulk_composition.keys())
    )
    # divide by 100 to get mass fraction
    return {
        "recondensed_melt_oxide_mass_fraction": {k: v for k, v in c.items()},
        "lost_vapor_mass": lost_vapor_mass,
        "retained_vapor_mass": retained_vapor_mass,
        "recondensed_melt_mass": recondensed_melt_mass
    }


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


def mix_target_impactor_composition(target_composition, impactor_composition, impactor_mass_fraction):
    """
    Mixes the target and impactor composition together.
    :param target_composition:
    :param impactor_composition:
    :param impactor_mass_fraction:
    :return:
    """
    target_mass_fraction = 1 - impactor_mass_fraction
    mixed_composition = {}
    for k, v in target_composition.items():
        mixed_composition[k] = v * target_mass_fraction + impactor_composition[k] * impactor_mass_fraction
    return mixed_composition

global_element_vmfs = {}
global_element_loss_fractions = {}
global_mixed_bulk_composition = {}
global_disk_bulk_composition_no_recondensation = {}
global_disk_bulk_composition_with_recondensation = {}
global_disk_bulk_composition_no_recondensation_relative_to_ic = {}
global_disk_bulk_composition_with_recondensation_relative_to_ic = {}
for i in [global_element_vmfs, global_element_loss_fractions]:
    i.update({"run_name": []})
    i.update({element: [] for element in cations_ordered})
global_mixed_bulk_composition.update({"run_name": []})
global_mixed_bulk_composition.update({oxide: [] for oxide in oxides_ordered})
for i in [global_disk_bulk_composition_no_recondensation, global_disk_bulk_composition_with_recondensation,
          global_disk_bulk_composition_no_recondensation_relative_to_ic,
          global_disk_bulk_composition_with_recondensation_relative_to_ic]:
    i.update({"run_name": []})
    i.update({oxide: [] for oxide in oxides_ordered})

# make a 1 column 3 row figure
fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex='all', sharey='all')

for run_index, run in enumerate(runs):
    # assign dictionary values to variables
    run_name = run["run_name"]
    temperature = run["temperature"]
    impactor_disk_mass_fraction = run["impactor%"]
    new_simulation = run["new_simulation"]
    mixed_composition = mix_target_impactor_composition(
        target_composition=mars_composition,
        impactor_composition=d_type_asteroid_composition,
        impactor_mass_fraction=impactor_disk_mass_fraction
    )

    for comp_index, (comp_name, bulk_composition) in enumerate(zip(['BSM', "D-type", "Mixed"],
                                           [mars_composition, d_type_asteroid_composition, mixed_composition])):

        if new_simulation:
            c = Composition(
                composition=bulk_composition
            )

            g = GasPressure(
                composition=c,
                major_gas_species=[
                    "SiO", "O2", "MgO", "Fe", "Ca", "Al", "Ti", "Na", "K", "Zn"
                ],
                minor_gas_species="__all__",
            )

            l = LiquidActivity(
                composition=c,
                complex_species="__all__",
                gas_system=g
            )

            t = ThermoSystem(composition=c, gas_system=g, liquid_system=l)

            reports = Report(composition=c, liquid_system=l, gas_system=g,
                             thermosystem=t, to_dir=run_name + f" ({comp_name})")

            count = 1
            while t.weight_fraction_vaporized < 0.2:
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
                if count % 50 == 0 or count <= 10:
                    reports.create_composition_report(iteration=count)
                    reports.create_liquid_report(iteration=count)
                    reports.create_gas_report(iteration=count)
                count += 1

        # get the residual melt data
        melt_oxide_as_func_vmf = collect_data(path=run_name + f" ({comp_name})/magma_oxide_mass_fraction",
                                                x_header='mass fraction vaporized')
        melt_cation_as_func_vmf = collect_data(path=run_name + f" ({comp_name})/magma_element_mass",
                                                x_header='mass fraction vaporized')
        vapor_cation_as_func_vmf = collect_data(path=run_name + f" ({comp_name})/total_vapor_element_mass",
                                                x_header='mass fraction vaporized')
        melt_oxide_at_vmf = get_composition_at_vmf(d=melt_oxide_as_func_vmf, vmf_val=run['vmf'])
        melt_elements_at_vmf = get_composition_at_vmf(d=melt_cation_as_func_vmf, vmf_val=run['vmf'])
        vapor_elements_at_vmf = get_composition_at_vmf(d=vapor_cation_as_func_vmf, vmf_val=run['vmf'])
        lost_vapor_elements_at_vmf = {
            element: val * run['vapor_loss_fraction'] for element, val in vapor_elements_at_vmf.items()
        }
        retained_vapor_elements_at_vmf = {
            element: val - lost_vapor_elements_at_vmf[element] for element, val in vapor_elements_at_vmf.items()
        }

        element_vmf = {
            element: vapor_elements_at_vmf[element] / (vapor_elements_at_vmf[element] + melt_elements_at_vmf[element]) * 100 for element in cations_ordered
        }
        hydrodynamic_loss_fraction = {
            element: lost_vapor_elements_at_vmf[element] / (vapor_elements_at_vmf[element] + melt_elements_at_vmf[element]) * 100 for element in cations_ordered
        }

        rc = recondense_vapor(
            melt_absolute_cation_masses=melt_elements_at_vmf,
            vapor_absolute_cation_mass=vapor_elements_at_vmf,
            vapor_loss_fraction=run['vapor_loss_fraction'],
            bulk_composition=bulk_composition
        )

        recondensed_melt_oxide_at_vmf = rc['recondensed_melt_oxide_mass_fraction']
        recondensed_melt_element_mass_at_vmf = rc['recondensed_melt_mass']

        if os.path.exists(f"{run_name} ({comp_name})/{run_name} ({comp_name})_compositions.csv"):
            os.remove(f"{run_name} ({comp_name})/{run_name} ({comp_name})_compositions.csv")
        with open(f"{run_name} ({comp_name})/{run_name} ({comp_name})_compositions.csv", "w") as f:
            f.write("component," + ",".join(i for i in oxides_ordered) + "\n")
            f.write("melt (not recondensed)," + ",".join(str(melt_oxide_at_vmf[i] * 100) for i in oxides_ordered) + "\n")
            f.write("melt (recondensed)," + ",".join(str(recondensed_melt_oxide_at_vmf[i]) for i in oxides_ordered) + "\n")
            f.write("component," + ",".join(i for i in cations_ordered) + "\n")
            f.write("melt (not recondensed)," + ",".join(str(melt_elements_at_vmf[i]) for i in cations_ordered) + "\n")
            f.write("melt (recondensed)," + ",".join(str(recondensed_melt_element_mass_at_vmf[i]) for i in cations_ordered) + "\n")
            f.write("vapor," + ",".join(str(vapor_elements_at_vmf[i]) for i in cations_ordered) + "\n")
            f.write("lost vapor," + ",".join(str(lost_vapor_elements_at_vmf[i]) for i in cations_ordered) + "\n")
            f.write("retained vapor," + ",".join(str(retained_vapor_elements_at_vmf[i]) for i in cations_ordered) + "\n")
            f.write("element VMF," + ",".join(str(element_vmf[i]) for i in cations_ordered) + "\n")
            f.write("hydrodynamic loss fraction," + ",".join(str(hydrodynamic_loss_fraction[i]) for i in cations_ordered) + "\n")
        f.close()

        run['results'] = {
            "melt_oxide_at_vmf": melt_oxide_at_vmf,
            "recondensed_melt_oxide_at_vmf": recondensed_melt_oxide_at_vmf,
            "melt_elements_at_vmf": melt_elements_at_vmf,
            "vapor_elements_at_vmf": vapor_elements_at_vmf,
            "lost_vapor_elements_at_vmf": lost_vapor_elements_at_vmf,
            "retained_vapor_elements_at_vmf": retained_vapor_elements_at_vmf,
            "element_vmf": element_vmf,
            "hydrodynamic_loss_fraction": hydrodynamic_loss_fraction,
        }

        for element in cations_ordered:
            val = element_vmf[element]
            if val < 0.01:
                # turn into a scientific notation string
                val = "{:.2e}".format(val)
            else:
                val = str(round(val, 2))
            global_element_vmfs[element].append(val)
            val = hydrodynamic_loss_fraction[element]
            if val < 0.01:
                # turn into a scientific notation string
                val = "{:.2e}".format(val)
            else:
                val = str(round(val, 2))
            global_element_vmfs[element].append
            global_element_loss_fractions[element].append(val)
        for oxide in oxides_ordered:
            val = mixed_composition[oxide]
            if val < 0.01:
                # turn into a scientific notation string
                val = "{:.2e}".format(val)
            else:
                val = str(round(val, 2))
            global_mixed_bulk_composition[oxide].append(val)
        global_mixed_bulk_composition["run_name"].append(f'{run_name} ({comp_name})')

        for oxide in oxides_ordered:
            val = melt_oxide_at_vmf[oxide] * 100
            if val < 0.01:
                # turn into a scientific notation string
                val = "{:.2e}".format(val)
            else:
                val = str(round(float(val), 2))
            global_disk_bulk_composition_no_recondensation[oxide].append(val)
            val = recondensed_melt_oxide_at_vmf[oxide]
            if val < 0.01:
                # turn into a scientific notation string
                val = "{:.2e}".format(val)
            else:
                val = str(round(val, 2))
            global_disk_bulk_composition_with_recondensation[oxide].append(val)

        for oxide in oxides_ordered:
            val = melt_oxide_at_vmf[oxide] * 100 / bulk_composition[oxide]
            if val < 0.01:
                # turn into a scientific notation string
                val = "{:.2e}".format(val)
            else:
                val = str(round(float(val), 2))
            global_disk_bulk_composition_no_recondensation_relative_to_ic[oxide].append(val)
            val = recondensed_melt_oxide_at_vmf[oxide] * 100 / bulk_composition[oxide]
            if val < 0.01:
                # turn into a scientific notation string
                val = "{:.2e}".format(val)
            else:
                val = str(round(val, 2))
            global_disk_bulk_composition_with_recondensation_relative_to_ic[oxide].append(val)


        global_element_vmfs["run_name"].append(f'{run_name} ({comp_name})')
        global_element_loss_fractions["run_name"].append(f'{run_name} ({comp_name})')
        global_disk_bulk_composition_no_recondensation["run_name"].append(f'{run_name} ({comp_name})')
        global_disk_bulk_composition_with_recondensation["run_name"].append(f'{run_name} ({comp_name})')
        global_disk_bulk_composition_no_recondensation_relative_to_ic["run_name"].append(f'{run_name} ({comp_name})')
        global_disk_bulk_composition_with_recondensation_relative_to_ic["run_name"].append(f'{run_name} ({comp_name})')

        axs[comp_index].plot(
            [format_species_string(i) for i in oxides_ordered],
            np.array([melt_oxide_at_vmf[i] * 100 / bulk_composition[i] for i in oxides_ordered]),
            linewidth=2.0,
            color=colors[run_index],
            label="Run " + run_name,
        )
        axs[comp_index].scatter(
            [format_species_string(i) for i in oxides_ordered],
            np.array([melt_oxide_at_vmf[i] * 100 / bulk_composition[i] for i in oxides_ordered]),
            color=colors[run_index],
            s=80
        )
        axs[comp_index].plot(
            [format_species_string(i) for i in oxides_ordered],
            np.array([recondensed_melt_oxide_at_vmf[i] / bulk_composition[i] for i in oxides_ordered]),
            linewidth=2.0,
            color=colors[run_index],
            linestyle="--",
        )
        axs[comp_index].scatter(
            [format_species_string(i) for i in oxides_ordered],
            np.array([recondensed_melt_oxide_at_vmf[i] / bulk_composition[i] for i in oxides_ordered]),
            color=colors[run_index],
            marker="D",
            s=80
        )

for comp_label, ax in zip(['BSM', "D-type Asteroid", "Mixed"], axs.flatten()):
    ax.text(
        0.05, 0.85, f"{comp_label}", transform=ax.transAxes, fontweight='bold', size=20
    )
    ax.set_ylabel(f"Disk / {comp_label}", fontsize=16)

for ax in axs.flatten():
    ax.grid()
    ax.set_yscale("log")
    ax.set_ylim(10 ** -3, 10 ** 1)
    ax.axhline(1, color='black', label="1:1")

legend = axs[0].legend(loc='lower left')
for line in legend.get_lines():
    line.set_linewidth(5.0)  # make lines thicker
plt.tight_layout()
plt.savefig("mars_disk_composition.png", format='png', dpi=200)

# make a 2 column 1 row figure
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex='all', sharey='all')
axs = axs.flatten()

for run_index, run in enumerate(runs):
    # on the left plot, plot the element VMF
    axs[0].plot(
        [format_species_string(i) for i in cations_ordered],
        np.array([run['results']['element_vmf'][i] for i in cations_ordered]),
        linewidth=2.0,
        color=colors[run_index],
        marker='o',
        markersize=6,
        label="Run " + run['run_name'],
    )
    # on the right plot, plot the hydrodynamic loss fraction
    axs[1].plot(
        [format_species_string(i) for i in cations_ordered],
        np.array([run['results']['hydrodynamic_loss_fraction'][i] for i in cations_ordered]),
        linewidth=2.0,
        color=colors[run_index],
        marker='o',
        markersize=6,
        label="Run " + run['run_name'],
    )

axs[0].set_ylabel("Element VMF (%)", fontsize=16)
axs[1].set_ylabel("Hydrodynamic Loss Fraction (%)", fontsize=16)
axs[1].legend(loc='lower right')
letters = string.ascii_lowercase
for index, ax in enumerate(axs):
    ax.grid()
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(tck.LogLocator(numticks=999, subs="auto"))
    # annotate the letter in the upper left corner
    ax.text(
        0.05, 0.90, f"{letters[index]}", transform=ax.transAxes, fontweight='bold', size=20
    )

plt.tight_layout()
plt.savefig("mars_element_vmf_and_loss_frac.png", format='png', dpi=200)

# output the globals to a latex table
vmf_table = pd.DataFrame(global_element_vmfs).to_latex(index=False)
if "mars_element_vmf.tex" in os.listdir():
    os.remove("mars_element_vmf.tex")
with open("mars_element_vmf.tex", "w") as f:
    f.write(vmf_table)
f.close()
hydrodynamic_loss_table = pd.DataFrame(global_element_loss_fractions).to_latex(index=False)
if "mars_hydrodynamic_loss.tex" in os.listdir():
    os.remove("mars_hydrodynamic_loss.tex")
with open("mars_hydrodynamic_loss.tex", "w") as f:
    f.write(hydrodynamic_loss_table)
f.close()
mixed_bulk_composition_table = pd.DataFrame(global_mixed_bulk_composition).to_latex(index=False)
if "mars_mixed_bulk_composition.tex" in os.listdir():
    os.remove("mars_mixed_bulk_composition.tex")
with open("mars_mixed_bulk_composition.tex", "w") as f:
    f.write(mixed_bulk_composition_table)
f.close()
disk_bulk_composition_no_recondensation_table = pd.DataFrame(global_disk_bulk_composition_no_recondensation).to_latex(index=False)
if "mars_disk_bulk_composition_no_recondensation.tex" in os.listdir():
    os.remove("mars_disk_bulk_composition_no_recondensation.tex")
with open("mars_disk_bulk_composition_no_recondensation.tex", "w") as f:
    f.write(disk_bulk_composition_no_recondensation_table)
f.close()
disk_bulk_composition_with_recondensation_table = pd.DataFrame(global_disk_bulk_composition_with_recondensation).to_latex(index=False)
if "mars_disk_bulk_composition_with_recondensation.tex" in os.listdir():
    os.remove("mars_disk_bulk_composition_with_recondensation.tex")
with open("mars_disk_bulk_composition_with_recondensation.tex", "w") as f:
    f.write(disk_bulk_composition_with_recondensation_table)
f.close()
disk_bulk_composition_no_recondensation_relative_to_ic_table = pd.DataFrame(global_disk_bulk_composition_no_recondensation_relative_to_ic).to_latex(index=False)
if "mars_disk_bulk_composition_no_recondensation_relative_to_ic.tex" in os.listdir():
    os.remove("mars_disk_bulk_composition_no_recondensation_relative_to_ic.tex")
with open("mars_disk_bulk_composition_no_recondensation_relative_to_ic.tex", "w") as f:
    f.write(disk_bulk_composition_no_recondensation_relative_to_ic_table)
f.close()
disk_bulk_composition_with_recondensation_relative_to_ic_table = pd.DataFrame(global_disk_bulk_composition_with_recondensation_relative_to_ic).to_latex(index=False)
if "mars_disk_bulk_composition_with_recondensation_relative_to_ic.tex" in os.listdir():
    os.remove("mars_disk_bulk_composition_with_recondensation_relative_to_ic.tex")
with open("mars_disk_bulk_composition_with_recondensation_relative_to_ic.tex", "w") as f:
    f.write(disk_bulk_composition_with_recondensation_relative_to_ic_table)
f.close()

disk_bulk_composition_no_recondensation_relative_to_ic_table = pd.DataFrame(global_disk_bulk_composition_no_recondensation_relative_to_ic)
disk_bulk_composition_no_recondensation_relative_to_ic_table.to_csv("mars_disk_bulk_composition_no_recondensation_relative_to_ic.csv", index=False)
disk_bulk_composition_with_recondensation_relative_to_ic_table = pd.DataFrame(global_disk_bulk_composition_with_recondensation_relative_to_ic)
disk_bulk_composition_with_recondensation_relative_to_ic_table.to_csv("mars_disk_bulk_composition_with_recondensation_relative_to_ic.csv", index=False)
