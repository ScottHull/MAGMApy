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
        "run_name": "",
        "temperature": 2682.61,  # K
        "vmf": 0.96,  # %
        "impactor%": 0.0,  # %
        "new_simulation": True,  # True to run a new simulation, False to load a previous simulation
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
    if os.path.exists(f"{run_name}/mass_distribution.csv"):
        os.remove(f"{run_name}/mass_distribution.csv")
    with open(f"{run_name}/mass_distribution.csv", "w") as f:
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
    print(f"wrote file {run_name}/mass_distribution.csv")
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



