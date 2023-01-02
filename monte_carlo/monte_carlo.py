import copy

from src.composition import Composition, ConvertComposition, get_element_in_base_oxide, oxygen_accounting
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data

import os
import sys
import shutil
import numpy as np
from scipy.interpolate import interp1d


def renormalize_composition(oxide_masses: dict):
    """
    Normalizes the dictionary to 100%.
    :param oxide_masses:
    :return:
    """
    total_mass = sum(oxide_masses.values())
    return {oxide: oxide_masses[oxide] / total_mass * 100 for oxide in oxide_masses.keys()}


def interpolate_elements_at_vmf(at_vmf, at_composition, previous_vmf, previous_composition, target_vmf):
    """
    Interpolates composition at target VMF.
    :return:
    """

    vmfs = [previous_vmf, at_vmf]
    compositions = {oxide: [previous_composition[oxide], at_composition[oxide]] for oxide in
                    previous_composition.keys()}
    interpolated_elements = {}
    for oxide in compositions.keys():
        interp = interp1d(vmfs, compositions[oxide])
        interpolated_elements[oxide] = interp(target_vmf)
    return renormalize_composition(interpolated_elements)


def adjust_guess(previous_bulk_composition: dict, liquid_composition_at_vmf: dict, residuals: dict):
    """
    Returns a new guess to start the next Monte Carlo iteration by minimizing the residuals from the previous iteration.
    :return:
    """
    # find out if the function increases or decreases between the start and end compositions
    # adjust based on function behavior and the residuals
    new_starting_composition = {}
    for oxide in previous_bulk_composition.keys():
        if oxide == "Fe2O3":  # set 0 if Fe2O3
            new_starting_composition[oxide] = 0.0
        else:
            new_starting_composition[oxide] = previous_bulk_composition[oxide] + residuals[
                oxide]
        if new_starting_composition[oxide] < 0.0:  # to prevent negative composition adjustments
            new_starting_composition[oxide] = previous_bulk_composition[oxide]
    # renormalize the composition and return
    return renormalize_composition(new_starting_composition)

def return_vmf_and_element_lists(data):
    """
    Returns a list of VMF values and a dictionary of lists of element values.
    :param data:
    :return:
    """
    vmf_list = list(sorted(data.keys()))
    elements_at_vmf = {element: [data[vmf][element] for vmf in vmf_list] for element in data[vmf_list[0]].keys()}
    return vmf_list, elements_at_vmf


def find_best_fit_vmf(vmfs: list, composition: dict, target_composition: dict, restricted_composition=None):
    """
    Find the vmf with the lowest residual error between all composition.
    :param target_composition: given in wt% (assumes composition is normalized to 100)
    :param restricted_composition: Do not include these composition in the fit.
    :param vmfs:
    :param composition: wt% composition of the liquid, assumes normaalized to 1
    :return:
    """
    if restricted_composition is None:
        restricted_composition = []
    best_vmf = vmfs[0]
    best_error = np.inf
    for vmf in vmfs:
        error = 0
        for element in composition.keys():
            if element not in restricted_composition:
                error += ((target_composition[element] / 100) - composition[element][vmfs.index(vmf)]) ** 2
        if error < best_error:
            best_error = error
            best_vmf = vmf
    return best_vmf * 100

def write_file(data, metadata, to_path, filename):
    """
    Writes the metadata and data to a CSV file.
    :param data:
    :param metadata:
    :param to_path:
    :return:
    """
    if os.path.exists(to_path + "/" + filename):
        os.remove(to_path + "/" + filename)
    with open(to_path + "/" + filename, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key},{value}\n")
        for key, value in data.items():
            f.write(f"{key},{value}\n")
    f.close()


def __monte_carlo_search(starting_composition: dict, temperature: float, to_vmf: float):
    """
    The basic loop of a single Monte Carlo run.
    :param to_vmf:
    :param starting_composition:
    :param ending_composition:
    :param temperature:
    :return:
    """
    print("Beginning MAGMApy calculations...")
    major_gas_species = [
        "SiO", "O2", "MgO", "Fe", "Ca", "Al", "Ti", "Na", "K", "ZnO", "Zn"
    ]
    c = Composition(
        composition=starting_composition
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
    previous_vmf = 0  # the previous VMF
    previous_liquid_composition = {}  # stores the previous iteration's liquid composition
    iteration = 0  # the current iteration
    while t.weight_fraction_vaporized * 100 < to_vmf:
        iteration += 1
        previous_vmf = t.weight_fraction_vaporized * 100
        previous_liquid_composition = l.liquid_oxide_mass_fraction
        output_interval = 100
        if t.weight_fraction_vaporized * 100.0 > 5:  # vmf changes very fast towards end of simulation
            output_interval = 5
        if 80 < t.weight_fraction_vaporized:
            output_interval = 50
        l.calculate_activities(temperature=temperature)
        g.calculate_pressures(temperature=temperature, liquid_system=l)
        if l.counter == 1:
            l.calculate_activities(temperature=temperature)
            g.calculate_pressures(temperature=temperature, liquid_system=l)
        fraction = 0.05  # fraction of most volatile element to lose
        t.vaporize(fraction=fraction)
        l.counter = 0  # reset Fe2O3 counter for next vaporization step

    print(f"Finished MAGMApy calculations ({iteration} Iterations).")
    # the model has finshed, now we need to interpolate the composition at the target VMF
    if previous_liquid_composition == l.liquid_oxide_mass_fraction:
        raise ValueError("The starting and ending compositions are the same.")
    return interpolate_elements_at_vmf(t.weight_fraction_vaporized * 100, l.liquid_oxide_mass_fraction, previous_vmf,
                                       previous_liquid_composition, to_vmf), c, l, g, t


def __run_full_MAGMApy(composition, target_composition, temperature, to_vmf=90.0, to_dir="monte_carlo_full_solution"):
    major_gas_species = [
        "SiO", "O2", "MgO", "Fe", "Ca", "Al", "Ti", "Na", "K", "ZnO", "Zn"
    ]

    c = Composition(
        composition=composition
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

    reports = Report(composition=c, liquid_system=l, gas_system=g, thermosystem=t, to_dir=to_dir)
    count = 1
    while t.weight_fraction_vaporized * 100 < to_vmf:
        output_interval = 100
        if t.weight_fraction_vaporized * 100.0 > 5:  # vmf changes very fast towards end of simulation
            output_interval = 5
        if 80 < t.weight_fraction_vaporized:
            output_interval = 50
        l.calculate_activities(temperature=temperature)
        g.calculate_pressures(temperature=temperature, liquid_system=l)
        if l.counter == 1:
            l.calculate_activities(temperature=temperature)
            g.calculate_pressures(temperature=temperature, liquid_system=l)
        fraction = 0.05  # fraction of most volatile element to lose
        t.vaporize(fraction=fraction)
        l.counter = 0  # reset Fe2O3 counter for next vaporization step
        print("[~] At iteration: {} (Weight Fraction Vaporized: {} %)".format(count,
                                                                              round(t.weight_fraction_vaporized * 100.0,
                                                                                    4)))
        if count % output_interval == 0 or count == 1:
            reports.create_composition_report(iteration=count)
            reports.create_liquid_report(iteration=count)
            reports.create_gas_report(iteration=count)
        count += 1
    data = collect_data(path="{}/magma_oxide_mass_fraction".format(to_dir), x_header='mass fraction vaporized')
    vmf_list, elements = return_vmf_and_element_lists(data)
    best_vmf = round(find_best_fit_vmf(vmfs=vmf_list, composition=elements, target_composition=target_composition), 2)
    return c, l, g, t, best_vmf


def run_monte_carlo(initial_composition: dict, target_composition: dict, temperature: float, vmf: float,
                    full_run_vmf=90.0, full_report_path="theia_composition", sum_residuals_for_success=0.55,
                    starting_comp_filename="starting_composition.csv", delete_dir=True):
    # build report path
    if delete_dir:
        if os.path.exists(full_report_path):
            shutil.rmtree(full_report_path)
    if not os.path.exists(full_report_path):
        os.mkdir(full_report_path)
    # begin the Monte Carlo search
    iteration = 0
    residual_error = 1e99  # assign a large number to the initial residual error
    starting_composition = initial_composition  # set the starting composition to the BSE composition
    print("Starting Monte Carlo search...")
    while abs(residual_error) > sum_residuals_for_success and iteration <= 20:  # while total residual error is greater than a small number
        iteration += 1
        composition_at_vmf, c, l, g, t = __monte_carlo_search(starting_composition, temperature,
                                                              vmf)  # run the Monte Carlo search
        # calculate the residuals
        residuals = {oxide: target_composition[oxide] - composition_at_vmf[oxide] if oxide != "Fe2O3" else 0.0 for
                     oxide
                     in starting_composition.keys()}
        # calculate the total residual error
        residual_error = sum([abs(residuals[oxide]) for oxide in residuals.keys()])
        # adjust the guess
        starting_composition = adjust_guess(starting_composition, initial_composition, residuals)
        print(
            f"*** Iteration: {iteration}\nStarting composition: {starting_composition}\nResiduals: {residuals}\n"
            f"Residual error: {residual_error}"
        )
        if abs(residual_error) > sum_residuals_for_success:
            print("Calculation has NOT yet converged. Continuing search...")
    if iteration > 20:
        print("FAILED TO FIND SOLUTION!")
        return None

    print("FOUND SOLUTION!")
    # write starting composition and metadata to file
    print(f"Starting composition: {starting_composition}")
    best_vmf = None
    if full_run_vmf is not None:
        print("Running full solution...")
        c, l, g, t, best_vmf = __run_full_MAGMApy(
            composition=starting_composition, target_composition=target_composition, temperature=temperature,
            to_vmf=full_run_vmf, to_dir=full_report_path
        )
        print("Finished full solution.")

    metadata = {
        "vmf": vmf,
        "best vmf": best_vmf,
        "temperature": temperature,
    }
    write_file(data=starting_composition, metadata=metadata, filename=starting_comp_filename,
               to_path=full_report_path)
    return starting_composition

def run_monte_carlo_mp(args):
    initial_composition, target_composition, temperature, vmf, \
    full_run_vmf, full_report_path, sum_residuals_for_success = args
    # build report path
    if os.path.exists(full_report_path):
        shutil.rmtree(full_report_path)
    os.mkdir(full_report_path)
    # begin the Monte Carlo search
    iteration = 0
    residual_error = 1e99  # assign a large number to the initial residual error
    starting_composition = initial_composition  # set the starting composition to the BSE composition
    print("Starting Monte Carlo search...")
    while abs(residual_error) > sum_residuals_for_success and iteration <= 20:  # while total residual error is greater than a small number
        iteration += 1
        composition_at_vmf, c, l, g, t = __monte_carlo_search(starting_composition, temperature,
                                                              vmf)  # run the Monte Carlo search
        # calculate the residuals
        residuals = {oxide: target_composition[oxide] - composition_at_vmf[oxide] if oxide != "Fe2O3" else 0.0 for
                     oxide
                     in starting_composition.keys()}
        # calculate the total residual error
        residual_error = sum([abs(residuals[oxide]) for oxide in residuals.keys()])
        # adjust the guess
        starting_composition = adjust_guess(starting_composition, initial_composition, residuals)
        print(
            f"*** Iteration: {iteration}\nStarting composition: {starting_composition}\nResiduals: {residuals}\n"
            f"Residual error: {residual_error}"
        )
    if iteration > 20:
        print("FAILED TO FIND SOLUTION!")
        return None


    print("FOUND SOLUTION!")
    # write starting composition and metadata to file
    print(f"Starting composition: {starting_composition}")
    best_vmf = None
    if full_run_vmf is not None:
        print("Running full solution...")
        c, l, g, t, best_vmf = __run_full_MAGMApy(
            composition=starting_composition, target_composition=target_composition, temperature=temperature,
            to_vmf=full_run_vmf, to_dir=full_report_path
        )
        print("Finished full solution.")

    metadata = {
        "vmf": vmf,
        "best vmf": best_vmf,
        "temperature": temperature,
    }
    write_file(data=starting_composition, metadata=metadata, filename="starting_composition.csv",
                 to_path=full_report_path)
    return starting_composition

def run_monte_carlo_vapor_loss(initial_composition: dict, target_composition: dict, temperature: float, vmf: float, vapor_loss_fraction: float,
                    full_run_vmf=90.0, full_report_path="theia_composition", sum_residuals_for_success=10,
                    starting_comp_filename="starting_composition.csv", delete_dir=True):
    """
    We need to account for the fact that only a portion of the vapor is lost and the rest is retained and will recondense.
    Ultimately, the bulk composition of the disk needs to be the same as the target composition.
    We assume that the only mass lost is that of the escaping vapor.
    We know the VMF of the ejecta.  We know the fraction of this vapor that should escape.
    We must adjust this composition so that the bulk liquid + retained vapor composition is that of the bulk Moon.
    To accomplish this, we run MAGMApy to the given VMF.  At this VMF, some fraction of the vapor is lost and some is retained.
    The total bulk composition of the liquid + retained vapor is the same as the target composition.
    :param initial_composition:
    :param target_composition:
    :param temperature:
    :param vmf:
    :param full_run_vmf:
    :param full_report_path:
    :param sum_residuals_for_success:
    :param starting_comp_filename:
    :param delete_dir:
    :return:
    """
    # build report path
    if delete_dir:
        if os.path.exists(full_report_path):
            shutil.rmtree(full_report_path)
    if not os.path.exists(full_report_path):
        os.mkdir(full_report_path)
    # begin the Monte Carlo search
    if vapor_loss_fraction > 1.0:
        raise ValueError("Vapor loss fraction must be less than 1.0.")
    iteration = 0
    residual_error = 1e99  # assign a large number to the initial residual error
    starting_composition = initial_composition  # set the starting composition to the BSE composition
    print("Starting Monte Carlo search...")
    while abs(residual_error) > sum_residuals_for_success:  # while total residual error is greater than a small number
        iteration += 1
        composition_at_vmf, c, l, g, t = __monte_carlo_search(starting_composition, temperature,
                                                              vmf)  # run the Monte Carlo search  # TODO: is there an interpolation here?

        # get the mass of the liquid and the vapor and their constituent species/elements
        liquid_mass = l.melt_mass
        vapor_mass = g.vapor_mass
        liquid_species_masses = l.cation_mass
        # add in O
        liquid_species_masses["O"] = liquid_mass - sum(liquid_species_masses.values())
        vapor_species_masses = g.species_total_mass
        vapor_element_masses = g.element_total_mass

        # get a list of all oxides in the melt
        oxides = [oxide for oxide in composition_at_vmf.keys() if oxide != "Fe2O3"]

        # get the liquid elemental abundances
        liquid_cation_masses = c.liquid_abundances
        liquid_cation_masses["O"] = liquid_mass - sum(liquid_cation_masses.values())


        # multiply the vapor species/element masses by the fraction of the vapor that is lost
        vapor_species_masses_lost = {species: vapor_species_masses[species] * vapor_loss_fraction for species in
                                vapor_species_masses.keys()}
        vapor_element_masses_lost = {element: vapor_element_masses[element] * vapor_loss_fraction for element in
                                vapor_element_masses.keys()}
        # multiply the liquid species/element masses by the fraction of the vapor that is retained
        vapor_species_masses_retained = {species: vapor_species_masses[species] * (1.0 - vapor_loss_fraction) for species in
                                vapor_species_masses.keys()}
        vapor_element_masses_retained = {element: vapor_element_masses[element] * (1.0 - vapor_loss_fraction) for element in
                                vapor_element_masses.keys()}

        # add back in the retained vapor element masses to the liquid element masses
        liquid_element_masses = {element: liquid_cation_masses[element] + vapor_element_masses_retained[element] for element in
                            liquid_cation_masses.keys()}

        print(f"liquid_category_masses: {liquid_element_masses}\nvapor_element_masses_retained: {vapor_element_masses_retained}")
        print(f"vapor retained mass: {sum(vapor_element_masses_retained.values())}\nvapor lost mass: {sum(vapor_element_masses_lost.values())}\nliquid mass: {sum(liquid_element_masses.values())}")
        print("alskdfjlaskdfj", (sum(vapor_element_masses_retained.values()) + sum(vapor_element_masses_lost.values())) / (sum(liquid_element_masses.values()) + sum(vapor_element_masses_retained.values()) + sum(vapor_element_masses_lost.values())) * 100)


        # do oxygen accounting
        leftover_oxygen = oxygen_accounting(liquid_element_masses, oxides)

        new_liquid_mass = sum(liquid_element_masses.values())

        liquid_element_masses_element_mass = copy.copy(liquid_element_masses)
        # convert new liquid masses to oxide wt%
        liquid_element_masses = c.cations_mass_to_oxides_weight_percent(liquid_element_masses, oxides)

        composition_at_vmf_without_recondensed_vapor = copy.copy(composition_at_vmf)
        composition_at_vmf = liquid_element_masses

        # ensure that mass is conserved at all steps
        # that the vapor species masses lost is equal to the vapor element masses lost
        assert abs(sum(vapor_species_masses_lost.values()) - sum(vapor_element_masses_lost.values())) < 1e-6
        print('passed mass conservation check 1')
        # that the vapor species masses retained is equal to the vapor element masses retained
        assert abs(sum(vapor_species_masses_retained.values()) - sum(vapor_element_masses_retained.values())) < 1e-6
        print('passed mass conservation check 2')
        # that the retained vapor mass and lost vapor mass is equal to the total vapor mass
        assert abs(sum(vapor_species_masses_lost.values()) + sum(vapor_species_masses_retained.values()) - vapor_mass) < 1e-6
        print('passed mass conservation check 3')
        # that the sum of the new liquid mass plus the lost vapor mass is equal to the total mass
        assert abs(new_liquid_mass + sum(vapor_species_masses_lost.values()) - (liquid_mass + vapor_mass)) < 1e-6
        print('passed mass conservation check 4')

        # calculate the residuals
        print(">>>", composition_at_vmf, liquid_element_masses)
        residuals = {oxide: target_composition[oxide] - composition_at_vmf[oxide] if oxide != "Fe2O3" else 0.0 for
                     oxide
                     in starting_composition.keys()}
        # calculate the total residual error
        residual_error = sum([abs(residuals[oxide]) for oxide in residuals.keys()])
        # adjust the guess
        print(
            f"*** Iteration: {iteration}\nStarting composition: {starting_composition}\nTarget composition: {target_composition}\nResiduals: {residuals}\n"
            f"Residual error: {residual_error}\nComposition without recondensed vapor: {composition_at_vmf_without_recondensed_vapor}\n"
            f"Composition with recondensed vapor: {composition_at_vmf}\nLeftover oxygen: {leftover_oxygen} "
            f"(total liquid atoms: {sum(liquid_element_masses_element_mass.values())} "
            f"total liquid O atoms: {liquid_element_masses_element_mass['O']})\nVMF: {vapor_mass / (liquid_mass + vapor_mass) * 100.0}\n")
        if abs(residual_error) > sum_residuals_for_success:
            print("Calculation has NOT yet converged. Continuing search...")
            starting_composition = adjust_guess(starting_composition, initial_composition, residuals)

    print("FOUND SOLUTION!")
    # write starting composition and metadata to file
    print(f"Starting composition: {starting_composition}")
    best_vmf = None
    if full_run_vmf is not None:
        print("Running full solution...")
        c, l, g, t, best_vmf = __run_full_MAGMApy(
            composition=starting_composition, target_composition=target_composition, temperature=temperature,
            to_vmf=full_run_vmf, to_dir=full_report_path
        )
        print("Finished full solution.")

    metadata = {
        "vmf": vmf,
        "best vmf": best_vmf,
        "temperature": temperature,
    }
    write_file(data=starting_composition, metadata=metadata, filename=starting_comp_filename,
               to_path=full_report_path)
    return starting_composition
