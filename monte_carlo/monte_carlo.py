import copy

from src.composition import Composition, ConvertComposition, get_element_in_base_oxide, oxygen_accounting, \
    get_molecular_mass
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report
from src.plots import collect_data

from recondense.recondense import recondense_vapor

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


def interpolate_elements_at_vmf(at_vmf, at_composition, previous_vmf, previous_composition, target_vmf,
                                normalize=False):
    """
    Interpolates composition at target VMF.
    :return:
    """

    vmfs = [previous_vmf, at_vmf]
    compositions = {i: [previous_composition[i], at_composition[i]] for i in previous_composition.keys()}
    interpolated_elements = {}
    for i in compositions.keys():
        interp = interp1d(vmfs, compositions[i])
        interpolated_elements[i] = interp(target_vmf).item()
    if normalize:
        return renormalize_composition(interpolated_elements)
    return interpolated_elements


def adjust_guess(previous_bulk_composition: dict, residuals: dict):
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
    previous_vapor_species_mass = {}  # stores the previous iteration's vapor species mass
    previous_vapor_element_mass = {}  # stores the previous iteration's vapor element mass
    previous_liquid_cation_mass = {}  # stores the previous iteration's liquid cation mass
    previous_liquid_mass = {}  # stores the previous iteration's liquid mass
    previous_vapor_mass = {}  # stores the previous iteration's vapor mass
    iteration = 0  # the current iteration
    while t.weight_fraction_vaporized < to_vmf:
        if iteration % 20 == 0:
            print(f"VMF: {t.weight_fraction_vaporized * 100}%, Iteration: {iteration}")
        iteration += 1
        previous_vmf = t.weight_fraction_vaporized
        previous_liquid_composition = copy.copy(l.liquid_oxide_mass_fraction)
        previous_vapor_species_mass = copy.copy(g.species_total_mass)
        previous_vapor_element_mass = copy.copy(g.element_total_mass)
        previous_liquid_cation_mass = copy.copy(l.cation_mass)
        previous_liquid_mass = copy.copy(l.melt_mass)
        previous_vapor_mass = copy.copy(g.vapor_mass)
        output_interval = 100
        if t.weight_fraction_vaporized > 5:  # vmf changes very fast towards end of simulation
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
    liquid_composition_at_vmf = interpolate_elements_at_vmf(t.weight_fraction_vaporized,
                                                            l.liquid_oxide_mass_fraction, previous_vmf,
                                                            previous_liquid_composition, to_vmf, normalize=True)
    vapor_species_at_vmf = interpolate_elements_at_vmf(t.weight_fraction_vaporized, g.species_total_mass,
                                                       previous_vmf,
                                                       previous_vapor_species_mass, to_vmf)
    vapor_element_at_vmf = interpolate_elements_at_vmf(t.weight_fraction_vaporized, g.element_total_mass,
                                                       previous_vmf,
                                                       previous_vapor_element_mass, to_vmf)
    liquid_cation_at_vmf = interpolate_elements_at_vmf(t.weight_fraction_vaporized, l.cation_mass, previous_vmf,
                                                       previous_liquid_cation_mass, to_vmf)
    # interpolate the liquid mass and vapor mass at the target VMF
    liquid_mass_at_vmf = interp1d([previous_vmf, t.weight_fraction_vaporized],
                                  [previous_liquid_mass, l.melt_mass])(to_vmf).item()
    vapor_mass_at_vmf = interp1d([previous_vmf, t.weight_fraction_vaporized],
                                 [previous_vapor_mass, g.vapor_mass])(to_vmf).item()
    # get the value from liquid_mass_at_vmf and vapor_mass_at_vmf
    # return the results as a dictionary
    return {
        "liquid_composition_at_vmf": liquid_composition_at_vmf,
        "vapor_species_mass_at_vmf": vapor_species_at_vmf,
        "vapor_element_mass_at_vmf": vapor_element_at_vmf,
        "liquid_cation_mass_at_vmf": liquid_cation_at_vmf,
        # "liquid_mass_at_vmf": liquid_mass_at_vmf,
        # "vapor_mass_at_vmf": vapor_mass_at_vmf,
        'c': c,
        'l': l,
        'g': g,
        't': t,
    }


def run_full_MAGMApy(composition, target_composition, temperature, to_vmf=90.0, to_dir="monte_carlo_full_solution"):
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
    while abs(
            residual_error) > sum_residuals_for_success and iteration <= 20:  # while total residual error is greater than a small number
        iteration += 1
        composition_at_vmf, vapor_species_at_vmf, vapor_element_at_vmf, liquid_cation_at_vmf, liquid_mass_at_vmf, vapor_mass_at_vmf, c, l, g, t = __monte_carlo_search(
            starting_composition, temperature,
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
        c, l, g, t, best_vmf = run_full_MAGMApy(
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
    while abs(
            residual_error) > sum_residuals_for_success and iteration <= 20:  # while total residual error is greater than a small number
        iteration += 1
        composition_at_vmf, vapor_species_at_vmf, vapor_element_at_vmf, liquid_cation_at_vmf, liquid_mass_at_vmf, vapor_mass_at_vmf, c, l, g, t = __monte_carlo_search(
            starting_composition, temperature,
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
        c, l, g, t, best_vmf = run_full_MAGMApy(
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


def run_monte_carlo_vapor_loss(initial_composition: dict, target_composition: dict, temperature: float, vmf: float,
                               vapor_loss_fraction: float,
                               full_run_vmf=90.0, full_report_path="theia_composition", sum_residuals_for_success=0.55,
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
    composition_at_vmf_without_recondensed_vapor = {}  # initialize the composition at the given VMF without recondensed vapor
    vapor_species_masses_lost = {}  # initialize a dictionary to hold the masses of the vapor species lost
    vapor_element_masses_lost = {}  # initialize a dictionary to hold the masses of the vapor elements lost
    vapor_species_masses_retained = {}  # initialize a dictionary to hold the masses of the vapor species retained
    vapor_element_masses_retained = {}  # initialize a dictionary to hold the masses of the vapor elements retained
    liquid_cation_masses = {}  # initialize a dictionary to hold the masses of the liquid cations
    composition_at_vmf = {}  # initialize the composition at the given VMF (includes recondensed vapor)
    liquid_element_masses_absolute = {}  # initialize a dictionary to hold the absolute masses of the liquid elements
    new_liquid_mass = 0.0  # initialize a variable to hold the new liquid mass (after vapor recondensation)
    c, l, g, t = None, None, None, None
    print("Starting Monte Carlo search...")
    while abs(residual_error) > sum_residuals_for_success:  # while total residual error is greater than a small number
        iteration += 1
        data = __monte_carlo_search(starting_composition, temperature, vmf)  # run the Monte Carlo search
        composition_at_vmf, vapor_species_masses, vapor_element_masses, liquid_cation_masses, liquid_mass, \
            vapor_mass, c, l, g, t = data.values()
        oxides = list(composition_at_vmf.keys())

        # # get the mass of the liquid and the vapor and their constituent species/elements
        # liquid_mass = l.melt_mass
        # vapor_mass = g.vapor_mass
        # # add in O
        # vapor_species_masses = g.species_total_mass
        # vapor_element_masses = g.element_total_mass
        #
        # # get a list of all oxides in the melt
        # oxides = [oxide for oxide in composition_at_vmf.keys() if oxide != "Fe2O3"]
        #
        # # get the liquid elemental abundances
        # liquid_cation_masses = l.cation_mass

        # multiply the vapor species/element masses by the fraction of the vapor that is lost
        vapor_species_masses_lost = {species: vapor_species_masses[species] * vapor_loss_fraction for species in
                                     vapor_species_masses.keys()}
        vapor_element_masses_lost = {element: vapor_element_masses[element] * vapor_loss_fraction for element in
                                     vapor_element_masses.keys()}
        # multiply the liquid species/element masses by the fraction of the vapor that is retained
        vapor_species_masses_retained = {species: vapor_species_masses[species] * (1.0 - vapor_loss_fraction) for
                                         species in
                                         vapor_species_masses.keys()}
        vapor_element_masses_retained = {element: vapor_element_masses[element] * (1.0 - vapor_loss_fraction) for
                                         element in
                                         vapor_element_masses.keys()}

        # add back in the retained vapor element masses to the liquid element masses
        liquid_element_masses = {element: liquid_cation_masses[element] + vapor_element_masses_retained[element] for
                                 element in
                                 liquid_cation_masses.keys()}

        # do oxygen accounting
        leftover_oxygen = oxygen_accounting(liquid_element_masses, oxides)

        new_liquid_mass = sum(liquid_element_masses.values())

        liquid_element_masses_element_mass = copy.copy(liquid_element_masses)
        # convert new liquid masses to oxide wt%
        liquid_element_masses_absolute = copy.copy(liquid_element_masses)
        liquid_element_masses = c.cations_mass_to_oxides_weight_percent(liquid_element_masses, oxides)

        composition_at_vmf_without_recondensed_vapor = copy.copy(composition_at_vmf)
        composition_at_vmf = liquid_element_masses

        # ensure that mass is conserved at all steps
        # that the vapor species masses lost is equal to the vapor element masses lost
        # assert that the sum of the liquid cation masses is equal to the liquid mass
        assert abs(
            sum(liquid_cation_masses.values()) - liquid_mass) < 1e-6, "Liquid cation masses do not sum to the liquid mass."
        assert abs(sum(vapor_species_masses_lost.values()) - sum(vapor_element_masses_lost.values())) < 1e-6
        print('passed mass conservation check 1')
        # that the vapor species masses retained is equal to the vapor element masses retained
        assert abs(sum(vapor_species_masses_retained.values()) - sum(vapor_element_masses_retained.values())) < 1e-6
        print('passed mass conservation check 2')
        # that the retained vapor mass and lost vapor mass is equal to the total vapor mass
        assert abs(
            sum(vapor_species_masses_lost.values()) + sum(vapor_species_masses_retained.values()) - vapor_mass) < 1e-6
        print('passed mass conservation check 3')
        # that the sum of the new liquid mass plus the lost vapor mass is equal to the total mass
        assert abs(new_liquid_mass + sum(vapor_species_masses_lost.values()) - (liquid_mass + vapor_mass)) < 1e-6
        print('passed mass conservation check 4')
        # make sure that composition_at_vmf equals 100%
        assert np.isclose(sum(composition_at_vmf.values()), 100.0), "Composition at VMF does not sum to 100%."

        # calculate the residuals
        residuals = {oxide: target_composition[oxide] - composition_at_vmf[oxide] if oxide != "Fe2O3" else 0.0 for
                     oxide
                     in starting_composition.keys()}
        # calculate the total residual error
        residual_error = sum([abs(residuals[oxide]) for oxide in residuals.keys()])
        # adjust the guess
        print(
            f"*** Iteration: {iteration}\nStarting composition: {starting_composition}\nTarget composition: {target_composition}\nResiduals: {residuals}\n"
            f"Residual error: {residual_error}\nComposition without recondensed vapor: {composition_at_vmf_without_recondensed_vapor}\n"
            f"Composition with recondensed vapor: {composition_at_vmf}\nVapor Fraction: {(sum(vapor_species_masses_lost.values()) + sum(vapor_species_masses_retained.values())) / sum(liquid_cation_masses.values()) * 100.0} // {t.weight_fraction_vaporized * 100}\n")
        if abs(residual_error) > sum_residuals_for_success:
            print("Calculation has NOT yet converged. Continuing search...")
            starting_composition = adjust_guess(starting_composition, residuals)

    print("FOUND SOLUTION!")
    # write starting composition and metadata to file
    print(f"Starting composition: {starting_composition}")
    best_vmf = None
    if full_run_vmf is not None:
        print("Running full solution...")
        c, l, g, t, best_vmf = run_full_MAGMApy(
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
    return {
        "starting composition": starting_composition,
        "vapor_lost_composition": composition_at_vmf_without_recondensed_vapor,
        "vapor_element_masses_lost": vapor_element_masses_lost,
        "vapor_element_masses_retained": vapor_element_masses_retained,
        "vapor_species_masses_lost": vapor_species_masses_lost,
        "vapor_species_masses_retained": vapor_species_masses_retained,
        "liquid_composition_with_recondensed_vapor": composition_at_vmf,
        "liquid_composition_with_recondensed_vapor_element_masses": liquid_element_masses_absolute,
        "liquid_mass_with_recondensed_vapor": new_liquid_mass,
        'c': c,
        'l': l,
        'g': g,
        't': t,
    }


def test(guess_initial_composition: dict, target_composition: dict, temperature: float, vmf: float,
         vapor_loss_fraction: float,
         full_run_vmf=90.0, full_report_path="theia_composition", sum_residuals_for_success=0.55,
         starting_comp_filename="starting_composition.csv", delete_dir=True):
    # normalize fractional abundances to 1 if they are not already
    if vmf > 1:
        print("Warning: The VMF input is expected to be a fractional value, not a percentage.")
    if vapor_loss_fraction > 1:
        print("Warning: The vapor loss fraction input is expected to be a fractional value, not a percentage.")

    # track some metadata
    iteration = 0
    residual_error = 1e99  # assign a large number to the initial residual error
    initial_composition = copy.copy(guess_initial_composition)  # this is what we are searching to find

    # run the simulation
    while abs(residual_error) > sum_residuals_for_success:  # while total residual error is greater than a small number
        data = __monte_carlo_search(initial_composition, temperature, vmf)  # run the Monte Carlo search
        liquid_composition_at_vmf, vapor_species_masses, vapor_element_masses, liquid_cation_masses, liquid_mass, \
            vapor_mass, c, l, g, t = data.values()  # unpack the interpolated data

        # liquid composition at vmf: the composition of the liquid at the vapor mass fraction
        # vapor species masses: the mass of each vapor species (complex species, not elements)
        # vapor element masses: the mass of each element in the vapor
        # liquid cation masses: the mass of each cation in the liquid
        # liquid mass: the mass of the liquid
        # vapor mass: the mass of the vapor
        # c: the composition object
        # l: the liquid object
        # g: the gas object
        # t: the thermodynamic object

        # ======================== CALCULATE PRE-LOSS VAPOR/LIQUID MOLE FRACTIONS ========================
        # # calculate the vapor mole fractions
        # vapor_mole_fractions_pre_loss = {i: vapor_species_masses[i] / get_molecular_mass(i) * 100 for i in vapor_species_masses.keys()}
        # # calculate the liquid mole fractions
        # liquid_mole_fractions_pre_loss = {i: liquid_cation_masses[i] / get_molecular_mass(i) * 100 for i in liquid_cation_masses.keys()}

        # ======================== RECONDENSE RETAINED VAPOR ========================
        vapor_species_masses_lost = {species: vapor_species_masses[species] * vapor_loss_fraction for species in
                                     vapor_species_masses.keys()}
        vapor_element_masses_lost = {element: vapor_element_masses[element] * vapor_loss_fraction for element in
                                     vapor_element_masses.keys()}
        # multiply the liquid species/element masses by the fraction of the vapor that is retained
        vapor_species_masses_retained = {species: vapor_species_masses[species] * (1.0 - vapor_loss_fraction) for
                                         species in
                                         vapor_species_masses.keys()}
        vapor_element_masses_retained = {element: vapor_element_masses[element] * (1.0 - vapor_loss_fraction) for
                                         element in
                                         vapor_element_masses.keys()}

        # add back in the retained vapor element masses to the liquid element masses
        liquid_element_masses = {element: liquid_cation_masses[element] + vapor_element_masses_retained[element] for
                                 element in
                                 liquid_cation_masses.keys()}
        new_liquid_mass = sum(liquid_element_masses.values())  # calculate the new liquid mass (with recondensed vapor)
        # calculate the new liquid oxide wt pct (with recondensed vapor)
        new_liquid_oxide_wt_pct = c.cations_mass_to_oxides_weight_percent(liquid_element_masses,
                                                                          liquid_composition_at_vmf.keys())

        # ======================== CALCULATE POST-LOSS VAPOR/LIQUID MOLE FRACTIONS ========================
        # # calculate the vapor mole fractions
        # vapor_mole_fractions_post_loss = {i: vapor_species_masses[i] / get_molecular_mass(i) * 100 for i in vapor_species_masses.keys()}
        # # calculate the liquid mole fractions
        # liquid_mole_fractions_post_loss = {i: liquid_element_masses[i] / get_molecular_mass(i) * 100 for i in liquid_element_masses.keys()}

        # assess whether the target composition has been reached at the given VMF
        # calculate the residuals
        residuals = {oxide: target_composition[oxide] - new_liquid_oxide_wt_pct[oxide] if oxide != "Fe2O3" else 0.0 for
                     oxide
                     in initial_composition.keys()}
        # calculate the total residual error
        residual_error = sum([abs(residuals[oxide]) for oxide in residuals.keys()])

        # print out the results
        print(
            f"Iteration: {iteration}, \n"
            f"Residual error: {residual_error}, \n"
            f"Vapor mass fraction: {vmf * 100}, \n"
            f"Vapor loss fraction: {vapor_loss_fraction}, \n",
            f"Target composition: {target_composition}, \n",
            f"Initial composition: {initial_composition}, \n",
            f"Predicted Liquid Composition at VMF (without recondensed vapor): {liquid_composition_at_vmf}, \n",
            f"Predicted Liquid Composition at VMF (with recondensed vapor): {new_liquid_oxide_wt_pct}",
        )

        # if the residual error is too large, adjust the initial composition and run again
        if abs(residual_error) > sum_residuals_for_success:
            print("Calculation has NOT yet converged. Continuing search...")
            initial_composition = adjust_guess(initial_composition, residuals)
        else:
            print("Calculation has converged. Stopping search.")
            print("Running full solution...")
            c_, l_, g_, t_, best_vmf = run_full_MAGMApy(
                composition=initial_composition, target_composition=target_composition, temperature=temperature,
                to_vmf=full_run_vmf, to_dir=full_report_path
            )
            print("Finished full solution.")
            # return the results
            return {
                "ejecta composition": initial_composition,
                "liquid composition at vmf (w/o recondensed vapor)": liquid_composition_at_vmf,
                "liquid composition at vmf (w/ recondensed vapor)": new_liquid_oxide_wt_pct,
                # "liquid element mole fractions (before recondensation)": liquid_mole_fractions_pre_loss,
                # "liquid element mole fractions (after recondensation)": liquid_mole_fractions_post_loss,
                "vapor species (before loss/recondensation)": vapor_species_masses,
                "vapor elements (before loss/recondensation)": vapor_element_masses,
                # "vapor species mole fractions (before loss)": vapor_mole_fractions_pre_loss,
                # "vapor species mole fractions (after loss)": vapor_mole_fractions_post_loss,
                "vapor species masses lost": vapor_species_masses_lost,
                "vapor species masses retained": vapor_species_masses_retained,
                "vapor element masses lost": vapor_element_masses_lost,
                "vapor element masses retained": vapor_element_masses_retained,
                "liquid mass (w/ recondensed vapor)": new_liquid_mass,
                "c": c,
                "l": l,
                "g": g,
                "t": t,
            }


def theia_mixing(guess_initial_composition: dict, target_composition: dict, bse_composition: dict, temperature: float,
                 vmf: float, vapor_loss_fraction: float, full_run_vmf=90.0, full_report_path="theia_composition",
                 sum_residuals_for_success=0.55, target_melt_composition_type='recondensed'):
    # track some metadata
    iteration = 0
    residual_error = 1e99  # assign a large number to the initial residual error
    initial_composition = copy.copy(guess_initial_composition)  # this is what we are searching to find
    while abs(residual_error) > sum_residuals_for_success:  # while total residual error is greater than a small number
        data = __monte_carlo_search(initial_composition, temperature, vmf)  # run the Monte Carlo search
        recondensed_model = recondense_vapor(
            melt_element_masses=data["liquid_cation_at_vmf"], bulk_vapor_element_masses=data["vapor_element_at_vmf"],
            vapor_loss_fraction=vapor_loss_fraction, oxides=list(initial_composition.keys())
        )
        for key in recondensed_model.keys():
            data[f"recondensed__{key}"] = recondensed_model[key]
        target_melt_composition = data['liquid_composition_at_vmf']
        if target_melt_composition_type == 'recondensed':
            target_melt_composition = recondensed_model['recondensed_melt_oxide_composition']
        # calculate the residuals
        residuals = {oxide: target_composition[oxide] - target_melt_composition[oxide] if oxide != "Fe2O3" else 0.0
                     for oxide in initial_composition.keys()}
        # calculate the total residual error
        residual_error = sum([abs(residuals[oxide]) for oxide in residuals.keys()])

        # print out the results
        print(
            f"Iteration: {iteration}, \n"
            f"Residual error: {residual_error}, \n"
            f"Vapor mass fraction: {vmf * 100}, \n"
            f"Vapor loss fraction: {vapor_loss_fraction}"
        )

        # if the residual error is too large, adjust the initial composition and run again
        if abs(residual_error) > sum_residuals_for_success:
            print("Calculation has NOT yet converged. Continuing search...")
            initial_composition = adjust_guess(initial_composition, residuals)
        else:
            print("Calculation has converged. Stopping search.")
            print("Running full solution...")

            print("Finished full solution.")
            data.update({'ejecta_composition': initial_composition})
            # return the results
            return data
