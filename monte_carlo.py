from src.composition import Composition
from src.liquid_chemistry import LiquidActivity
from src.gas_chemistry import GasPressure
from src.thermosystem import ThermoSystem
from src.report import Report

from scipy.interpolate import interp1d

"""
Takes the MAMGApy code and uses it to run a Monte Carlo search for the composition of Theia.
"""

bse_composition = {
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

bsm_composition = {
    "SiO2": 44.60,
    'MgO': 35.10,
    'Al2O3': 3.90,
    'TiO2': 0.17,
    'Fe2O3': 0.00000,
    'FeO': 12.40,
    'CaO': 3.30,
    'Na2O': 0.050,
    'K2O': 0.004,
    'ZnO': 2.0e-4,
}


def is_function_increasing_or_decreasing(starting_composition: dict, ending_composition_dict: dict):
    """
    Returns -1 if the function is decreasing and +1 if the function is increasing.
    :param starting_composition:
    :param ending_composition_dict:
    :return:
    """
    return {oxide: -1 if ending_composition_dict[oxide] - starting_composition[oxide] < 0 else 1 for oxide in
            starting_composition.keys()}


def renormalize_composition(oxide_masses: dict):
    """
    Normalizes the dictionary to 100%.
    :param oxide_masses:
    :return:
    """
    total_mass = sum(oxide_masses.values())
    return {oxide: oxide_masses[oxide] / total_mass * 100 for oxide in oxide_masses.keys()}


def adjust_guess(previous_bulk_composition: dict, liquid_composition_at_vmf: dict, residuals: dict):
    """
    Returns a new guess to start the next Monte Carlo iteration by minimizing the residuals from the previous iteration.
    :return:
    """
    # find out if the function increases or decreases between the start and end compositions
    function_behavior = is_function_increasing_or_decreasing(previous_bulk_composition, liquid_composition_at_vmf)
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


def get_oxide_masses(oxide_wt_pct: dict, mass: float):
    """
    Takes the total mass and a dictionary of oxide weight percents and returns the absolute oxide masses.
    :param oxide_wt_pct:
    :return:
    """
    oxide_masses = {}
    for oxide in oxide_wt_pct.keys():
        oxide_masses[oxide] = oxide_wt_pct[oxide] * mass / 100
    return oxide_masses


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


def monte_carlo_search(starting_composition: dict, temperature: float, to_vmf: float):
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
                                       previous_liquid_composition, to_vmf)


earth_mass_fraction = 20  # %
theia_mass_fraction = 80  # %
disk_mass = 0.07346 * 10 ** 24  # kg, mass of the Moon
theia_mass = disk_mass * theia_mass_fraction / 100  # mass of theia in the disk
earth_mass = disk_mass * earth_mass_fraction / 100  # mass of earth in the disk

# begin the Monte Carlo search
iteration = 0
residual_error = 1e99  # assign a large number to the initial residual error
starting_composition = bse_composition  # set the starting composition to the BSE composition
temperature, vmf = 3664.25, 19.21  # 500b073S
print("Starting Monte Carlo search...")
while abs(residual_error) > 1e-6:  # while total residual error is greater than a small number
    iteration += 1
    composition_at_vmf = monte_carlo_search(starting_composition, temperature, vmf)  # run the Monte Carlo search
    # calculate the residuals
    residuals = {oxide: bsm_composition[oxide] - composition_at_vmf[oxide] if oxide != "Fe2O3" else 0.0 for oxide
                 in starting_composition.keys()}
    # calculate the total residual error
    residual_error = sum([abs(residuals[oxide]) for oxide in residuals.keys()])
    # adjust the guess
    starting_composition = adjust_guess(starting_composition, bse_composition, residuals)
    print(
        f"*** Iteration: {iteration}\nStarting composition: {starting_composition}\nResiduals: {residuals}\nResidual error: {residual_error}"
    )

print("FOUND SOLUTION!")
print(f"Starting composition: {starting_composition}")
