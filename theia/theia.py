import numpy as np

from src.composition import ConvertComposition, normalize


def get_theia_composition(starting_composition, earth_composition, disk_mass, earth_mass):
    starting_weights = {oxide: starting_composition[oxide] / 100 * disk_mass for oxide in starting_composition.keys()}
    earth_composition = normalize(earth_composition)
    bse_weights = {oxide: earth_composition[oxide] / 100 * earth_mass for oxide in earth_composition.keys()}
    theia_weights = {oxide: starting_weights[oxide] - bse_weights[oxide] for oxide in starting_weights.keys()}
    theia_weight_pct = {oxide: theia_weights[oxide] / sum(theia_weights.values()) * 100 for oxide in
                        theia_weights.keys()}
    theia_moles = ConvertComposition().mass_to_moles(theia_weights)
    theia_cations = ConvertComposition().oxide_to_cations(theia_moles)
    # theia_x_si = {cation: theia_cations[cation] / theia_cations['Si'] for cation in theia_cations.keys()}
    # theia_x_al = {cation: theia_cations[cation] / theia_cations['Al'] for cation in theia_cations.keys()}
    # make sure that the theia weights and the BSE weights sum to the starting weights and equal the disk mass
    assert np.isclose(sum(theia_weights.values()) + sum(bse_weights.values()), sum(starting_weights.values()))
    assert np.isclose(sum(theia_weights.values()) + sum(bse_weights.values()), disk_mass)
    return {
        'theia_weight_pct': theia_weight_pct,
        'theia_moles': theia_moles,
        'theia_cations': theia_cations,
    }


def recondense_vapor(melt_mass_element: dict, vapor_mass_element: dict, vapor_mass_loss_fraction: float, oxides: list):
    """
    Recondenses the unescaped vapor back into the melt.
    :param melt_mass_element: Given by the collect_data method for liquid bulk element mass.
    :param vapor_mass_element: Given by the collect_data method for vapor bulk element mass.
    :param vapor_mass_loss_fraction: Given by hydrodynamic loss simulation.
    :return:
    """
    recondensed_melt_mass = {}
    escaping_vapor_mass = {}
    recondensed_melt_oxide_weight_pct = {}
    for vmf, melt_composition in melt_mass_element.items():
        vapor_mass = vapor_mass_element[vmf]
        recondensed_melt_mass[vmf] = {oxide: melt_composition[oxide] + (vapor_mass[oxide] * vapor_mass_loss_fraction)
                                      for oxide in melt_composition.keys()}
        escaping_vapor_mass[vmf] = {oxide: vapor_mass[oxide] - (vapor_mass[oxide] * vapor_mass_loss_fraction) for
                                    oxide in vapor_mass.keys()}

    # convert melt element mass to oxide weight percent
    for vmf, melt_mass in recondensed_melt_mass.items():
        recondensed_melt_oxide_weight_pct[vmf] = ConvertComposition().cations_mass_to_oxides_weight_percent(
            {i: j for i, j in melt_mass.items() if i != 'O'}, oxides
        )

    return {
        'recondensed_melt_mass': recondensed_melt_mass,
        'escaping_vapor_mass': escaping_vapor_mass,
        'recondensed_melt_oxide_weight_pct': recondensed_melt_oxide_weight_pct,
    }

def get_mass_sourced_from_bse_and_theia(theia_mass_fraction: float, bse_composition: dict, bst_composition: dict):
    """
    Given the composition of the system, returns the absolute mass of each element sourced from the BSE and Theia.
    :param theia_mass_fraction:
    :param bse_composition:
    :param bst_composition:
    :return:
    """
    if theia_mass_fraction > 1:
        raise ValueError('Theia mass fraction cannot be greater than 1.')
    if sum(bse_composition.values()) != sum(bst_composition.values()) != 100:
        raise ValueError('BSE and BST compositions must sum to 100 individually.')

