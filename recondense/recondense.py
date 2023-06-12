from src.composition import ConvertComposition

def recondense_vapor(melt_element_masses: dict, bulk_vapor_element_masses: dict, vapor_loss_fraction: float,
                     oxides: list):
    """
    Recondenses the retained vapor back into the melt, assuming full recondensation.
    :param melt_element_masses: dict of element masses in the melt
    :param bulk_vapor_element_masses: dict of element masses in the bulk vapor
    :param vapor_loss_fraction: fraction of the vapor that was lost
    """
    total_element_mass = {
        element: melt_element_masses[element] + bulk_vapor_element_masses[element] for element in
        melt_element_masses.keys()
    }
    lost_vapor_element_masses = {element: mass * vapor_loss_fraction for element, mass in
                                 bulk_vapor_element_masses.items()}
    retained_vapor_element_masses = {element: mass - lost_vapor_element_masses[element] for element, mass in
                                     bulk_vapor_element_masses.items()}
    recondensed_melt_element_masses = {element: melt_element_masses[element] + retained_vapor_element_masses[element]
                                       for element in melt_element_masses.keys()}
    recondensed_melt_composition = ConvertComposition().cations_mass_to_oxides_weight_percent(
        recondensed_melt_element_masses, oxides)
    total_retained_element_masses = {
        element: melt_element_masses[element] + retained_vapor_element_masses[element] for element in
        melt_element_masses.keys()
    }
    fraction_element_retained = {element: total_retained_element_masses[element] / total_element_mass[element] for
                                    element in total_element_mass.keys()}
    fraction_retained_element_in_melt = {element: melt_element_masses[element] / total_retained_element_masses[element]
                                            for element in total_retained_element_masses.keys()}
    return {
        "original_melt_element_masses": recondensed_melt_element_masses,
        "bulk_vapor_element_masses": bulk_vapor_element_masses,
        "lost_vapor_element_masses": lost_vapor_element_masses,
        "retained_vapor_element_masses": retained_vapor_element_masses,
        "total_element_masses": total_element_mass,
        "recondensed_melt_element_masses": recondensed_melt_element_masses,
        "recondensed_melt_oxide_composition": recondensed_melt_composition,
        "total_retained_element_masses": total_retained_element_masses,
        "fraction_element_retained": fraction_element_retained,
        "fraction_retained_element_in_melt": fraction_retained_element_in_melt,
    }

def no_recondense_vapor(melt_element_masses: dict, bulk_vapor_element_masses: dict, vapor_loss_fraction: float,
                     oxides: list):
    """
    Does not recondense the retained vapor back into the melt, assuming no recondensation.
    :param melt_element_masses: dict of element masses in the melt
    :param bulk_vapor_element_masses: dict of element masses in the bulk vapor
    :param vapor_loss_fraction: fraction of the vapor that was lost
    """
    total_element_mass = {
        element: melt_element_masses[element] + bulk_vapor_element_masses[element] for element in
        melt_element_masses.keys()
    }
    lost_vapor_element_masses = {element: mass for element, mass in
                                 bulk_vapor_element_masses.items()}
    retained_vapor_element_masses = {element: mass - lost_vapor_element_masses[element] for element, mass in
                                     bulk_vapor_element_masses.items()}
    recondensed_melt_element_masses = {element: melt_element_masses[element] + retained_vapor_element_masses[element]
                                       for element in melt_element_masses.keys()}
    recondensed_melt_composition = ConvertComposition().cations_mass_to_oxides_weight_percent(
        recondensed_melt_element_masses, oxides)
    total_retained_element_masses = {
        element: melt_element_masses[element] + retained_vapor_element_masses[element] for element in
        melt_element_masses.keys()
    }
    fraction_element_retained = {element: total_retained_element_masses[element] / total_element_mass[element] for
                                    element in total_element_mass.keys()}
    fraction_retained_element_in_melt = {element: melt_element_masses[element] / total_retained_element_masses[element]
                                            for element in total_retained_element_masses.keys()}
    return {
        "original_melt_element_masses": recondensed_melt_element_masses,
        "bulk_vapor_element_masses": bulk_vapor_element_masses,
        "lost_vapor_element_masses": lost_vapor_element_masses,
        "retained_vapor_element_masses": retained_vapor_element_masses,
        "total_element_masses": total_element_mass,
        "recondensed_melt_element_masses": recondensed_melt_element_masses,
        "recondensed_melt_oxide_composition": recondensed_melt_composition,
        "total_retained_element_masses": total_retained_element_masses,
        "fraction_element_retained": fraction_element_retained,
        "fraction_retained_element_in_melt": fraction_retained_element_in_melt,
    }
