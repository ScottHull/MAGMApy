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
    return {
        'theia_weight_pct': theia_weight_pct,
        'theia_moles': theia_moles,
        'theia_cations': theia_cations,
    }

