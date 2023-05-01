def check_mass_balance_theia_model(ejecta_dict, theia_dict, disk_mass_in_kg: float, theia_wt_pct: float):
    ejecta_composition_wt_pct = ejecta_dict['ejecta_composition']
    theia_mass_in_disk = disk_mass_in_kg * theia_wt_pct / 100.0
    ejecta_composition_mass = {
        oxide: ejecta_composition_wt_pct[oxide] / 100.0 * disk_mass_in_kg for oxide in ejecta_composition_wt_pct.keys()
    }
    theia_composition_mass = {
        oxide: theia_dict[oxide] / 100.0 * theia_mass_in_disk for oxide in theia_dict.keys()
    }
    bse_composition_mass = {
        oxide: ejecta_composition_mass[oxide] - theia_composition_mass[oxide] for oxide in ejecta_composition_mass.keys()
    }
