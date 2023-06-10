import numpy as np
from isotopes import rayleigh

runs = [
    {
        "run_name": "Canonical Model",
        "temperature": 2682.61,  # K
        "vmf": 0.96,  # %
        "disk_theia_mass_fraction": 66.78,  # %
        "disk_mass": 1.02,  # lunar masses
        "vapor_loss_fraction": 74.0,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    },
    {
        "run_name": "Half-Earths Model",
        "temperature": 3517.83,  # K
        "vmf": 4.17,  # %
        "disk_theia_mass_fraction": 51.97,  # %
        "disk_mass": 1.70,  # lunar masses
        "vapor_loss_fraction": 16.0,  # %
        "new_simulation": False,  # True to run a new simulation, False to load a previous simulation
    }
]

f_vaporization_fractions = np.arange(0.01, 99, .1)
delta_K_BSE = -0.479  # mean of 41K/39K isotope ratios from BSE
delta_K_Lunar_BSE = 0.415  # mean of 41K/39K isotope ratios from lunar samples
delta_K_Lunar_BSE_std_error = 0.05  # standard error of 41K/39K isotope ratios from lunar samples

Delta_K_EM = []

for run in runs:
    for f in f_vaporization_fractions:
        total_mass = 100
        bulk_vapor_mass = f * total_mass
        bulk_melt_mass = (1 - f) * total_mass
        r = rayleigh.FullSequenceRayleighDistillation_SingleReservior(
            heavy_z=41,
            light_z=39,
            vapor_escape_fraction=run['vapor_loss_fraction'],
            system_element_mass=total_mass,
            melt_element_mass=bulk_melt_mass,
            vapor_element_mass=bulk_vapor_mass,
            earth_isotope_composition=delta_K_BSE,
            theia_ejecta_fraction=0,
            total_melt_mass=total_mass,
            total_vapor_mass=sum(
                [mass_distribution[i]['bulk vapor mass'] for i in mass_distribution.keys() if len(i) < 3]),
        )
