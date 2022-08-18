import pandas as pd

"""
Notes: may be best to run MAGMA to sufficient VMF and then interpolate desired VMF.
"""

def read_MAGMA_file(path, iteration):
    """
    Returns MAGMA file as a dictionary
    :param path:
    :param iteration:
    :return:
    """
    reader = pd.read_csv(path + "/{}.csv".format(iteration), header=None)
    return {reader[0][index]: reader[1][index] for index in reader.index}

def get_vapor_masses(path, iteration):
    d_vap = read_MAGMA_file(path + "/atmosphere_total_mole_fraction", iteration)
    d_liq = read_MAGMA_file(path + "/atmosphere_total_mole_fraction", iteration)
    mass_liquid = d_vap["mass liquid"]
    mass_vapor = d_vap["initial melt mass"] - d_vap["mass liquid"]
