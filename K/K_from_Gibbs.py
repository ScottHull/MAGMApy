import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

products = ["Si_g", "O_g"]
reactants = ["SiO2_l"]
standard_state_temp = 298.15
temperatures = np.arange(100, 4000 + 100, 100)
base_path = "/Users/scotthull/Documents - Scott’s MacBook Pro/PhD Research/MAGMApy/K"
# base_path = r"C:\Users\Scott\Documents\MAGMApy\K"

def magma_code_SiO2_l(temperature):
    return 22.13 - (94311.0 / temperature)

def read_janaf_file(species):
    path = base_path + "/" + species + ".dat"
    df = pd.read_csv(path, sep="\t", skiprows=1, index_col="T(K)")
    return df

def get_standard_state(species, standard_t):
    return read_janaf_file(species).loc[standard_t]

def get_temperatures_from_janaf(products):
    """
    Returns the temperatures from the Janaf file.
    :param products:
    :return:
    """
    t = list(read_janaf_file(products).index.values)
    return [i for i in t if 2000 <= float(i) <= 4000]

def get_reaction_thermo(products, reactants, temperature, standard_state_temp):
    """
    Returns the thermo properties of a reaction.
    Products - Reacants for both delta H and delta S.
    :param products:
    :param reactants:
    :return:
    """
    R = 8.314  # J/Mol-K
    # R = 0.008314 # kJ/mol-K
    product_sum_deltaH = 0
    reactant_sum_deltaH = 0
    product_sum_deltaS = 0
    reactant_sum_deltaS = 0
    for prod in products:
        if prod == "O_g":
            product_sum_deltaH += 2 * float(get_standard_state(prod, standard_state_temp)["delta-f H"])
            product_sum_deltaS += 2 * float(get_standard_state(prod, standard_state_temp)["-[G-H(Tr)]/T"])
        else:
            product_sum_deltaH += float(get_standard_state(prod, standard_state_temp)["delta-f H"])
            product_sum_deltaS += float(get_standard_state(prod, standard_state_temp)["-[G-H(Tr)]/T"])
    for react in reactants:
        reactant_sum_deltaH += float(get_standard_state(react, standard_state_temp)["delta-f H"])
        reactant_sum_deltaS += float(get_standard_state(react, standard_state_temp)["-[G-H(Tr)]/T"])
    deltaH = (product_sum_deltaH - reactant_sum_deltaH) * 1000  # S in in J, and H is in kJ
    deltaS = product_sum_deltaS - reactant_sum_deltaS
    delta_G = deltaH - temperature * deltaS
    logK = -delta_G / (R * temperature) / 2.303  # lnK --> log10K to match JANAF
    return delta_G, deltaH, deltaS, logK

def linear_regression(x, y):
    """
    y = mx + b
    :param x:
    :param y:
    :return:
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept

def regress_temperature_against_logK(temperatures, logKs, temperature):
    slope, intercept = linear_regression(1 / temperatures, logKs)
    return slope / temperature + intercept


# temperatures = get_temperatures_from_janaf(products[0])
logKs = [get_reaction_thermo(products, reactants, t, standard_state_temp)[3] for t in temperatures]
regressed_logKs = [regress_temperature_against_logK(temperatures, logKs, t) for t in temperatures]
slope, intercept = linear_regression(1 / temperatures, logKs)

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
ax.plot(
    temperatures,logKs, linewidth=2, label="Calculated"
)
ax.plot(
    temperatures, regressed_logKs, linewidth=2, label="Regressed"
)
ax.plot(
    temperatures, [magma_code_SiO2_l(t) for t in temperatures], linewidth=2, label="MAGMA Code"
)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("log K")
ax.grid()
ax.legend()

x1, x2, y1, y2 = ax.axis()
x_loc = x2 - (.4 * (x2 - x1))
y_loc = y2 - (0.2 * (y2 - y1))
y_loc2 = y2 - (0.25 * (y2 - y1))
ax.text(x_loc, y_loc, "Mine: y = {}/T + {}".format(round(slope, 2), round(intercept, 2)), fontweight="bold", fontsize=16)
ax.text(x_loc, y_loc2, "MAGMA: y = -94311.0/T + 22.13", fontweight="bold", fontsize=16)

plt.show()
