import os
from math import log
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt


products = ["Zn_g", "O2_g"]
reactants = ["ZnO_l"]
standard_state_temp = 298.15
temperatures = np.arange(100, 4000 + 100, 100)
# base_path = "/Users/scotthull/Documents - Scott’s MacBook Pro/PhD Research/MAGMApy/K"
base_path = r"C:\Users\Scott\PycharmProjects\MAGMApy\K"

# Jak et al. 1997
# https://link.springer.com/content/pdf/10.1007/s11663-997-0055-x.pdf
A = -309524
B = 47.936

a = 60.668
b = 0
c = 0
d = 0
e = 0

def tian_2019_model(T):
    return 10.596 - (24424 / T)

def calculate_cp(T):
    return a + b * T + c * T ** -2 + d * T ** -3 + e * T ** -0.5

def calculate_S0(T):
    """
    S = B + int_298.15^T (cp / T) dt
    therefore
    S = B + cp(ln(T) - ln(298.15))
    :param T:
    :return:
    """
    return B + calculate_cp(T) * (log(T) - log(298.15))

def calculate_H0(T):
    """
    H = A + int_298.15^T cp dT
    """
    return A + calculate_cp(T) * (T - 298.15)

def read_janaf_file(species):
    # path = base_path + "/" + species + ".dat"
    path = os.path.join(base_path, species + ".dat")
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
        if prod == "O2_g":
            product_sum_deltaH += 0.5 * float(get_standard_state(prod, standard_state_temp)["delta-f H"])
            product_sum_deltaS += 0.5 * float(get_standard_state(prod, standard_state_temp)["-[G-H(Tr)]/T"])
        else:
            product_sum_deltaH += float(get_standard_state(prod, standard_state_temp)["delta-f H"])
            product_sum_deltaS += float(get_standard_state(prod, standard_state_temp)["-[G-H(Tr)]/T"])
    for react in reactants:
        reactant_sum_deltaH += calculate_H0(standard_state_temp)
        reactant_sum_deltaS += calculate_S0(standard_state_temp)
    deltaH = ((product_sum_deltaH * 1000) - reactant_sum_deltaH)  # S in in J, and H is in kJ
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

logKs = [get_reaction_thermo(products, reactants, t, standard_state_temp)[3] for t in temperatures]
deltaGs = [get_reaction_thermo(products, reactants, t, standard_state_temp)[0] / 1000 for t in temperatures]
regressed_logKs = [regress_temperature_against_logK(temperatures, logKs, t) for t in temperatures]
slope, intercept = linear_regression(1 / temperatures, logKs)

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
ax.plot(
    temperatures, logKs, linewidth=2, label="Calculated"
)
ax.plot(
    temperatures, regressed_logKs, linewidth=2, label="Regressed"
)
ax.plot(
    temperatures, [tian_2019_model(T) for T in temperatures], linewidth=2, label="Tian et al. 2019"
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
ax.text(x_loc, y_loc2, "Tian et al. 2019: y = -24454/T + 10.596", fontweight="bold", fontsize=16)

plt.show()

# fig = plt.figure(figsize=(16, 9))
# ax = fig.add_subplot(111)
# ax.plot(
#     temperatures, deltaGs, linewidth=2, label="Calculated"
# )
# ax.set_xlabel("Temperature (K)")
# ax.set_ylabel("delta G (kJ/mol)")
# ax.grid()
# ax.legend()
# ax.set_xlim(1500, 1900)
# ax.set_ylim(0, 200)
#
# plt.show()
