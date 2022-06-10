
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

products = ["Si_g", "O2_g"]
reactants = ["SiO2_l"]
standard_state_temp = 298.15
temperatures = np.arange(100, 4000 + 100, 100)

def magma_code_SiO2_l(temperature):
    return 22.13 - 94311.0 / temperature

def read_janaf_file(species, base_path="/Users/scotthull/Documents - Scott’s MacBook Pro/PhD Research/MAGMApy/K"):
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
    return [i for i in t if 2000 <= float(i) <= 3500]

def get_reaction_thermo(products, reactants, temperature, standard_state_temp):
    """
    Returns the thermo properties of a reaction.
    Products - Reacants for both delta H and delta S.
    :param products:
    :param reactants:
    :return:
    """
    R = 8.314  # kJ/mol-K
    product_sum_deltaH = 0
    reactant_sum_deltaH = 0
    product_sum_deltaS = 0
    reactant_sum_deltaS = 0
    for prod in products:
        if prod == "O2_g":
            print("here")
            product_sum_deltaH += 0.5 * float(get_standard_state(prod, standard_state_temp)["delta-f H"])
            product_sum_deltaS += 0.5 * float(get_standard_state(prod, standard_state_temp)["-[G-H(Tr)]/T"])
        else:
            product_sum_deltaH += float(get_standard_state(prod, standard_state_temp)["delta-f H"])
            product_sum_deltaS += float(get_standard_state(prod, standard_state_temp)["-[G-H(Tr)]/T"])
    for react in reactants:
        reactant_sum_deltaH += float(get_standard_state(react, standard_state_temp)["delta-f H"])
        reactant_sum_deltaS += float(get_standard_state(react, standard_state_temp)["-[G-H(Tr)]/T"])
    deltaH = product_sum_deltaH - reactant_sum_deltaH
    deltaS = product_sum_deltaS - reactant_sum_deltaS
    delta_G = deltaH - temperature * deltaS
    logK = -delta_G / (R * temperature) / 2.303  # lnK --> log10K to match JANAF
    return delta_G, deltaH, deltaS, logK


# temperatures = get_temperatures_from_janaf(products[0])
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
ax.plot(
    temperatures, [get_reaction_thermo(products, reactants, t, standard_state_temp)[3] for t in temperatures], linewidth=2, label="Calculated"
)
ax.plot(
    temperatures, [magma_code_SiO2_l(t) for t in temperatures], linewidth=2, label="MAGMA Code"
)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("log K")
ax.grid()
ax.legend()
plt.show()
