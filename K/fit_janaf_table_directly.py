import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

base_path = "/Users/scotthull/Documents - Scott’s MacBook Pro/PhD Research/MAGMApy/K"
species = "SiO2_l"

def read_janaf_file(species):
    path = base_path + "/" + species + ".dat"
    df = pd.read_csv(path, sep="\t", skiprows=1, index_col="T(K)")
    return df

def linear_regression(x, y):
    """
    y = mx + b
    :param x:
    :param y:
    :return:
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept

df = read_janaf_file(species)
T, K = df.index, df["log Kf"]

slope, intercept = linear_regression(T, K)

print("y = {}x + {}".format(slope, intercept))


