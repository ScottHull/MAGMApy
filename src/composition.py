import os
import re
import numpy as np
import pandas as pd
from copy import copy
from scipy.interpolate import interp1d

# Define the period table path as in the data folder of the project directory
PERIODIC_TABLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data/periodic_table.csv')
# Load periodic table data
PERIODIC_TABLE = pd.read_csv(PERIODIC_TABLE_PATH, index_col='element')

class AtomOrMolecule:

    def __init__(self, periodic_symbol: str):
        self.periodic_table = PERIODIC_TABLE
        self.formula = periodic_symbol
        self.stoichiometry = self.__get_stoichiometry()
        self.atomic_mass = self.__lookup_atomic_mass()
        self.moles = 0.0  # Total moles existing in the mixture

    def __lookup_atomic_mass(self) -> float:
        # If stoichiometry is empty, raise error
        if not self.stoichiometry:
            raise ValueError("Invalid molecular formula: {}".format(self.formula))
        for (element, count) in self.stoichiometry.items():
            if element not in self.periodic_table.index:
                raise ValueError("Element {} not found in periodic table.".format(element))
        atomic_mass = sum(self.periodic_table.loc[element, 'atomic_mass'] * count for (element, count) in self.stoichiometry.items())
        # If the atomic mass is negative or not a number, raise error
        if atomic_mass <= 0 or np.isnan(atomic_mass):
            raise ValueError("Invalid atomic mass for formula: {}".format(self.formula))
        return atomic_mass

    def __get_stoichiometry(self) -> dict:
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, self.formula)
        stoichiometry = {}
        for (component, count) in matches:
            stoichiometry[component] = int(count) if count else 1
        # If stoich is empty, raise error
        if not stoichiometry:
            raise ValueError("Invalid molecular formula: {}".format(self.formula))
        return stoichiometry

    def get_elemental_moles(self) -> dict:
        return {element: count * self.moles for (element, count) in self.stoichiometry.items()}


class Mixture:

    def __init__(self, composition: dict):
        self.initial_composition = copy(composition)  # Raw composition input as wt%
        self.__initial_mass = 0.0  # Initial total mass of the mixture
        self.__composition = {}  # Current composition as moles
        # Call the setup method to initialize the mixture
        self.__setup()

    def get_composition(self) -> dict:
        return {str(k.formula): v for (k, v) in self.__composition.items()}

    def get_total_mass(self):
        return self.__total_mass

    def get_moles(self):
        pass


    def __normalize(self, composition: dict) -> dict:
        total = sum(composition.values())
        if total == 0:
            raise ValueError("Total composition cannot be zero.")
        return {k: v / total for (k, v) in composition.items()}

    def __setup(self) -> None:
        # Convert the raw input composition into a structured format
        # If the initial composition is empty, raise error
        if not self.initial_composition:
            raise ValueError("Composition cannot be empty.")
        # If there are duplicate components, raise error
        if len(self.initial_composition) != len(set(self.initial_composition.keys())):
            raise ValueError("Duplicate components found in composition. Please ensure all components are unique.")
        # Normalize the composition
        normalized_input = self.__normalize(self.initial_composition)
        for (component, weight_fraction) in normalized_input.items():
            atom_or_molecule = AtomOrMolecule(component)
            self.__composition[atom_or_molecule] = weight_fraction
        self.__initial_mass = sum(self.__composition.values())
