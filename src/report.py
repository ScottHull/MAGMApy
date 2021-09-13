import os
import shutil
import pandas as pd

class Report:

    def __init__(self, composition, liquid_system, gas_system, to_dir="reports"):
        self.composition = composition
        self.liquid_system = liquid_system
        self.gas_system = gas_system
        self.to_dir = to_dir
        if os.path.exists(self.to_dir):
            shutil.rmtree(self.to_dir)
        os.mkdir(self.to_dir)

    def create_composition_report(self, iteration):
        df_cation_fraction = pd.DataFrame(self.composition.cation_fraction)
        df_oxide_fraction = pd.DataFrame(self.composition.oxide_mole_fraction)
        df_planetary_abundances = pd.DataFrame(self.composition.planetary_abundances)

    def create_liquid_report(self, iteration):
        df_activity_coefficients = pd.DataFrame(self.liquid_system.activity_coefficients)
        df_activities = pd.DataFrame(self.liquid_system.activities)

    def create_gas_report(self, iteration):
        df_gas_pressures = pd.DataFrame({**self.gas_system.partial_pressure_major_species,
                                         **self.gas_system.partial_pressure_minor_species})
