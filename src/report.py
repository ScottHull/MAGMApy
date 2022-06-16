import os
import shutil
import csv
import pandas as pd


class Report:

    def __init__(self, composition, liquid_system, gas_system, thermosystem, to_dir="reports"):
        self.composition = composition
        self.liquid_system = liquid_system
        self.gas_system = gas_system
        self.thermosystem = thermosystem
        self.to_dir = to_dir
        if os.path.exists(self.to_dir):
            shutil.rmtree(self.to_dir)
        os.mkdir(self.to_dir)

    def __make_subdirs(self, paths):
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)

    def __get_metadata(self):
        return "atomic fraction vaporized,{}\nmass liquid,{}\nmass fraction vaporized,{}\nliquid mass fraction,{}\ntemperature (K),{}\nfO2,{} {}\n".format(
            self.thermosystem.atomic_fraction_vaporized,
            self.liquid_system.initial_melt_mass - self.thermosystem.weight_vaporized,
            self.thermosystem.weight_fraction_vaporized,
            1 - self.thermosystem.weight_fraction_vaporized,
            self.liquid_system.temperature,
            self.gas_system.fO2_buffer,
            self.gas_system.oxygen_fugacity
        )

    def __make_report(self, path, iteration, data):
        outfile = open(path + "/{}.csv".format(iteration), 'w')
        outfile.write(self.__get_metadata())
        for i in data.keys():
            outfile.write("{},{}\n".format(i, data[i]))
        outfile.close()

    def create_composition_report(self, iteration):
        paths = [
            self.to_dir + "/cation_fraction",
            self.to_dir + "/oxide_fraction",
            self.to_dir + "/magma_composition"
        ]
        self.__make_subdirs(paths=paths)
        self.__make_report(path=paths[0], iteration=iteration, data=self.composition.cation_fraction)
        self.__make_report(path=paths[1], iteration=iteration, data=self.composition.oxide_mole_fraction)
        self.__make_report(path=paths[2], iteration=iteration, data=self.composition.planetary_abundances)

    def create_liquid_report(self, iteration):
        paths = [
            self.to_dir + "/activities",
            self.to_dir + "/activity_coefficients",
        ]
        self.__make_subdirs(paths=paths)
        self.__make_report(path=paths[0], iteration=iteration, data=self.liquid_system.activities)
        self.__make_report(path=paths[1], iteration=iteration, data=self.liquid_system.activity_coefficients)

    def create_gas_report(self, iteration):
        paths = [
            self.to_dir + "/partial_pressures",
            self.to_dir + "/atmosphere_total_mole_fraction",
            self.to_dir + "/atmosphere_mole_fraction",
        ]
        self.__make_subdirs(paths=paths)
        self.__make_report(path=paths[0], iteration=iteration, data=self.gas_system.partial_pressures)
        self.__make_report(path=paths[1], iteration=iteration, data=self.gas_system.total_mole_fraction)
        self.__make_report(path=paths[2], iteration=iteration, data=self.gas_system.mole_fractions)
