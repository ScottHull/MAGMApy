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
        return "atomic fraction vaporized,{}\nmass liquid,{}\nmass fraction vaporized,{}\nliquid mass fraction,{}\n" \
               "initial liquid mass,{}\nmass vapor,{}\ntemperature (K),{}\nfO2,{} {}\nmost volatile species,{}\n".format(
            self.thermosystem.atomic_fraction_vaporized,
            self.liquid_system.initial_melt_mass - self.thermosystem.weight_vaporized,
            self.thermosystem.weight_fraction_vaporized,
            1 - self.thermosystem.weight_fraction_vaporized,
            self.liquid_system.initial_melt_mass,
            self.gas_system.vapor_mass,
            self.liquid_system.temperature,
            self.gas_system.fO2_buffer,
            self.gas_system.oxygen_fugacity,
            self.thermosystem.most_volatile_species
        )

    def __make_report(self, path, iteration, data, round_digits=None):
        outfile = open(path + "/{}.csv".format(iteration), 'w')
        outfile.write(self.__get_metadata())
        for i in data.keys():
            if isinstance(data[i], float):  # round to 4 decimal places if float
                if round_digits is not None:
                    outfile.write("{},{}\n".format(i, round(data[i], round_digits)))
                else:
                    outfile.write("{},{}\n".format(i, data[i]))
            else:
                outfile.write("{},{}\n".format(i, data[i]))
        outfile.close()

    def create_composition_report(self, iteration):
        paths = [
            self.to_dir + "/cation_fraction",
            self.to_dir + "/oxide_fraction",
            self.to_dir + "/magma_composition",
        ]
        self.__make_subdirs(paths=paths)
        self.__make_report(path=paths[0], iteration=iteration, data=self.composition.cation_fraction)
        self.__make_report(path=paths[1], iteration=iteration, data=self.composition.oxide_mole_fraction)
        self.__make_report(path=paths[2], iteration=iteration, data=self.composition.liquid_abundances)

    def create_liquid_report(self, iteration):
        paths = [
            self.to_dir + "/activities",
            self.to_dir + "/activity_coefficients",
            self.to_dir + "/magma_cation_mass_fraction",
            self.to_dir + "/magma_oxide_mass_fraction",
            self.to_dir + "/magma_element_mass",
            self.to_dir + "/magma_oxide_mole_fraction",
        ]
        self.__make_subdirs(paths=paths)
        self.__make_report(path=paths[0], iteration=iteration, data=self.liquid_system.activities)
        self.__make_report(path=paths[1], iteration=iteration, data=self.liquid_system.activity_coefficients)
        self.__make_report(path=paths[2], iteration=iteration, data=self.liquid_system.cation_mass_fraction)
        self.__make_report(path=paths[3], iteration=iteration, data=self.liquid_system.liquid_oxide_mass_fraction)
        self.__make_report(path=paths[4], iteration=iteration, data=self.liquid_system.cation_mass)
        self.__make_report(path=paths[5], iteration=iteration, data=self.liquid_system.liquid_oxide_mole_fraction)

    def create_gas_report(self, iteration):
        paths = [
            self.to_dir + "/partial_pressures",
            self.to_dir + "/atmosphere_total_mole_fraction",
            self.to_dir + "/atmosphere_mole_fraction",
            self.to_dir + "/atmosphere_cation_moles",
            self.to_dir + "/atmosphere_cation_mass_fraction",
            self.to_dir + "/f",
            self.to_dir + "/total_vapor_element_mass",
            self.to_dir + "/total_vapor_species_mass",
        ]
        self.__make_subdirs(paths=paths)
        self.__make_report(path=paths[0], iteration=iteration, data=self.gas_system.partial_pressures)
        self.__make_report(path=paths[1], iteration=iteration, data=self.gas_system.total_mole_fraction)
        self.__make_report(path=paths[2], iteration=iteration, data=self.gas_system.mole_fractions)
        self.__make_report(path=paths[3], iteration=iteration, data=self.gas_system.cation_moles)
        self.__make_report(path=paths[4], iteration=iteration, data=self.gas_system.element_mass_fraction)
        self.__make_report(path=paths[5], iteration=iteration, data=self.gas_system.f)
        self.__make_report(path=paths[6], iteration=iteration, data=self.gas_system.element_total_mass)
        self.__make_report(path=paths[7], iteration=iteration, data=self.gas_system.species_total_mass)
