import os
import re
from math import isnan
import numpy as np
import pandas as pd
from copy import copy
from scipy.interpolate import interp1d

from src.plots import collect_data


def get_molecule_stoichiometry(molecule, return_oxygen=True, force_O2=False):
    """
    Requires that molecule be formatted correctly with capitalization, i.e. SiO2, not sio2.
    :param force_O2: Makes it so that it returns O2 instead of O if return_oxygen=True.
    :param return_oxygen:
    :param molecule:
    :return:
    """
    stoich = re.findall(r'([A-Z][a-z]*)(\d*)', molecule)
    d = {}
    for i in stoich:
        d.update({i[0]: i[1]})
        if d[i[0]] == '':
            d[i[0]] = 1
        d[i[0]] = int(d[i[0]])
        if force_O2:
            d.update({"O2": d["O"] / 2})
            del d["O"]
    if not return_oxygen:
        if "O" in d.keys():
            del d["O"]
        if "O2" in d.keys():
            del d["O2"]
    return d


def get_species_stoich_in_molecule(species, molecule):
    """
    Given an element, returns its stoichiometry in a molecule.
    i.e. Given element Na and molecule Na2O, would return integer 2.
    :param element:
    :param molecule:
    :param force_O2:
    :return:
    """
    d = 0
    is_O = False
    is_O2 = False
    if species == "O" or species == "O2":
        is_O = True
    if species == "O2":
        is_O2 = True
    stoich = get_molecule_stoichiometry(molecule=molecule)
    element_stoich = get_molecule_stoichiometry(molecule=species, return_oxygen=is_O, force_O2=False)
    for i in stoich.keys():
        if i in element_stoich.keys():
            d = stoich[i] / element_stoich[i]
    if molecule == "O2":
        d /= 2
    return d


def is_si(oxides):
    """
    Returns True if there is Si in the composition.  Else, return False.
    :return:
    """
    for i in oxides:
        stoich = get_molecule_stoichiometry(molecule=i)
        for j in stoich:
            if j.lower() == "si":
                return True
    return False


def get_species_with_element_appearance(element, species, return_ions=True):
    """
    Returns the species in which an element appears.
    i.e. if given Si, would return MgSiO3, Mg2SIO4, etc. with stoichiometry.
    :param return_ions: Will not return ion species if True
    :param element:
    :param species:
    :return:
    """
    d = {}
    for i in species:
        stoich = get_molecule_stoichiometry(molecule=i)
        if element in stoich.keys():  # if the element is in the species
            d.update({i: stoich[element]})  # return the species and the stoich of the element in the species
    if return_ions is False:  # delete ion species if we don't want them returned
        for i in list(d.keys()):
            if "-" in i or "+" in i:
                del d[i]
    return d


def get_element_in_base_oxide(element, oxides):
    """
    Given a list of base oxides, will return the first oxide that contains the species
    (i.e. will return SiO2 if given Si).
    :param element:
    :param oxides:
    :return:
    """
    if element == "Fe":  # don't want to return Fe2O3 since FeO is more fundamental
        return "FeO"
    for i in oxides:
        stoich = get_molecule_stoichiometry(molecule=i)
        if element in stoich.keys():
            return i
    return None


def get_list_of_cations(oxides):
    """
    Returns a list of all cations in the oxide system.
    :param oxides:
    :return:
    """
    cations = []
    for i in oxides:
        stoich = get_molecule_stoichiometry(molecule=i)
        for j in stoich:
            if j != "O":
                cations.append(j)
    return set(cations)


def normalize(composition):
    total = sum(composition.values())
    for i in composition.keys():
        composition[i] = composition[i] / total * 100.0
    return composition


def __get_isnan(r):
    try:
        if isnan(r):
            return True
        else:
            return False
    except:
        return False


def get_stoich_from_sheet(molecule, df):
    """
    Returns molecule stoichiometry from the master spreadsheet.
    :param molecule:
    :param df:
    :return:
    """
    d = {}
    reactants = df['Reactants'][molecule]
    if not __get_isnan(r=reactants):
        reactants = reactants.replace(" ", "").split(",")
        for i in reactants:
            stoich = i.split("*")
            species, stoich = stoich[1], float(stoich[0])
            d.update({species: stoich})
    return d


def interpolate_composition_at_vmf(run, vmf, subdir):
    """
    Given a VMF, reads all vapor outputs, gets VMF and composition, gets nearest VMF, and then interpolates the
    composition based on the nearest VMF.
    :param vapor_path:
    :param vmf:
    :return:
    """
    # read in all vapor outputs
    d = collect_data(path=f"{run}/{subdir}", x_header='mass fraction vaporized')
    d = {i: normalize(d[i]) for i in d.keys()}  # normalize compositions
    vmfs = d.keys()  # get all VMFs
    species = d[list(vmfs)[0]].keys()  # get species from first vmf
    interpolated_composition = {}
    for i in species:
        # interpolate each species based on the given VMF
        interpolated_composition.update({i: float(interp1d(x=list(vmfs), y=[d[j][i] for j in vmfs])(vmf / 100.0))})
    return interpolated_composition


def get_molecular_mass(molecule: str):
    """
    Returns the number of moles of the given composition in weight percent.
    """
    # read in the period table
    pt = pd.read_csv("data/periodic_table.csv", index_col='element')
    # get the stoichiometry of the molecule
    stoich = re.findall(r'([A-Z][a-z]*)(\d*)', molecule)
    # get the sum of the masses of each atom in the molecule
    moles = 0
    for atom, count in stoich:
        if len(count) == 0:
            count = 1
        moles += int(count) * pt.loc[atom, 'atomic_mass']
    return moles


def get_mean_molecular_mass(composition: dict):
    """
    Returns the mean molecular mass of the given composition in weight percent.
    """
    # remove the gas label
    cleaned_composition = {key.split("_")[0].replace("+", "").replace("-", ""): value
                           for key, value in composition.items()}
    # remove species with 0 weight percent
    cleaned_composition = {key: value for key, value in cleaned_composition.items() if value != 0}
    # get the sum of the masses in the composition dict
    total_mass = sum(cleaned_composition.values())
    # get the number of moles in the composition dict
    total_moles = sum(
        [(1 / get_molecular_mass(molecule)) * cleaned_composition[molecule] if get_molecular_mass(molecule) != 0 else 0
         for molecule in cleaned_composition.keys()])
    return total_mass / total_moles


def mole_fraction_to_weight_percent(mole_fraction: dict):
    """
    Takes in a dictionary of species in mole fraction and converts to weight percent.
    :param mole_fraction:
    :return:
    """
    weight_percent = {}
    for i in mole_fraction.keys():
        weight_percent.update({i: mole_fraction[i] * get_molecular_mass(i.split("_")[0])})
    return normalize(weight_percent)


class ConvertComposition:

    def __init__(self):
        self.avogadro = 6.02214076 * 10 ** 23
        self.atomic_masses = pd.read_csv("data/periodic_table.csv", index_col="element")

    def get_atomic_mass(self, element):
        return self.atomic_masses['atomic_mass'][element.capitalize()]

    def get_molecule_mass(self, molecule):
        stoich = get_molecule_stoichiometry(molecule=molecule)
        mass = 0
        for i in stoich.keys():
            mass += stoich[i] * self.get_atomic_mass(element=i)  # i.e. mass SiO2 = (1 * Si) + (2 * O)
        return mass

    def weight_pct_to_moles(self, wt_pct_composition):
        moles = {}
        for i in wt_pct_composition:
            molecular_wt = self.get_molecule_mass(molecule=i)
            moles.update({i: wt_pct_composition[i] / molecular_wt})  # total moles of oxide
        return moles

    def mole_pct_to_atoms(self, moles_composition):
        """
        We normalize to Si = 10^6.  If no Si, then we do not normalize.
        This corresponds to lines 186-204 in MAGMA during for AE variables.
        :param moles_composition:
        :param mole_pct_composition:
        :return:
        """
        IS_SI = is_si(oxides=moles_composition.keys())  # if there is Si in the composition
        cations = get_list_of_cations(oxides=moles_composition.keys())  # get a list of all cations in solution
        si_normalize = 10 ** 6
        atoms = {}
        for i in cations:  # build the dictionary beforehand to avoid key errors
            atoms.update({i: 0})
        for i in moles_composition.keys():
            if i != "Fe2O3":  # want all Fe to be in FeO
                stoich = get_molecule_stoichiometry(molecule=i)
                if IS_SI:
                    if i.lower() == "sio2":
                        atoms.update({"Si": si_normalize})  # we want abundance of Si to be normalized to 10^6
                    else:
                        for j in stoich:
                            if j != "O":
                                cation_abundance = (moles_composition[i] * stoich[j])
                                if j == "Fe":
                                    cation_abundance = (moles_composition[i] * stoich[j]) + (
                                        (moles_composition["Fe2O3"] * 2))
                                atoms[j] += cation_abundance * si_normalize / moles_composition["SiO2"]
                else:
                    for j in stoich:
                        if j != "O":
                            atoms[j] += (moles_composition[i] * stoich[j]) * self.avogadro
        return atoms

    def get_molecular_abundance_si_normalized(self, composition, molecules):
        """
        Returns the molecular abundance of all molecules (e.g. SiO2) normalzied to Si = 10^6.
        If no Si, then this is simply the unnormalized molecular abundance.
        :return:
        """
        # TODO: Make an function that returns molecules that excludes Fe2O3
        total = 0
        for i in molecules:
            if i != "Fe2O3":  # want all Fe as FeO
                stoich = get_molecule_stoichiometry(molecule=i)
                for j in stoich:
                    if j != "O":
                        total += composition[j] / stoich[j]
        return total

    def cations_to_oxides(self, cations: dict, oxides: list):
        """
        Takes absolute cation molar abundances, finds their corresponding oxide, and returns the oxide molar abundance.
        :param cations:
        :return:
        """
        oxides_moles = {i: 0 for i in oxides}
        for c in cations:
            abundance = cations[c]  # get the molar abundance of the cation
            if c not in ["O", "O2"]:  # skip oxygen and oxygen dioxide if it's included in the cations list
                corresponding_oxide = get_element_in_base_oxide(element=c, oxides=oxides)
                oxide_stoich = get_molecule_stoichiometry(molecule=corresponding_oxide)  # get the molecule stoich
                element_stoich_in_molecule = oxide_stoich[c]  # get the stoich of the element in the molecule
                oxides_moles[
                    corresponding_oxide] += abundance / element_stoich_in_molecule  # add the moles of the cation
        return oxides_moles

    def cations_mass_to_oxides_weight_percent(self, cations: dict, oxides: list):
        """
        Takes absolute cation mass abundances, finds their corresponding oxide, and returns the oxide molar abundance.
        :param cations:
        :return:
        :param cations:
        :param oxides:
        :return:
        """
        # convert absolute mass of each cation to moles
        cations_moles = {i: cations[i] / self.get_atomic_mass(element=i) for i in cations.keys()}  # g --> moles
        # convert moles of each cation to moles of each oxide
        oxides_moles = self.cations_to_oxides(cations=cations_moles, oxides=oxides)
        # convert moles of each oxide to weight percent
        oxides_weight_percent = {i: oxides_moles[i] / sum(oxides_moles.values()) * 100.0 for i in oxides_moles}
        return oxides_weight_percent

    def moles_to_mass(self, composition: dict):
        """
        Returns a dictionary of the absolute masses of the given molar compositions.
        :param composition: The molar composition to be converted to absolute mass.
        :return:
        """
        return {i: self.get_molecule_mass(molecule=i) * composition[i] for i in composition}  # mol * g/mol = g

    def mass_to_mass_percent(self, composition: dict):
        """
        Returns a dictionary of the mass percentages of the given absolute masses.
        :param composition: The absolute mass composition to be converted to mass percentages.
        :return:
        """
        return {i: composition[i] / sum(composition.values()) for i in composition}

    def mass_to_moles(self, composition: dict):
        """
        Returns a dictionary of the molar composition of the given absolute masses.
        :param composition: The absolute mass composition to be converted to molar composition.
        mol = g / g/mol
        :return:
        """
        return {i: composition[i] / self.get_molecule_mass(molecule=i) for i in composition}

    def oxide_to_cations(self, composition: dict):
        """
        Returns a dictionary of the cation composition of the given oxide composition.
        :param composition: The oxide composition to be converted to cation composition.
        :return:
        """
        cations = {}
        for oxide in composition:
            stoich = get_molecule_stoichiometry(molecule=oxide)
            for cation in stoich:
                if cation not in cations and cation != "O":  # do not include O
                    cations[cation] = 0
                if cation != "O":
                    cations[cation] += stoich[cation] * composition[oxide]
        return cations


class Composition(ConvertComposition):
    """
    This class is used to track the composition of the liquid and the system.
    """

    def __init__(self, composition):
        super().__init__()  # initialize base class so we can use its functions
        self.wt_pct_composition = normalize(
            composition=composition)  # the user inputs oxide weights, we normalize to wt%
        self.moles_composition = self.weight_pct_to_moles(
            wt_pct_composition=self.wt_pct_composition)  # absolute moles of composition
        self.mole_pct_composition = normalize(composition=self.moles_composition)  # mole% of composition
        #  NOTE THAT FOR ALL COMPOSITIONS BELOW, EVERYTHING IS NORMALIZED TO SI = 10^6 IF SI IS IN THE SYSTEM
        self.atoms_composition = self.mole_pct_to_atoms(
            moles_composition=self.mole_pct_composition)  # absolute atoms of composition
        self.total_atoms = sum(
            self.atoms_composition.values())  # total atoms of cations in composition.  if Si is present, this is normalized to Si
        self.total_atoms_oxide = self.get_molecular_abundance_si_normalized(composition=self.atoms_composition,
                                                                            molecules=self.moles_composition.keys())  # total molecules of oxides in composition
        self.oxide_mole_fraction = self.get_molecule_fraction()  # base oxide fraction
        self.cation_fraction = self.get_cation_fraction()  # cation elemental fraction
        self.liquid_abundances = self.__initial_liquid_abundances()  # cation elemental fraction
        self.initial_liquid_number_of_moles = copy(self.liquid_abundances)  # cation elemental fraction

    def get_molecule_fraction(self):
        """
        Returns oxide mole fraction.
        e.g. SiO2 / (SiO2 + MgO + FeO + ...)
        :return:
        """
        fraction = {}
        for i in self.moles_composition:
            stoich = get_molecule_stoichiometry(molecule=i)
            for j in stoich.keys():
                if j != "O" and i != "Fe2O3":
                    fraction.update({i: (1.0 / stoich[j]) * self.atoms_composition[j] / self.total_atoms_oxide})
        self.oxide_mole_fraction = fraction
        return fraction

    def get_cation_fraction(self):
        """
        Returns cation fraction relative to total atoms,
        i.e. Si / (Si + Mg + Fe + ...)
        :return:
        """
        total_cations = sum(self.atoms_composition.values())
        fraction = {}
        for i in self.atoms_composition:
            fraction.update({i: self.atoms_composition[i] / total_cations})
        self.cation_fraction = fraction
        return fraction

    def get_major_oxides_by_abundance(self):
        """
        Returns a list of cations sorted from most to least abundant.
        :return:
        """
        rank = []
        for i in self.oxide_mole_fraction:
            abun = self.oxide_mole_fraction[i]
            if len(rank) == 0:
                rank.append((i, abun))
            else:
                for index, j in enumerate(rank):
                    if abun > j[1]:
                        rank.insert(index, (i, abun))
                        break
                    elif index + 1 == len(rank):
                        rank.append((i, abun))
                        break
        return rank

    def __initial_liquid_abundances(self):
        """
        liquid elemental abundances, where we initially assume that the liquid abundances is equal to the
        full system input composition.
        Assume that initially, the liquid abundances are just equal to the elemental abundances
        (i.e. cation fraction).
        :return:
        """
        abundances = {}
        for i in self.cation_fraction:
            abundances.update({i: self.atoms_composition[i]})
        total_liquid_cations = sum(abundances.values())  # sum of fractional cation abundances
        self.initial_liquid_cations = total_liquid_cations  # the initial liquid cation abundance
        # take the ratio of current liquid cations to initial cations, initially will be 1
        self.liquid_cation_ratio = self.initial_liquid_cations / total_liquid_cations
        return abundances
