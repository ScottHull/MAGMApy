import re
import pandas as pd


def get_molecule_stoichiometry(molecule, return_oxygen=True):
    """
    Requires that molecule be formatted correctly with capitalization, i.e. SiO2, not sio2.
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
    if not return_oxygen:
        if "O" in d.keys():
            del d["O"]
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


def get_species_with_element_appearance(element, species):
    """
    Returns the species in which an element appears.
    i.e. if given Si, would return MgSiO3, Mg2SIO4, etc. with stoichiometry.
    :param element:
    :param species:
    :return:
    """
    d = {}
    for i in species:
        stoich = get_molecule_stoichiometry(molecule=i)
        if element in stoich.keys():  # if the element is in the species
            d.update({i: stoich[element]})  # return the species and the stoich of the element in the species
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


class Composition(ConvertComposition):

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
        self.planetary_abundances = self.__initial_planetary_abundances()  # cation elemental fraction

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

    def __initial_planetary_abundances(self):
        """
        Planetary elemental abundances, where we initially assume that the planetary abundances is equal to the
        full system input composition.
        Assume that initially, the planetary abundances are just equal to the elemental abundances
        (i.e. cation fraction).
        :return:
        """
        abundances = {}
        for i in self.cation_fraction:
            abundances.update({i: self.cation_fraction[i]})
        total_planetary_cations = sum(abundances.values())  # sum of fractional cation abundances
        self.initial_planetary_cations = total_planetary_cations  # the initial planetary cation abundance
        # take the ratio of current planetary cations to initial cations, initially will be 1
        self.planetary_cation_ratio = self.initial_planetary_cations / total_planetary_cations
        return abundances
