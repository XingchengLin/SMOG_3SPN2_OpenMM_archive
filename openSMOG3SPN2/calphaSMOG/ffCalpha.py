import pandas as pd
import simtk.openmm
import os
import sys
import shutil
import numpy as np
import mdtraj
from Bio.PDB.Polypeptide import three_to_one

__location__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__location__)
import shadow_map

__location__ = os.path.dirname(os.path.abspath(__file__))
__author__ = 'Xingcheng Lin adapted from Carlos Bueno'

_PROTEIN_RESIDUES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                    'SER', 'THR', 'TRP', 'TYR', 'VAL']

xml = f'{__location__}/awsem.xml'

################################################

def my_lt_range(start, end, step):
    while start < end:
        yield start
        start += step


def my_le_range(start, end, step):
    while start <= end:
        yield start
        start += step
###########################################


def parsePDB(pdb_file):
    '''Reads a pdb file and outputs a pandas DataFrame'''
    def pdb_line(line):
        return dict(recname=str(line[0:6]).strip(),
                    serial=int(line[6:11]),
                    name=str(line[12:16]).strip(),
                    altLoc=str(line[16:17]),
                    resname=str(line[17:20]).strip(),
                    chainID=str(line[21:22]),
                    resSeq=int(line[22:26]),
                    iCode=str(line[26:27]),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                    occupancy=0.0 if line[54:60].strip() == '' else float(line[54:60]),
                    tempFactor=0.0 if line[60:66].strip() == '' else float(line[60:66]),
                    element=str(line[76:78]),
                    charge=str(line[78:80]))

    with open(pdb_file, 'r') as pdb:
        lines = []
        for line in pdb:
            if len(line) > 6 and line[:6] in ['ATOM  ', 'HETATM']:
                lines += [pdb_line(line)]
    pdb_atoms = pd.DataFrame(lines)
    pdb_atoms = pdb_atoms[['recname', 'serial', 'name', 'altLoc',
                           'resname', 'chainID', 'resSeq', 'iCode',
                           'x', 'y', 'z', 'occupancy', 'tempFactor',
                           'element', 'charge']]
    return pdb_atoms
 
def writePDB_protein(atoms, pdb_file, write_TER=False):
    '''Reads a pandas DataFrame of atoms and outputs a pdb file'''
    protein_atoms = atoms[atoms.resname.isin(_PROTEIN_RESIDUES)].copy()
    chainID = None
    with open(pdb_file, 'w') as pdb:
        for i, atom in protein_atoms.iterrows():
            if chainID is not None:
                if write_TER and (atom['chainID'] != chainID):
                    pdb.write('TER\n')
            chainID = atom['chainID']
            pdb_line = f'{atom.recname:<6}{atom.serial:>5} {atom["name"]:^4}{atom.altLoc:1}'+\
                       f'{atom.resname:<3} {atom.chainID:1}{atom.resSeq:>4}{atom.iCode:1}   '+\
                       f'{atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}' +\
                       f'{atom.occupancy:>6.2f}{atom.tempFactor:>6.2f}'+' ' * 10 +\
                       f'{atom.element:>2}{atom.charge:>2}'
            assert len(pdb_line) == 80, f'An item in the atom table is longer than expected ({len(pdb_line)})\n{pdb_line}'
            pdb.write(pdb_line + '\n')
        # Put an "END" to the end of the file, required by SMOG2
        pdb.write('END\n')



def writePDB(atoms, pdb_file, write_TER=False):
    '''Reads a pandas DataFrame of atoms and outputs a pdb file'''
    chainID = None
    with open(pdb_file, 'w') as pdb:
        for i, atom in atoms.iterrows():
            if chainID is not None:
                if write_TER and (atom['chainID'] != chainID):
                    pdb.write('TER\n')
            chainID = atom['chainID']
            pdb_line = f'{atom.recname:<6}{atom.serial:>5} {atom["name"]:^4}{atom.altLoc:1}'+\
                       f'{atom.resname:<3} {atom.chainID:1}{atom.resSeq:>4}{atom.iCode:1}   '+\
                       f'{atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}' +\
                       f'{atom.occupancy:>6.2f}{atom.tempFactor:>6.2f}'+' ' * 10 +\
                       f'{atom.element:>2}{atom.charge:>2}'
            assert len(pdb_line) == 80, f'An item in the atom table is longer than expected ({len(pdb_line)})\n{pdb_line}'
            pdb.write(pdb_line + '\n')
        # write END
        pdb.write('END\n')


def parseConfigTable(config_section):
    """Parses a section of the configuration file as a table"""

    def readData(config_section, a):
        """Filters comments and returns values as a list"""
        temp = config_section.get(a).split('#')[0].split()
        l = []
        for val in temp:
            val = val.strip()
            try:
                x = int(val)
                l += [x]
            except ValueError:
                try:
                    y = float(val)
                    l += [y]
                except ValueError:
                    l += [val]
        return l

    data = []
    for a in config_section:
        if a == 'name':
            columns = readData(config_section, a)
        elif len(a) > 3 and a[:3] == 'row':
            data += [readData(config_section, a)]
        else:
            print(f'Unexpected row {readData(config_section, a)}')
    return pd.DataFrame(data, columns=columns)


def copy_parameter_files():
    src = f"{__location__}/parameters"
    dest = '.'
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)

def save_protein_sequence(Coarse, sequence_file='protein.seq'):
    """Saves protein sequence to a file from table"""
    protein_data = Coarse[Coarse.resname.isin(_PROTEIN_RESIDUES)].copy()
    resix = (protein_data.chainID + '_' + protein_data.resSeq.astype(str))
    res_unique = resix.unique()
    protein_data['resID'] = resix.replace(dict(zip(res_unique, range(len(res_unique)))))
    protein_sequence=[r.iloc[0]['real_resname'] for i, r in protein_data.groupby('resID')]
    protein_sequence_one = [three_to_one(a) for a in protein_sequence]

    with open(sequence_file, 'w') as ps:
        ps.write(''.join(protein_sequence_one))
        ps.write('\n')

def get_protein_sequence(Coarse):
    protein_data = Coarse[Coarse.resname.isin(_PROTEIN_RESIDUES)].copy()
    resix = (protein_data.chainID + '_' + protein_data.resSeq.astype(str))
    res_unique = resix.unique()
    protein_data['resID'] = resix.replace(dict(zip(res_unique, range(len(res_unique)))))
    protein_sequence=[r.iloc[0]['real_resname'] for i, r in protein_data.groupby('resID')]
    protein_sequence_one = ''.join([three_to_one(a) for a in protein_sequence])
    return protein_sequence_one


class BaseError(Exception):
    pass


class SMOGExclusionParser(object):
    # define some general methods that can be inherited by multiple classes
    def parse_smog_exclusions(self, exclude12=True, exclude13=True, exclude14=True, exclude_native_pairs=True):
        exclusions = []
        if exclude12 and hasattr(self, 'smog_bonds'):
            for i, row in self.smog_bonds.iterrows():
                exclusions.append((int(row['aai']), int(row['aaj'])))
        if exclude13 and hasattr(self, 'smog_angles'):
            for i, row in self.smog_angles.iterrows():
                exclusions.append((int(row['aai']), int(row['aak'])))
        if exclude14 and hasattr(self, 'smog_dihedrals'):
            for i, row in self.smog_dihedrals.iterrows():
                exclusions.append((int(row['aai']), int(row['aal'])))
        if exclude_native_pairs and hasattr(self, 'smog_native_pairs'):
            for i, row in self.smog_native_pairs.iterrows():
                exclusions.append((int(row['aai']), int(row['aaj'])))
        # correct order
        for i in range(len(exclusions)):
            aai, aaj = int(exclusions[i][0]), int(exclusions[i][1])
            if aai > aaj:
                aai, aaj = aaj, aai
            exclusions[i] = (aai, aaj)
        # remove duplicates
        exclusions = sorted(list(set(exclusions)))
        exclusions = np.array([[int(x[0]), int(x[1])] for x in exclusions])
        df_exclusions = pd.DataFrame(exclusions, columns=['aai', 'aaj'])
        # set smog exclusions
        if hasattr(self, 'smog_exclusions'):
            if self.smog_exclusions is not None:
                print('Replace smog exclusions with the new one')
        self.smog_exclusions = df_exclusions
        return None


class SMOGProteinParser(SMOGExclusionParser):
    # class that can use our shadow algorithm to parse protein
    def __init__(self, atomistic_pdb, ca_pdb, smog_energy_scale=2.5):
        self.pdb = ca_pdb
        if atomistic_pdb is None:
            print('No input all-atom pdb file')
        else:
            # do coarse-graining
            self.atomistic_pdb = atomistic_pdb
            self.atomistic_atoms = parsePDB(self.atomistic_pdb)
            flag = ((self.atomistic_atoms['name'] == 'CA') & (self.atomistic_atoms['resname'].isin(_PROTEIN_RESIDUES)))
            self.atoms = self.atomistic_atoms.loc[flag].copy()
            n_atoms = len(self.atoms.index)
            self.atoms.index = list(range(n_atoms))
            self.atoms.serial = list(range(1, n_atoms + 1))
            writePDB(self.atoms, self.pdb)
        print(f'Set smog bonded energy scale as {smog_energy_scale}')
        self.smog_energy_scale = smog_energy_scale
    
    def parse_atomistic_pdb(self, frame=0, get_native_pairs=True, radius=0.1, bonded_radius=0.05, cutoff=0.6, box=None, 
                            pbc=False, get_exclusions=True, exclude12=True, exclude13=True, exclude14=True, 
                            exclude_native_pairs=True):
        # parse configuration to get bonded interactions and get native pairs with our shadow algorithm
        traj = mdtraj.load_pdb(self.pdb) # load CA atom pdb file
        bonds, angles, dihedrals = [], [], []
        n_atoms = len(self.atoms.index)
        self.atoms.index = list(range(len(self.atoms.index)))
        for atom1 in range(n_atoms):
            chain1 = self.atoms.loc[atom1, 'chainID']
            if atom1 < n_atoms - 1:
                atom2 = atom1 + 1
                chain2 = self.atoms.loc[atom2, 'chainID']
                if chain1 == chain2:
                    bonds.append([atom1, atom2])
            if atom1 < n_atoms - 2:
                atom3 = atom1 + 2
                chain3 = self.atoms.loc[atom3, 'chainID']
                if (chain1 == chain2) and (chain1 == chain3):
                    angles.append([atom1, atom2, atom3])
            if atom1 < n_atoms - 3:
                atom4 = atom1 + 3
                chain4 = self.atoms.loc[atom4, 'chainID']
                if (chain1 == chain2) and (chain1 == chain3) and (chain1 == chain4):
                    dihedrals.append([atom1, atom2, atom3, atom4])
        bonds = np.array(bonds)
        df_bonds = pd.DataFrame(bonds, columns=['aai', 'aaj'])
        df_bonds['r0 (nm)'] = mdtraj.compute_distances(traj, bonds, periodic=pbc)[frame]
        df_bonds['Kb'] = [20000]*len(df_bonds.index)
        self.smog_bonds = df_bonds
        angles = np.array(angles)
        df_angles = pd.DataFrame(angles, columns=['aai', 'aaj', 'aak'])
        df_angles['theta0 (deg)'] = mdtraj.compute_angles(traj, angles, periodic=pbc)[frame]*180/np.pi
        df_angles['Ka'] = [40]*len(df_angles.index)
        self.smog_angles = df_angles
        # for dihedrals there are dihedrals with periodicity as 1 or 3
        dihedrals = np.array(dihedrals)
        phi0 = mdtraj.compute_dihedrals(traj, dihedrals, periodic=pbc)[frame]
        df_dihedrals = pd.DataFrame(columns=['aai', 'aaj', 'aak', 'aal', 'mult', 'phi0 (deg)', 'Kd'])
        for i in range(dihedrals.shape[0]):
            row = dihedrals[i].tolist() + [1, (phi0[i] + np.pi)*180/np.pi, 1]
            df_dihedrals.loc[len(df_dihedrals.index)] = row
            row = dihedrals[i].tolist() + [3, 3*(phi0[i] + np.pi)*180/np.pi, 0.5]
            df_dihedrals.loc[len(df_dihedrals.index)] = row
        self.smog_dihedrals = df_dihedrals
        if get_native_pairs:
            print(f'Find native pairs with shadow algorithm')
            self.smog_native_pairs = shadow_map.find_ca_pairs_from_atomistic_pdb(self.atomistic_pdb, frame, radius, bonded_radius, cutoff, box, pbc)
        if get_exclusions:
            self.parse_smog_exclusions(exclude12, exclude13, exclude14, exclude_native_pairs)
            

class Protein(SMOGExclusionParser):
    def __init__(self, atoms, smog_energy_scale=2.5, sequence=None):
        # set sequence default value as None
        # it seems like at this stage we do not need to input protein sequence
        self.atoms = atoms # this may also include DNA atoms
        
        # make chainID unique for each chain
        old_chainID_list = self.atoms['chainID'].tolist()
        n_atoms = len(old_chainID_list)
        new_chainID_list = []
        chainID = 1
        for i in range(n_atoms):
            if i >= 1:
                if old_chainID_list[i] != old_chainID_list[i - 1]:
                    chainID += 1
            new_chainID_list.append(chainID)
        self.atoms['chainID'] = new_chainID_list
        
        # include real residue name in atoms
        atoms = self.atoms.copy()
        atoms['chain_res'] = atoms['chainID'].astype(str) + '_' + atoms['resSeq'].astype(str)
        sel = atoms[atoms['resname'].isin(_PROTEIN_RESIDUES)]
        resix = sel['chain_res'].unique()
        
        '''
        assert len(resix) == len(sequence), \
            f'The number of residues {len(resix)} does not agree with the length of the sequence {len(sequence)}'
        
        atoms.index = atoms['chain_res']
        
        for r, s in zip(resix, sequence):
            atoms.loc[r, 'real_resname'] = s
        '''
        
        atoms.index = list(range(len(atoms)))
        self.atoms = atoms

        protein_data = atoms[atoms.resname.isin(_PROTEIN_RESIDUES)].copy() # pick out protein atoms and residues
        # renumber residues
        resix = (protein_data.chainID.astype(str) + '_' + protein_data.resSeq.astype(str))
        res_unique = resix.unique()
        protein_data['resID'] = resix.replace(dict(zip(res_unique, range(len(res_unique)))))
        #print(protein_data['resID'])
        # renumber atom types
        atom_types_table = {'CA': 'ca'}
        protein_data['atom_list'] = protein_data['name'].replace(atom_types_table)
        protein_data['idx'] = protein_data.index.astype(int)
        self.protein_data = protein_data
        # lxc: change resID to serial, b/c we have many repeated resIDs in different chains
        self.atom_lists = protein_data.pivot(index='serial', columns='atom_list', values='idx').fillna(-1).astype(int)
        self.ca = self.atom_lists['ca'].tolist()
        self.nres = len(self.atom_lists) # the number of amino acids
        self.res_type = [r.iloc[0]['resname'] for i, r in protein_data.groupby('resID')]
        self.chain_starts = [c.iloc[0].resID for i, c in protein_data.groupby('chainID')]
        self.chain_ends = [c.iloc[-1].resID for i, c in protein_data.groupby('chainID')]
        #print(self.chain_ends)
        self.natoms = len(atoms) # the total number of atoms, may include DNA atoms
        #self.bonds = self._setup_bonds() # at this stage we do not need this
        #self.seq = sequence # at this stage we do not need this
        self.resi = pd.merge(self.atoms, self.protein_data, how='left').resID.fillna(-1).astype(int).tolist()
        print(f'Set smog bonded energy scale as {smog_energy_scale}')
        self.smog_energy_scale = smog_energy_scale
    
    def _setup_bonds(self):
        bonds = []
        #print(self.chain_ends)
        for i in range(self.nres):
            if i not in self.chain_ends:
                bonds.append((self.ca[i], self.ca[i + 1]))
        return bonds
    
    # for smog bonds, angles, dihedrals, native pairs, and exclusions, use names 'aai', 'aaj', 'aak', 'aal' for atoms to keep consistent with 3spn
    def add_smog_bonds(self, bonds_file):
        bonds = np.loadtxt(bonds_file, comments=';')
        if bonds.ndim == 1:
            bonds = np.reshape(bonds, (1, -1))
        bonds[:, :2] -= 1 # make atom index start from 0
        df_bonds = pd.DataFrame(bonds, columns=['aai', 'aaj', 'func', 'r0 (nm)', 'Kb'])
        self.smog_bonds = df_bonds
        return None

    def add_smog_angles(self, angles_file):
        angles = np.loadtxt(angles_file, comments=';')
        if angles.ndim == 1:
            angles = np.reshape(angles, (1, -1))
        angles[:, :3] -= 1 # make atom index start from 0
        df_angles = pd.DataFrame(angles, columns=['aai', 'aaj', 'aak', 'func', 'theta0 (deg)', 'Ka'])
        self.smog_angles = df_angles
        return None

    def add_smog_dihedrals(self, dihedrals_file):
        dihedrals = np.loadtxt(dihedrals_file, comments=';')
        if dihedrals.ndim == 1:
            dihedrals = np.reshape(dihedrals, (1, -1))
        dihedrals[:, :4] -= 1 # make atom index start from 0
        df_dihedrals = pd.DataFrame(dihedrals, columns=['aai', 'aaj', 'aak', 'aal', 'func', 'phi0 (deg)', 'Kd', 'mult'])
        self.smog_dihedrals = df_dihedrals
        return None

    def add_smog_native_pairs(self, native_pairs_file):
        native_pairs = np.loadtxt(native_pairs_file, comments=';')
        if native_pairs.ndim == 1:
            native_pairs = np.reshape(native_pairs, (1, -1))
        native_pairs[:, :2] -= 1 # make atom index start from 0
        df_native_pairs = pd.DataFrame(native_pairs, columns=['aai', 'aaj', 'type', 'epsilon', 'mu', 'sigma', 'alpha'])
        self.smog_native_pairs = df_native_pairs
        return None

    def add_smog_exclusions(self, exclusions_file):
        print('Be careful, smog output exclusions.dat file only includes native pair exclusions!')
        exclusions = np.loadtxt(exclusions_file, comments=';')
        if exclusions.ndim == 1:
            exclusions = np.reshape(exclusions, (1, -1))
        exclusions[:, :2] -= 1 # make atom index start from 0
        df_exclusions = pd.DataFrame(exclusions, columns=['aai', 'aaj'])
        self.smog_exclusions = df_exclusions
        return None

    @classmethod
    def fromPDB(cls, pdb, pdbout='CoarseProtein.pdb'):
        """ Initializes a protein form a pdb, making all the atoms coarse-grained"""
        pass

    @classmethod
    def fromCoarsePDB(cls, pdb_file, sequence=None):
        """ Initializes the protein from an already coarse grained pdb"""
        atoms = parsePDB(pdb_file)
        return cls(atoms, sequence)
    
    @classmethod
    def fromCoarsePandasDataFrame(cls, df, sequence=None):
        # old name: fromCoarsePDB_through_pdframe
        """ Initializes the protein from an already coarse grained pdb"""
        atoms = df
        return cls(atoms, sequence)


    def parseConfigurationFile(self):
        """ Parses the AWSEM configuration file to use for the topology and to set the forces"""
        pass

    def computeTopology(self):
        """ Compute the bonds and angles from the pdb"""
        pass

    @staticmethod
    # STILL NEED TO WORK ON THAT!
    def GetHeavyAtom(pdb_table):
        """ Selects heavy atoms from a pdb table and returns a table containing only the heavy atoms """
        protein_residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                            'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                            'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                            'SER', 'THR', 'TRP', 'TYR', 'VAL']
        heavy_atoms = ["N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2", "OG1", ""]

        # Select coarse grained atoms
        selection = pdb_table[pdb_table.resname.isin(protein_residues) & pdb_table.name.isin(heavy_atoms)].copy()


        # Replace resnames
        selection['real_resname'] = selection.resname.copy()
        #resname = selection.resname.copy()
        #resname[:] = 'NGP'
        #selection.resname = resname


        # Renumber
        selection['serial'] = range(len(selection))
        return selection

    @staticmethod
    def CoarseGrain(pdb_table):
        # note this function only returns protein CA atoms and ignores other atoms
        """ Selects CA atoms from a pdb table and returns a table containing only the coarse-grained atoms for CA """
        protein_residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                            'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                            'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                            'SER', 'THR', 'TRP', 'TYR', 'VAL']
        calpha_atoms = ["CA"]

        # Select coarse grained atoms
        selection = pdb_table[pdb_table.resname.isin(protein_residues) & pdb_table.name.isin(calpha_atoms)].copy()

        # Replace resnames
        selection['real_resname'] = selection.resname.copy()
        #resname = selection.resname.copy()
        #resname[:] = 'NGP'
        #selection.resname = resname

        # Renumber
        selection['serial'] = range(len(selection))
        selection = selection.reset_index(drop=True)
        return selection

    @staticmethod
    def write_sequence(Coarse, seq_file='protein.seq'):
        protein_data = Coarse[Coarse.resname.isin(_PROTEIN_RESIDUES)].copy()
        resix = (protein_data.chainID + '_' + protein_data.resSeq.astype(str))
        res_unique = resix.unique()
        protein_data['resID'] = resix.replace(dict(zip(res_unique, range(len(res_unique)))))
        protein_sequence = [r.iloc[0]['real_resname'] for i, r in protein_data.groupby('resID')]
        protein_sequence_one = [three_to_one(a) for a in protein_sequence]

        with open(seq_file, 'w+') as ps:
            ps.write(''.join(protein_sequence_one))


def buildProteinNonBondedExclusionsList(smog_bonds, smog_angles, smog_dihedrals, smog_pairs, rigid_body_identity=None, 
                                        exclude12=True, exclude13=True, exclude14=True, exclude_native_pairs=True):
    protein_exclusions_candidate_list = []
    if exclude12:
        for i in range(smog_bonds.shape[0]):
            p0 = int(smog_bonds[i, 0])
            p1 = int(smog_bonds[i, 1])
            protein_exclusions_candidate_list.append((p0, p1))
    if exclude13:
        for i in range(smog_angles.shape[0]):
            p0 = int(smog_angles[i, 0])
            p2 = int(smog_angles[i, 2])
            protein_exclusions_candidate_list.append((p0, p2))
    if exclude14:
        for i in range(smog_dihedrals.shape[0]):
            p0 = int(smog_dihedrals[i, 0])
            p3 = int(smog_dihedrals[i, 3])
            protein_exclusions_candidate_list.append((p0, p3))
    if exclude_native_pairs:
        for i in range(smog_pairs.shape[0]):
            p0 = int(smog_pairs[i, 0])
            p1 = int(smog_pairs[i, 1])
            protein_exclusions_candidate_list.append((p0, p1))
    protein_exclusions_list = []
    for each in protein_exclusions_candidate_list:
        p0, p1 = each[0], each[1]
        flag = True
        if rigid_body_identity is not None:
            r0 = rigid_body_identity[p0]
            r1 = rigid_body_identity[p1]
            if (r0 is not None) and (r0 == r1):
                flag = False
        if flag:
            if p0 > p1:
                p0, p1 = p1, p0
            protein_exclusions_list.append((p0, p1))
    return protein_exclusions_list


'''
# old code
def buildProteinNonBondedExclusionsList(protein, smog_exclusions_data, rigid_body_identity=None, 
                                        exclude_native_pairs=True, exclude_1_2=True, exclude_1_3=True, 
                                        exclude_1_4=True):
    protein_exclusions_list = []
    # add protein exclusions
    # if rigid_body_identity != None, then do not add atom pairs to exclusion list if they are within the same rigid body
    # by default we add exclusions for 1-2, 1-3, and 1-4 interactions in each protein chain based on analyzing protein topology
    # based on the model and conditions, we can decide whether some 1-2, 1-3, or 1-4 nonbonded interactions can be included
    # pick out protein atoms, as the input protein may include CG dna atoms
    protein_atoms = protein.atoms.copy()
    is_protein = protein_atoms['resname'].isin(_PROTEIN_RESIDUES)
    protein_atoms = protein_atoms[is_protein]
    protein_atoms['index'] = protein_atoms.index
    protein_atoms.index = protein_atoms['chainID'].astype(str) + '_' + protein_atoms['resSeq'].astype(str)
    exclude_neighbor_list = []
    if exclude_1_2:
        exclude_neighbor_list += [1]
    if exclude_1_3:
        exclude_neighbor_list += [2]
    if exclude_1_4:
        exclude_neighbor_list += [3]
    for atom1_id, resSeq, chainID in zip(protein_atoms['index'], protein_atoms['resSeq'], protein_atoms['chainID']):
        resSeq = int(resSeq)
        for i in exclude_neighbor_list:
            target_index = str(chainID) + '_' + str(resSeq + i)
            try:
                atom2_id = protein_atoms.loc[target_index, 'index']
                atom1_id, atom2_id = int(atom1_id), int(atom2_id)
                if atom1_id > atom2_id:
                    atom1_id, atom2_id = atom2_id, atom1_id
                protein_exclusions_list.append((atom1_id, atom2_id))
            except KeyError:
                pass
    if exclude_native_pairs:
        # add exclusions for native pairs
        for i in range(smog_exclusions_data.shape[0]):
            atom1_id, atom2_id = int(smog_exclusions_data[i, 0]), int(smog_exclusions_data[i, 1])
            if atom1_id > atom2_id:
                atom1_id, atom2_id = atom2_id, atom1_id
            protein_exclusions_list.append((atom1_id, atom2_id))
    protein_exclusions_list = list(set(protein_exclusions_list)) # remove duplicates
    if rigid_body_identity != None:
        # if two atoms are within the same rigid body, then we do not need to put this pair into exclusion list
        new_protein_exclusions_list = []
        for each in protein_exclusions_list:
            atom1, atom2 = each[0], each[1]
            r1, r2 = rigid_body_identity[atom1], rigid_body_identity[atom2]
            if r1 == None or r2 == None or r1 != r2:
                new_protein_exclusions_list.append(each)
        protein_exclusions_list = new_protein_exclusions_list
    return protein_exclusions_list
'''

def test_Protein_fromCoarsePDB():
    pass


def test_Protein_fromPDB():
    pass


def test_Protein_parseConfigurationFile():
    pass


def test_Protein_computeTopology():
    pass


    

