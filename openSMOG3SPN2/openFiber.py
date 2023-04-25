# define some helper functions for running chromatin fiber simulations by openmm

import numpy as np
import pandas as pd
import simtk.openmm
import simtk.unit as unit
import simtk.openmm.app as app
import os
import sys
import itertools
import MDAnalysis as mda
import time
import copy
from .open3SPN2 import ff3SPN2 as ff3SPN2
from .calphaSMOG import ffCalpha as ffCalpha
from .calphaSMOG import basicTerms as basicTerms
from . import openRigid

__location__ = os.path.dirname(os.path.abspath(__file__))

_complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
_dnaResidues = ['DA', 'DC', 'DT', 'DG']
_proteinResidues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                    'SER', 'THR', 'TRP', 'TYR', 'VAL']

_proteinMass = dict(ALA=71.0788, ARG=156.1875, ASN=114.1038, ASP=115.0886, CYS=103.1388, 
                    GLU=129.1155, GLN=128.1307, GLY=57.0519, HIS=137.1411, ILE=113.1594, 
                    LEU=113.1594, LYS=128.1741, MET=131.1926, PHE=147.1766, PRO=97.1167, 
                    SER=87.0782, THR=101.1051, TRP=186.2132, TYR=163.1760, VAL=99.1326)

_dnaMass = dict(P=94.9696, S=83.1104, A=134.1220, T=125.1078, G=150.1214, C=110.0964)


def change_serial_resSeq(mol, change_serial=True, change_resSeq=True):
    # update serial and resSeq for given molecule
    # make the serial number increase from 1 to n_atoms 
    # n_atoms is the total number of atoms
    # make resSeq increase from 1 to n_res
    # n_res is the total number of residues
    mol_new = mol.copy()
    n_atoms = len(mol.index)
    if change_serial:
        mol_new['serial'] = list(range(1, n_atoms + 1))
    old_resSeq = mol['resSeq'].tolist()
    chainID = mol['chainID'].tolist()
    if change_resSeq:
        new_resSeq = []
        i = 1
        for i in range(n_atoms):
            if i >= 1:
                if (old_resSeq[i - 1] != old_resSeq[i]) or (chainID[i - 1] != chainID[i]):
                    i += 1
            new_resSeq.append(i)
        mol_new['resSeq'] = new_resSeq
    return mol_new


def combine_molecules(mol1, mol2, add_serial=True, add_resSeq=True): 
    # combine the pandas table mol1 and mol2 sequentially
    mol2_new = mol2.copy()
    mol1_n_atoms = len(mol1.index)
    if add_serial:
        mol2_new['serial'] += mol1_n_atoms
    if add_resSeq:
        mol1_resSeq = mol1['resSeq'].tolist()
        mol1_chainID = mol1['chainID'].tolist()
        if mol1_n_atoms >= 1:
            mol1_n_res = 1
            for i in range(1, mol1_n_atoms):
                if (mol1_resSeq[i - 1] != mol1_resSeq[i]) or (mol1_chainID[i - 1] != mol1_chainID[i]):
                    mol1_n_res += 1
        else:
            mol1_n_res = 0
        mol2_new['resSeq'] += mol1_n_res
    mol_combined = pd.concat([mol1, mol2_new], ignore_index=True)
    mol_combined['serial'] = mol_combined['serial'].astype('int64')
    mol_combined['resSeq'] = mol_combined['resSeq'].astype('int64')
    return mol_combined


def add_TER_END_and_remove_OXT_for_pdb(input_pdb_file_path, output_pdb_file_path):
    # use this function to add TER and END to pdb files
    if not os.path.exists(input_pdb_file_path):
        print(f'input file {input_pdb_file_path} does not exist')
        return None
    with open(input_pdb_file_path, 'r') as input_pdb:
        input_pdb_lines = input_pdb.readlines()
    # remove empty lines or lines with only whitespace
    input_pdb_no_empty_lines = []
    for i in range(len(input_pdb_lines)):
        if len(input_pdb_lines[i].strip()) != 0:
            input_pdb_no_empty_lines.append(input_pdb_lines[i])
    output_pdb_lines = []
    output_pdb_lines.append(input_pdb_no_empty_lines[0])
    n_input_pdb_no_empty_lines = len(input_pdb_no_empty_lines)
    for i in range(1, n_input_pdb_no_empty_lines):
        if input_pdb_no_empty_lines[i][:4] == 'ATOM':
            chain_ID = input_pdb_no_empty_lines[i][21]
            if input_pdb_no_empty_lines[i - 1][:4] == 'ATOM' and input_pdb_no_empty_lines[i - 1][21] != chain_ID:
                output_pdb_lines.append('TER\n')
        # Remove the line with "OXT"
        if str(input_pdb_no_empty_lines[i][12:15]) == "OXT":
            pass
        else:
            output_pdb_lines.append(input_pdb_no_empty_lines[i])
    if output_pdb_lines[-1][:3] != 'END':
        output_pdb_lines.append('END\n')
    with open(output_pdb_file_path, 'w') as output_pdb:
        for each_line in output_pdb_lines:
            output_pdb.write(each_line)


def change_unique_chainID(mol):
    # make chainID unique for each chain
    old_chainID_list = mol['chainID'].tolist()
    n_atoms = len(old_chainID_list)
    new_chainID_list = []
    chainID = 1
    for i in range(n_atoms):
        if i >= 1:
            if old_chainID_list[i] != old_chainID_list[i - 1]:
                chainID += 1
        new_chainID_list.append(chainID)
    new_mol = mol.copy()
    new_mol['chainID'] = new_chainID_list
    return new_mol


def get_single_fiber_histones_chains(n_nucl):
    # set up a 2d list histone_chains
    # histone_chains[i] is a list includes the atom index in the i-th chain 
    n_CA_each_chain_in_each_nucl = [135, 102, 128, 122, 135, 102, 128, 122]
    cumsum_n_CA_each_chain_in_each_nucl = np.cumsum(np.array(n_CA_each_chain_in_each_nucl, dtype=int))
    histone_chains = []
    n_chains_each_histone = len(n_CA_each_chain_in_each_nucl)
    for i in range(n_chains_each_histone):
        histone_chains.append([])
    atom_id = 0
    chain_id = 0
    n_CA_each_nucl = 974
    while atom_id < n_CA_each_nucl:
        histone_chains[chain_id].append(atom_id)
        atom_id += 1
        if atom_id >= cumsum_n_CA_each_chain_in_each_nucl[chain_id]:
            chain_id += 1
    # set up a 2d list histone_chains
    # histones_chains[i] is a list includes the atom index in the i-th chain 
    # for all the histones, there are 8*n_nucl chains in all
    histones_chains = []
    for i in range(n_nucl):
        for j in range(n_chains_each_histone):
            chain = np.array(histone_chains[j], dtype=int) + i*n_CA_each_nucl
            histones_chains.append(chain.tolist())
    return histones_chains
   

# define a new class to hold SMOG+3SPN2 molecules and set up simulations
# note we have to keep DNA atom chainIDs unique (i.e. different chains have different chainIDs), so that base pair and cross stacking interactions can be correctly parsed
class SMOG3SPN2(ffCalpha.SMOGExclusionParser):
    def __init__(self):
        # initialize
        self.atoms = None
        self.ff_attribute_names = ['smog_bonds', 'smog_angles', 'smog_dihedrals', 'smog_native_pairs', 'smog_exclusions', 'dna_bonds', 'dna_angles', 'dna_stackings', 'dna_dihedrals', 'dna_exclusions', 'exclusions']
        self.dna_def_attribute_names = ['config', 'particle_definition', 'bond_definition', 'angle_definition', 'dihedral_definition', 'stacking_definition', 'pair_definition', 'cross_definition']
        self.property_attribute_names = ['DNAtype', 'periodic', 'temperature', 'salt_concentration']
        attribute_names = self.ff_attribute_names + self.dna_def_attribute_names + self.property_attribute_names
        for each_name in attribute_names:
            setattr(self, each_name, None)
    
    def add_protein_dna_object(self, mol, change_chainID=True):
        new_atoms = mol.atoms.copy()
        chainIDs = new_atoms['chainID'].tolist()
        new_chainIDs = []
        if self.atoms is None:
            add_index = 0
        else:
            add_index = len(self.atoms.index)
        if change_chainID:
            # make chainIDs unique
            if self.atoms is None:
                n_existing_chains = 0
            else:
                n_existing_chains = len(set(self.atoms['chainID'].tolist()))
            c = n_existing_chains + 1
            for i in range(len(chainIDs)):
                if i >= 1:
                    if chainIDs[i] != chainIDs[i - 1]:
                        c += 1
                new_chainIDs.append(c)
            new_atoms['chainID'] = new_chainIDs
        # combine atoms
        if self.atoms is None:
            self.atoms = new_atoms
        else:
            self.atoms = pd.concat([self.atoms, new_atoms], ignore_index=True)
        self.atoms['serial'] = list(range(1, len(self.atoms.index) + 1))
        # update force field parameters
        for each_name in self.ff_attribute_names:
            if hasattr(mol, each_name):
                if getattr(mol, each_name) is not None:
                    new_attribute = getattr(mol, each_name).copy()
                    for aa in ['aai', 'aaj', 'aak', 'aal']:
                        if aa in new_attribute.columns:
                            new_attribute[aa] += add_index
                    if hasattr(self, each_name):
                        if getattr(self, each_name) is None:
                            setattr(self, each_name, new_attribute)
                        else:
                            combined_attribute = pd.concat([getattr(self, each_name), new_attribute], ignore_index=True)
                            setattr(self, each_name, combined_attribute)
                    else:
                        setattr(self, each_name, new_attribute)
        # update dna definitions and properties
        # print warnings if the new attribute is not consistent with the original one
        attribute_names = self.dna_def_attribute_names + self.property_attribute_names
        for each_name in attribute_names:
            if hasattr(mol, each_name):
                new_attribute = getattr(mol, each_name)
                if new_attribute is not None:
                    if hasattr(self, each_name):
                        original_attribute = getattr(self, each_name)
                        if original_attribute is None:
                            setattr(self, each_name, new_attribute)
                        else:
                            # check if the new attribute and the original attribute are consistent
                            if isinstance(original_attribute, pd.DataFrame):
                                if original_attribute.equals(new_attribute):
                                    pass
                                else:
                                    print(f'Warning: attribute {each_name} of the new molecule is not consistent with the original one, and we keep the original one')
                            else:
                                if original_attribute == new_attribute:
                                    pass
                                else:
                                    print(f'Warning: attribute {each_name} of the new molecule is not consistent with the original one, and we keep the original one')
                    else:
                        setattr(self, each_name, new_attribute)

    def create_system(self, box_a=100, box_b=100, box_c=100, remove_cmmotion=False):
        self.system = simtk.openmm.System()
        if self.periodic:
            box_vec_a = np.array([box_a, 0, 0])*unit.nanometer
            box_vec_b = np.array([0, box_b, 0])*unit.nanometer
            box_vec_c = np.array([0, 0, box_c])*unit.nanometer
            self.system.setDefaultPeriodicBoxVectors(box_vec_a, box_vec_b, box_vec_c)
        for i, row in self.atoms.iterrows():
            atom_name = row['name']
            resname = row['resname']
            if resname in _proteinResidues:
                if atom_name == 'CA':
                    mass = _proteinMass[resname]
                else:
                    print('Warning: atom from pdb file cannot be recognized! Set this atom mass as 0.')
                    mass = 0
            elif resname in _dnaResidues:
                if atom_name in _dnaMass.keys():
                    mass = _dnaMass[atom_name]
                else:
                    print('Warning: atom from pdb file cannot be recognized! Set this atom mass as 0.')
                    mass = 0
            else:
                print('Warning: atom from pdb file cannot be recognized! Set this atom mass as 0.')
                mass = 0
            self.system.addParticle(mass)
        if remove_cmmotion:
            self.system.addForce(simtk.openmm.CMMotionRemover())
    
    def set_rigid_bodies(self):
        # this function set rigid bodies and remove bonded interactions and exclusions within same rigid bodies
        n_atoms = len(self.atoms.index)
        rigid_body_identities = []
        for i in range(n_atoms):
            rigid_body_identities.append(None)
        for i in range(len(self.rigid_bodies)):
            for j in self.rigid_bodies[i]:
                rigid_body_identities[j] = i
        for each_name in self.ff_attribute_names:
            if hasattr(self, each_name):
                if getattr(self, each_name) is not None:
                    original_df = getattr(self, each_name).copy()
                    new_df = pd.DataFrame(columns=original_df.columns)
                    for i, row in original_df.iterrows():
                        flag = True
                        involved_atoms = []
                        for col_name in ['aai', 'aaj', 'aak', 'aal']:
                            if col_name in original_df.columns:
                                involved_atoms.append(int(row[col_name]))
                        involved_atom_identities = [rigid_body_identities[j] for j in involved_atoms]
                        if involved_atom_identities[0] is not None:
                            if all([x==involved_atom_identities[0] for x in involved_atom_identities]):
                                flag = False
                        if flag:
                            new_df.loc[len(new_df.index)] = row
                    setattr(self, each_name, new_df)
        openRigid.createRigidBodies(self.system, self.rigid_coord, self.rigid_bodies)
    
    def add_forces(self, exclude_force_names=[], combine_smog_dna_exclusions=True):
        # add forces with default settings
        # if we do not want to add certain force, put the force name in exclude_forces
        all_force_names = ['SMOGBond', 'SMOGAngle', 'SMOGDihedral', 'SMOGNativePair', 'DNABond', 'DNAAngle', 
                           'DNAStacking', 'DNADihedral', 'DNABasePair', 'DNACrossStacking', 'AllVanderWaals', 
                           'AllElectrostatics']
        
        all_forces = dict(SMOGBond=basicTerms.smog_bond_term,
                          SMOGAngle=basicTerms.smog_angle_term,
                          SMOGDihedral=basicTerms.smog_dihedral_term,
                          SMOGNativePair=basicTerms.smog_native_pair_term,
                          DNABond=ff3SPN2.dna_bond_term,
                          DNAAngle=ff3SPN2.dna_angle_term,
                          DNAStacking=ff3SPN2.dna_stacking_term,
                          DNADihedral=ff3SPN2.dna_dihedral_term,
                          DNABasePair=ff3SPN2.dna_base_pair_term_ensemble,
                          DNACrossStacking=ff3SPN2.dna_cross_stacking_term_ensemble,
                          AllVanderWaals=basicTerms.combined_DD_PD_vdwl_PP_MJ_term,
                          AllElectrostatics=ff3SPN2.all_elec_term)
        
        all_force_groups = dict(SMOGBond=1,
                                SMOGAngle=2,
                                SMOGDihedral=3,
                                SMOGNativePair=4,
                                DNABond=5,
                                DNAAngle=6,
                                DNAStacking=7,
                                DNADihedral=8,
                                DNABasePair=9,
                                DNACrossStacking=10,
                                AllVanderWaals=11, 
                                AllElectrostatics=12)

        # add forces to protein-DNA system with SMOG and 3SPN force field
        if combine_smog_dna_exclusions:
            # combine protein and dna exclusions
            print('Combine smog and dna exclusions')
            exclusions = []
            if self.smog_exclusions is not None:
                for _, row in self.smog_exclusions.iterrows():
                    exclusions.append((int(row['aai']), int(row['aaj'])))
            if self.dna_exclusions is not None:
                for _, row in self.dna_exclusions.iterrows():
                    exclusions.append((int(row['aai']), int(row['aaj'])))
            exclusions = sorted(list(set(exclusions)))
            exclusions = np.array([[int(x[0]), int(x[1])] for x in exclusions])
            self.exclusions = pd.DataFrame(exclusions, columns=['aai', 'aaj'])
        for each_name in all_force_names:
            if each_name in exclude_force_names:
                print(f'Do not add force {each_name}')
            else:
                time1 = time.time()
                force_group = all_force_groups[each_name]
                if each_name == 'DNABasePair':
                    bp_forces = all_forces[each_name](self, force_group=force_group)
                    for i in bp_forces:
                        self.system.addForce(bp_forces[i])
                elif each_name == 'DNACrossStacking':
                    cstk_forces = all_forces[each_name](self, force_group=force_group)
                    for i in cstk_forces:
                        c1, c2 = cstk_forces[i]
                        self.system.addForce(c1)
                        self.system.addForce(c2)
                else:
                    force = all_forces[each_name](self, force_group=force_group)
                    self.system.addForce(force)
                time2 = time.time()
                print(f'Adding force {each_name} takes {time2 - time1} seconds')

    def set_simulation(self, integrator_name='NoseHoover', collision=1/unit.picosecond, friction=1/unit.picosecond, 
                       timestep=10*unit.femtosecond, platform_name='CPU', properties={'Precision': 'mixed'}, 
                       init_coord=None):
        if integrator_name == 'NoseHoover':
            print('Use NoseHooverIntegrator')
            integrator = simtk.openmm.NoseHooverIntegrator(self.temperature, collision, timestep)
        elif integrator_name == 'Langevin':
            print('Use LangevinIntegrator')
            integrator = simtk.openmm.LangevinIntegrator(self.temperature, friction, timestep)
        print(f'Use platform {platform_name}')
        platform = simtk.openmm.Platform.getPlatformByName(platform_name)
        if platform_name in ['CUDA', 'OpenCL']:
            if 'Precision' not in properties:
                properties['Precision'] = 'mixed'
            precision = properties['Precision']
            print(f'Use precision: {precision}')
            self.simulation = app.Simulation(self.top, self.system, integrator, platform, properties)
        else:
            self.simulation = app.Simulation(self.top, self.system, integrator, platform)
        if init_coord is not None:
            self.simulation.context.setPositions(init_coord)
            
    def add_reporters(self, report_interval, output_dcd='output.dcd', report_dcd=True, report_state=True):
        if report_dcd:
            dcd_reporter = app.DCDReporter(output_dcd, report_interval, enforcePeriodicBox=self.use_pbc)
            self.simulation.reporters.append(dcd_reporter)
        if report_state:
            state_reporter = app.StateDataReporter(sys.stdout, report_interval, step=True, time=True, 
                                                   potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
                                                   temperature=True, speed=True)
            self.simulation.reporters.append(state_reporter)
    
    def save_system(self, system_xml='system.xml'):
        with open(system_xml, 'w') as f:
            f.write(simtk.openmm.XmlSerializer.serialize(self.system))

'''
# old code
def load_smog_bonds(bonds_file):
    bonds_data = np.loadtxt(bonds_file, comments=';')
    if bonds_data.ndim == 1:
        bonds_data = np.reshape(bonds_data, (1, -1))
    bonds_data[:, :2] -= 1 # make atom index start from 0
    return bonds_data


def load_smog_angles(angles_file):
    angles_data = np.loadtxt(angles_file, comments=';')
    if angles_data.ndim == 1:
        angles_data = np.reshape(angles_data, (1, -1))
    angles_data[:, :3] -= 1 # make atom index start from 0
    return angles_data


def load_smog_dihedrals(dihedrals_file):
    dihedrals_data = np.loadtxt(dihedrals_file, comments=';')
    if dihedrals_data.ndim == 1:
        dihedrals_data = np.reshape(dihedrals_data, (1, -1))
    dihedrals_data[:, :4] -= 1 # make atom index start from 0
    return dihedrals_data


def load_smog_pairs(pairs_file):
    pairs_data = np.loadtxt(pairs_file, comments=';')
    if pairs_data.ndim == 1:
        pairs_data = np.reshape(pairs_data, (1, -1))
    pairs_data[:, :2] -= 1 # make atom index start from 0
    return pairs_data


def load_smog_exclusions(exclusions_file):
    exclusions_data = np.loadtxt(exclusions_file, comments=';')
    if exclusions_data.ndim == 1:
        exclusions_data = np.reshape(exclusions_data, (1, -1))
    exclusions_data[:, :2] -= 1 # make atom index start from 0
    return exclusions_data


def write_smog_pairs(smog_pairs_data, pairs_file):
    # input smog_pairs_data uses the format that atom index starts from 0
    # output pairs_file uses smog format that atom index starts from 1
    header = ';ai aj type epsilon mu sigma alpha'
    with open(pairs_file, 'w') as output_writer:
        output_writer.write(header + '\n')
        for each in smog_pairs_data:
            i, j = int(each[0]) + 1, int(each[1]) + 1
            pair_type = int(each[2])
            epsilon = float(each[3])
            mu = float(each[4])
            sigma = float(each[5])
            alpha = float(each[6])
            new_line = '%d %d %d %.9e %.9e %.9e %.9e\n' % (i, j, pair_type, epsilon, mu, sigma, alpha)
            output_writer.write(new_line)


def write_smog_exclusions(smog_exclusions_data, exclusions_file):
    # input smog_exclusions_data uses the format that atom index starts from 0
    # output exclusions_file uses smog format that atom index starts from 1
    header = ';ai aj'
    with open(exclusions_file, 'w') as output_writer:
        output_writer.write(header + '\n')
        for each in smog_exclusions_data:
            i, j = int(each[0]) + 1, int(each[1]) + 1
            new_line = '%d %d\n' % (i, j)
            output_writer.write(new_line)


def write_smog_dihedrals(smog_dihedrals_data, dihedrals_file):
    # input smog_dihedrals_data uses the format that atom index starts from 0
    # output dihedrals_file uses smog format that atom index starts from 1
    header = ';ai aj ak al func phi0(deg) Kd mult'
    with open(dihedrals_file, 'w') as output_writer:
        output_writer.write(header + '\n')
        for each in smog_dihedrals_data:
            i, j, k, l = int(each[0]) + 1, int(each[1]) + 1, int(each[2]) + 1, int(each[3]) + 1
            func_type = int(each[4])
            phi0 = float(each[5])
            Kd = float(each[6])
            mult = int(each[7])
            new_line = '%d %d %d %d %d %.9e %.9e %d\n' % (i, j, k, l, func_type, phi0, Kd, mult)
            output_writer.write(new_line)


def extend_single_fiber_to_fibers_bonds(single_fiber_bonds_data, n_fibers, n_cg_atoms_each_fiber):
    n_bonds_each_fiber = single_fiber_bonds_data.shape[0]
    n_parameters = single_fiber_bonds_data.shape[1]
    fibers_bonds_data = np.zeros((n_bonds_each_fiber*n_fibers, n_parameters))
    for i in range(n_fibers):
        fiber_i_bonds_data = single_fiber_bonds_data.copy()
        fiber_i_bonds_data[:, :2] += n_cg_atoms_each_fiber*i
        j = i*n_bonds_each_fiber
        k = (i + 1)*n_bonds_each_fiber
        fibers_bonds_data[j:k, :] = fiber_i_bonds_data
    return fibers_bonds_data


def extend_single_fiber_to_fibers_angles(single_fiber_angles_data, n_fibers, n_cg_atoms_each_fiber):
    n_angles_each_fiber = single_fiber_angles_data.shape[0]
    n_parameters = single_fiber_angles_data.shape[1]
    fibers_angles_data = np.zeros((n_angles_each_fiber*n_fibers, n_parameters))
    for i in range(n_fibers):
        fiber_i_angles_data = single_fiber_angles_data.copy()
        fiber_i_angles_data[:, :3] += n_cg_atoms_each_fiber*i
        j = i*n_angles_each_fiber
        k = (i + 1)*n_angles_each_fiber
        fibers_angles_data[j:k, :] = fiber_i_angles_data
    return fibers_angles_data


def extend_single_fiber_to_fibers_dihedrals(single_fiber_dihedrals_data, n_fibers, n_cg_atoms_each_fiber):
    n_dihedrals_each_fiber = single_fiber_dihedrals_data.shape[0]
    n_parameters = single_fiber_dihedrals_data.shape[1]
    fibers_dihedrals_data = np.zeros((n_dihedrals_each_fiber*n_fibers, n_parameters))
    for i in range(n_fibers):
        fiber_i_dihedrals_data = single_fiber_dihedrals_data.copy()
        fiber_i_dihedrals_data[:, :4] += n_cg_atoms_each_fiber*i
        j = i*n_dihedrals_each_fiber
        k = (i + 1)*n_dihedrals_each_fiber
        fibers_dihedrals_data[j:k, :] = fiber_i_dihedrals_data
    return fibers_dihedrals_data


def extend_single_fiber_to_fibers_exclusions(single_fiber_exclusions_data, n_fibers, n_cg_atoms_each_fiber):
    n_exclusions_each_fiber = single_fiber_exclusions_data.shape[0]
    n_parameters = single_fiber_exclusions_data.shape[1]
    fibers_exclusions_data = np.zeros((n_exclusions_each_fiber*n_fibers, n_parameters))
    for i in range(n_fibers):
        fiber_i_exclusions_data = single_fiber_exclusions_data.copy()
        fiber_i_exclusions_data[:, :2] += n_cg_atoms_each_fiber*i
        j = i*n_exclusions_each_fiber
        k = (i + 1)*n_exclusions_each_fiber
        fibers_exclusions_data[j:k, :] = fiber_i_exclusions_data
    return fibers_exclusions_data


def extend_single_fiber_to_fibers_pairs(single_fiber_pairs_data, n_fibers, n_cg_atoms_each_fiber):
    n_pairs_each_fiber = single_fiber_pairs_data.shape[0]
    n_parameters = single_fiber_pairs_data.shape[1]
    fibers_pairs_data = np.zeros((n_pairs_each_fiber*n_fibers, n_parameters))
    for i in range(n_fibers):
        fiber_i_pairs_data = single_fiber_pairs_data.copy()
        fiber_i_pairs_data[:, :2] += n_cg_atoms_each_fiber*i
        j = i*n_pairs_each_fiber
        k = (i + 1)*n_pairs_each_fiber
        fibers_pairs_data[j:k, :] = fiber_i_pairs_data
    return fibers_pairs_data


# add forces
def add_protein_dna_forces(system, protein, dna, smog_data, protein_exclusions_list, dna_exclusions_list, 
                           rigid_body_identity, scale_factor, verbose=True):
    # check if protein and dna have consistent PBC setting
    if protein.periodic != dna.periodic:
        print('Error: protein and dna periodicity are not consistent!')
        return None
    periodic = protein.periodic
    
    # exclusions_list includes the pairs of atoms that should be excluded for certain interactions
    exclusions_list = protein_exclusions_list + dna_exclusions_list
    exclusions_list = list(set(exclusions_list))

    # add protein-protein bonds, angles, dihedrals, native pairs, and nonbonded MJ potential
    force_name_list1 = ['BondProtein', 'AngleProtein', 'DihedralProtein', 'NativePairProtein']
    for force_name in force_name_list1:
        time1 = time.time()
        group = force_groups[force_name]
        if force_name == 'BondProtein':
            smog_bonds_data = smog_data['bonds']
            force = all_forces[force_name](smog_bonds_data, rigid_body_identity, periodic, scale_factor, group)
        elif force_name == 'AngleProtein':
            smog_angles_data = smog_data['angles']
            force = all_forces[force_name](smog_angles_data, rigid_body_identity, periodic, scale_factor, group)
        elif force_name == 'DihedralProtein':
            smog_dihedrals_data = smog_data['dihedrals']
            force = all_forces[force_name](smog_dihedrals_data, rigid_body_identity, periodic, scale_factor, group)
        elif force_name == 'NativePairProtein':
            smog_pairs_data = smog_data['pairs']
            force = all_forces[force_name](smog_pairs_data, rigid_body_identity, periodic, scale_factor, group)
        else:
            print(f'Warning: force {force_name} is not included into the system!')
            continue
        # double check periodicity
        if force.usesPeriodicBoundaryConditions() != periodic:
            print(f'Error: PBC setting for {force_name} is not correct!')
            return None
        system.addForce(force)
        time2 = time.time()
        if verbose:
            print('Adding force %s takes %.6f seconds' % (force_name, time2 - time1))
    
    # add DNA-DNA bonds, angles, stackings, dihedrals, base-pairs, cross-stackings
    force_name_list2 = ['BondDNA', 'AngleDNA', 'Stacking', 'DihedralDNA', 'BasePair', 'CrossStacking']
    for force_name in force_name_list2:
        time1 = time.time()
        group = force_groups[force_name]
        if force_name == 'BasePair':
            bp_forces = all_forces[force_name](dna, force_group=group)
            for i in bp_forces:
                # double check periodicity
                if bp_forces[i].usesPeriodicBoundaryConditions() != periodic:
                    print(f'Error: PBC setting for {force_name} is not correct!')
                    return None
                system.addForce(bp_forces[i])
        elif force_name == 'CrossStacking':
            cstk_forces = all_forces[force_name](dna, force_group=group)
            for i in cstk_forces:
                c1, c2 = cstk_forces[i]
                # double check periodicity
                if (c1.usesPeriodicBoundaryConditions() != periodic) or (c2.usesPeriodicBoundaryConditions() != periodic):
                    print(f'Error: PBC setting for {force_name} is not correct!')
                    return None
                system.addForce(c1)
                system.addForce(c2)
        else:
            force = all_forces[force_name](dna, rigid_body_identity, force_group=group)
            # double check periodicity
            if force.usesPeriodicBoundaryConditions() != periodic:
                print(f'Error: PBC setting for {force_name} is not correct!')
                return None
            system.addForce(force)
        time2 = time.time()
        if verbose:
            print('Adding force %s takes %.6f seconds' % (force_name, time2 - time1))
    
    # add DNA-DNA and protein-DNA excluded volume interactions, and all the electrostatic interactions
    force_name_list3 = ['AllVanderWaals', 'AllElectrostatics']
    for force_name in force_name_list3:
        time1 = time.time()
        group = force_groups[force_name]
        force = all_forces[force_name](protein, dna, exclusions_list, force_group=group)
        # double check periodicity
        if force.usesPeriodicBoundaryConditions() != periodic:
            print(f'Error: PBC setting for {force_name} is not correct!')
            return None
        system.addForce(force)
        time2 = time.time()
        if verbose:
            print('Adding force %s takes %.6f seconds' % (force_name, time2 - time1))
'''

def remove_IDR_native_pairs(smog_native_pairs):
    # for single chromatin fiber, all the histones are listed before dsDNA
    # remove protein-protein native pairs if at least one atom is within histone tails
    # histone tails (atom index starts from 1): 1-43, 136-159, 238-257, 353-400, 488-530, 623-646, 725-744, 840-887
    tail_start_atoms = [1, 136, 238, 353, 488, 623, 725, 840]
    tail_end_atoms = [43, 159, 257, 400, 530, 646, 744, 887]
    histone_tails = []
    for i in range(len(tail_start_atoms)):
        histone_tails += list(range(tail_start_atoms[i], tail_end_atoms[i] + 1))
    histone_tails = np.array(histone_tails) - 1 # make atom index starts from 0
    new_smog_native_pairs = pd.DataFrame(columns=smog_native_pairs.columns)
    for _, row in smog_native_pairs.iterrows():
        aai, aaj = int(row['aai']), int(row['aaj'])
        # we assume histones are before DNA
        flag1 = ((aai % 974) not in histone_tails)
        flag2 = ((aaj % 974) not in histone_tails)
        flag3 = ((aai // 974) == (aaj // 974)) # also remove native pairs between different histones
        if flag1 and flag2 and flag3:
            new_smog_native_pairs.loc[len(new_smog_native_pairs.index)] = row
    return new_smog_native_pairs


def remove_inter_histone_native_pairs(smog_native_pairs):
    # remove redundant native pairs between two different native pairs
    new_smog_native_pairs = pd.DataFrame(columns=smog_native_pairs.columns)
    for _, row in smog_native_pairs.iterrows():
        aai, aaj = int(row['aai']), int(row['aaj'])
        flag = ((aai // 974) == (aaj // 974))
        if flag:
            new_smog_native_pairs.loc[len(new_smog_native_pairs.index)] = smog_native_pairs
    return new_smog_native_pairs


def remove_IDR_dihedrals(smog_dihedrals):
    # for single chromatin fiber, all the histones are listed before dsDNA
    # remove histone dihedrals if at least one atom is within histone tails
    # histone tails (atom index starts from 1): 1-43, 136-159, 238-257, 353-400, 488-530, 623-646, 725-744, 840-887
    tail_start_atoms = [1, 136, 238, 353, 488, 623, 725, 840]
    tail_end_atoms = [43, 159, 257, 400, 530, 646, 744, 887]
    histone_tails = []
    for i in range(len(tail_start_atoms)):
        histone_tails += list(range(tail_start_atoms[i], tail_end_atoms[i] + 1))
    histone_tails = np.array(histone_tails) - 1 # make atom index starts from 0
    new_smog_dihedrals = pd.DataFrame(columns=smog_dihedrals.columns)
    for _, row in smog_dihedrals.iterrows():
        aai, aaj, aak, aal = int(row['aai']), int(row['aaj']), int(row['aak']), int(row['aal'])
        # we assume histones are before DNA
        flag1 = ((aai % 974) not in histone_tails)
        flag2 = ((aaj % 974) not in histone_tails)
        flag3 = ((aak % 974) not in histone_tails)
        flag4 = ((aal % 974) not in histone_tails)
        flag5 = ((aai // 974) == (aaj // 974))
        flag6 = ((aai // 974) == (aak // 974))
        flag7 = ((aai // 974) == (aal // 974))
        if flag1 and flag2 and flag3 and flag4 and flag5 and flag6 and flag7:
            new_smog_dihedrals.loc[len(new_smog_dihedrals.index)] = row
    return new_smog_dihedrals


def load_dna_seq_file(dna_seq_file):
    # the first line in dna_seq_file has the number of bps
    # the second line in dna_seq_file has the target sequence
    with open(dna_seq_file, 'r') as reader:
        lines = reader.readlines()
        n_bp = int(lines[0].strip())
        dna_seq = lines[1].strip()
    if n_bp != len(dna_seq):
        print('DNA sequence length and the sequence are not consistent!')
        return None
    return n_bp, dna_seq


def update_cg_dna_seq(cg_dna, dna_seq):
    if dna_seq is None:
        return None
    n_bp = len(dna_seq)
    # clean residue index
    n_atoms = len(cg_dna.index)
    cg_dna.index = list(range(n_atoms))
    old_resSeq = cg_dna['resSeq'].tolist()
    new_resSeq = []
    chainID = cg_dna['chainID'].tolist()
    if n_atoms >= 1:
        n_res = 1
        i = 1
        for j in range(n_atoms):
            if j >= 1:
                if (old_resSeq[j - 1] != old_resSeq[j]) or (chainID[j - 1] != chainID[j]):
                    i += 1
                    n_res += 1
            new_resSeq.append(i)
    cg_dna['resSeq'] = new_resSeq
    assert n_res == 2*n_bp
    # get the full sequence
    paired_dna_seq = ''
    paired_base_dict = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    base_element_dict = {'A':'N', 'G':'C', 'C':'O', 'T':'S'}
    for i in range(n_bp):
        paired_dna_seq += paired_base_dict[dna_seq[i]]
    overall_dna_seq = dna_seq + paired_dna_seq[::-1]
    # replace CG base atoms
    # we need to change name, resname, element, and type
    new_cg_dna = cg_dna.copy()
    for i in range(len(new_cg_dna.index)):
        resSeq_i = int(new_cg_dna.loc[i, 'resSeq']) # resSeq starts from 1
        base_i = overall_dna_seq[resSeq_i - 1]
        if cg_dna.loc[i, 'name'] in ['A', 'G', 'C', 'T']:
            new_cg_dna.loc[i, 'name'] = base_i
            new_cg_dna.loc[i, 'type'] = base_i
            new_cg_dna.loc[i, 'element'] = base_element_dict[base_i]
        new_cg_dna.loc[i, 'resname'] = 'D' + base_i
    return new_cg_dna


'''
def write_exclusions_list(exclusions_list, output_file, header='# i j'):
    # exclusions_list is a list composed of tuples
    # keep using openmm atom index format (i.e. atom index starts from 0)
    with open(output_file, 'w') as output_writer:
        output_writer.write(header + '\n')
        for each in exclusions_list:
            output_writer.write('%d %d\n' % (each[0], each[1]))


def load_exclusions_list(input_file, comment='#', skiprows=0):
    # input_file should use openmm atom index format (i.e. atom index starts from 0)
    with open(input_file, 'r') as input_reader:
        input_file_lines = input_reader.readlines()
    input_file_lines = input_file_lines[skiprows:]
    exclusions_list = []
    for each_line in input_file_lines:
        if each_line[:len(comment)] != comment:
            row = each_line.split()
            row = [int(each) for each in row]
            exclusions_list.append((row[0], row[1]))
    return exclusions_list


def write_rigid_body_identity(rigid_body_identity, output_file, header=None):
    with open(output_file, 'w') as output_writer:
        if header is not None:
            output_writer.write(header + '\n')
        for each in rigid_body_identity:
            if each is None:
                output_writer.write('-1\n')
            else:
                output_writer.write('%d\n' % int(each))


def load_rigid_body_identity(input_file, comment='#', skiprows=0, non_rigid_index=-1):
    with open(input_file, 'r') as input_reader:
        input_file_lines = input_reader.readlines()
    input_file_lines = input_file_lines[skiprows:]
    rigid_body_identity = []
    for each_line in input_file_lines:
        if each_line[:len(comment)] != comment:
            row = each_line.split()
            if len(row) == 1:
                i = int(row[0])
                if i == non_rigid_index:
                    rigid_body_identity.append(None)
                else:
                    rigid_body_identity.append(i)
            else:
                print('error in input file format!')
    return rigid_body_identity


def extend_rigid_body_identity(single_fiber_rigid_identity, n_fibers, n_nucl_each_fiber):
    fibers_rigid_identity = []
    for i in range(n_fibers):
        for each in single_fiber_rigid_identity:
            if each is None:
                fibers_rigid_identity.append(each)
            else:
                fibers_rigid_identity.append(each + i*n_nucl_each_fiber)
    return fibers_rigid_identity


def get_rigid_body_list_from_rigid_body_identity(rigid_body_identity):
    n_cg_atoms = len(rigid_body_identity)
    identity_set = set(rigid_body_identity)
    identity_dict = {}
    for each_identity in identity_set:
        identity_dict[each_identity] = []
    for i in range(n_cg_atoms):
        rigid_i = rigid_body_identity[i]
        identity_dict[rigid_i].append(i)
    rigid_body_list = []
    for key in identity_dict:
        if key is not None:
            rigid_body_list.append(identity_dict[key])
    return rigid_body_list


def extend_exclusions(single_fiber_exclusions_list, n_fibers, n_cg_atoms_each_fiber):
    exclusions_list = []
    for i in range(n_fibers):
        for each in single_fiber_exclusions_list:
            j, k = each[0] + i*n_cg_atoms_each_fiber, each[1] + i*n_cg_atoms_each_fiber
            exclusions_list.append((j, k))
    return exclusions_list
'''
 

def write_openmm_coord_xyz(coord, cg_fibers, xyz_file):
    # write coordinate into .xyz file for vmd visualization
    # cg_fibers is the pandas DataFrame with column 'name' saving CG atom names
    name_list = cg_fibers['name']
    if len(name_list) != coord.shape[0]:
        print('input coordinate size and atom name list size are not consistent!')
        return None
    n_atoms = len(name_list)
    with open(xyz_file, 'w') as output_writer:
        output_writer.write('%d\n' % n_atoms)
        output_writer.write('generated by openFiberTools\n')
        for i in range(n_atoms):
            name = name_list[i]
            x = coord[i, 0]/unit.angstrom
            y = coord[i, 1]/unit.angstrom
            z = coord[i, 2]/unit.angstrom
            output_writer.write('%s %.10f %.10f %.10f\n' % (name, x, y, z))


'''
def add_topo_to_fibers_dna_from_single_fiber_dna(dna, single_fiber_dna_topo_dict, n_fibers, n_cg_atoms_each_fiber):
    # add bonds, angles, stackings and dihedrals to the multi-fiber system from single fiber topology data
    # this function plays similar role as function ff3SPN2.computeTopology
    # load single fiber dna topology data
    single_fiber_dna_bonds = single_fiber_dna_topo_dict['bond']
    single_fiber_dna_angles = single_fiber_dna_topo_dict['angle']
    single_fiber_dna_stackings = single_fiber_dna_topo_dict['stacking']
    single_fiber_dna_dihedrals = single_fiber_dna_topo_dict['dihedral']
    # add bonds
    col_names = single_fiber_dna_bonds.columns.values.tolist()
    fibers_dna_bonds = pd.DataFrame(columns=col_names)
    for i in range(n_fibers):
        fiber_i_dna_bonds = single_fiber_dna_bonds.copy()
        fiber_i_dna_bonds[['aai', 'aaj']] += i*n_cg_atoms_each_fiber
        fibers_dna_bonds = pd.concat([fibers_dna_bonds, fiber_i_dna_bonds])
    fibers_dna_bonds.reset_index(drop=True, inplace=True) # reset index as serial numbers
    dna.bonds = fibers_dna_bonds
    # add angles
    col_names = single_fiber_dna_angles.columns.values.tolist()
    fibers_dna_angles = pd.DataFrame(columns=col_names)
    for i in range(n_fibers):
        fiber_i_dna_angles = single_fiber_dna_angles.copy()
        fiber_i_dna_angles[['aai', 'aaj', 'aak', 'aax']] += i*n_cg_atoms_each_fiber
        fibers_dna_angles = pd.concat([fibers_dna_angles, fiber_i_dna_angles])
    fibers_dna_angles.reset_index(drop=True, inplace=True) # reset index as serial numbers
    dna.angles = fibers_dna_angles
    # add stackings
    col_names = single_fiber_dna_stackings.columns.values.tolist()
    fibers_dna_stackings = pd.DataFrame(columns=col_names)
    for i in range(n_fibers):
        fiber_i_dna_stackings = single_fiber_dna_stackings.copy()
        fiber_i_dna_stackings[['aai', 'aaj', 'aak']] += i*n_cg_atoms_each_fiber
        fibers_dna_stackings = pd.concat([fibers_dna_stackings, fiber_i_dna_stackings])
    fibers_dna_stackings.reset_index(drop=True, inplace=True) # reset index as serial numbers
    dna.stackings = fibers_dna_stackings
    # add dihedrals
    col_names = single_fiber_dna_dihedrals.columns.values.tolist()
    fibers_dna_dihedrals = pd.DataFrame(columns=col_names)
    for i in range(n_fibers):
        fiber_i_dna_dihedrals = single_fiber_dna_dihedrals.copy()
        fiber_i_dna_dihedrals[['aai', 'aaj', 'aak', 'aal']] += i*n_cg_atoms_each_fiber
        fibers_dna_dihedrals = pd.concat([fibers_dna_dihedrals, fiber_i_dna_dihedrals])
    fibers_dna_dihedrals.reset_index(drop=True, inplace=True) # reset index as serial numbers
    dna.dihedrals = fibers_dna_dihedrals
'''


def load_coord_from_dcd(pdb_path, dcd_path, dcd_unit=unit.angstrom):
    # the default unit for dcd file is angstrom
    # load numerical values of coordinates by setting dcd_unit=None
    u = mda.Universe(pdb_path, dcd_path)
    coord = u.atoms.positions
    if dcd_unit is not None:
        coord *= dcd_unit
    return coord


def compute_atom_pair_dist(x1, x2):
    r = (np.sum((x1 - x2)**2))**0.5
    return r


def get_fibers_coord_from_single_fiber_coord(single_fiber_coord, n_fibers, delta_r, move_to_center=True):
    # single_fiber_coord and delta_r should both include length unit
    n_atoms_in_single_fiber = single_fiber_coord.shape[0]
    multi_fiber_coord = np.zeros((n_atoms_in_single_fiber*n_fibers, 3))
    for i in range(n_fibers):
        fiber_i_coord = single_fiber_coord + delta_r*i
        multi_fiber_coord[i*n_atoms_in_single_fiber:(i + 1)*n_atoms_in_single_fiber] = fiber_i_coord/unit.angstrom
    if move_to_center:
        multi_fiber_coord -= np.mean(multi_fiber_coord, axis=0)
    multi_fiber_coord *= unit.angstrom
    return multi_fiber_coord


def pick_segments_from_xml(xml_file):
    # pick out segments of interest from input xml file
    if not os.path.exists(xml_file):
        print('%s does not exist!' % xml_file)
        return None
    
    with open(xml_file, 'r') as xml_file_reader:
        xml_file_lines = xml_file_reader.readlines()
    n_xml_file_lines = len(xml_file_lines)

    for i in range(n_xml_file_lines):
        if '<System' in xml_file_lines[i]:
            header_lines = xml_file_lines[:i + 1]
            break
    header = ''.join(header_lines)
    
    for i in range(n_xml_file_lines):
        if '<PeriodicBoxVectors>' in xml_file_lines[i]:
            start_line_index = i
        if '</PeriodicBoxVectors>' in xml_file_lines[i]:
            end_line_index = i
            break
    periodic_box_lines = xml_file_lines[start_line_index:end_line_index + 1]
    periodic_box = ''.join(periodic_box_lines)

    for i in range(n_xml_file_lines):
        if '<Particles>' in xml_file_lines[i]:
            start_line_index = i
        if '</Particles>' in xml_file_lines[i]:
            end_line_index = i
            break
    particles_lines = xml_file_lines[start_line_index:end_line_index + 1]
    particles = ''.join(particles_lines)
    
    # CMMotionRemover is considered
    for i in range(n_xml_file_lines):
        if 'CMMotionRemover' in xml_file_lines[i]:
            CMmotion = xml_file_lines[i]
            break

    # note here forces do not include CMMotionRemover
    start_line_index_list, end_line_index_list = [], []
    for i in range(n_xml_file_lines):
        if ('<Force ' in xml_file_lines[i]) and ('CMMotionRemover' not in xml_file_lines[i]):
            start_line_index_list.append(i)
        if '</Force>' in xml_file_lines[i]:
            end_line_index_list.append(i)
    if len(start_line_index_list) != len(end_line_index_list):
        print('File %s force information format has issue!' % xml_file)
        forces_list = None
    else:
        forces_list = []
        for i in range(len(start_line_index_list)):
            start_line_index = start_line_index_list[i]
            end_line_index = end_line_index_list[i]
            force_lines = xml_file_lines[start_line_index:end_line_index + 1]
            force = ''.join(force_lines)
            forces_list.append(force)
    
    return header, periodic_box, particles, CMmotion, forces_list


def combine_forces_from_xml(xml_files, output_xml_path):
    # combine the texts for individual force
    n_xml_files = len(xml_files)
    if n_xml_files == 0:
        print('No input xml file')
        return None
    header = None
    box = None
    particles = None
    CMmotion = None
    forces_list = []
    for i in range(n_xml_files):
        xml_file_i = xml_files[i]
        if os.path.exists(xml_file_i):
            header_i, box_i, particles_i, CMmotion_i, forces_list_i = pick_segments_from_xml(xml_file_i)
            if None in [header_i, box_i, particles_i, CMmotion_i, forces_list_i]:
                print('Some segment may have issue in %s' % xml_file_i)
            if header is None:
                header = header_i
            elif header_i != header:
                print('Header segment in %s is not consistent with previous input xml files' % xml_file_i)
            if box is None:
                box = box_i
            elif box_i != box:
                print('Box segment in %s is not consistent with previous input xml files' % xml_file_i)
            if particles is None:
                particles = particles_i
            elif particles_i != particles:
                print('Particles segment in %s is not consistent with previous input xml files' % xml_file_i)
            if CMmotion is None:
                CMmotion = CMmotion_i
            elif CMmotion_i != CMmotion:
                print('CMmotion in %s is not consistent with previous input xml files' % xml_file_i)
            if len(forces_list_i) >= 1:
                if len(forces_list_i) > 1:
                    print('More than 1 non-CMMotionRemover forces in %s' % xml_file_i)
                for each_force in forces_list_i:
                    forces_list.append(each_force)
            else:
                print('No non-CMMotionRemover force in %s' % xml_file_i)
        else:
            print('%s does not exist!' % xml_file_i)
    # write output combined system file
    with open(output_xml_path, 'w') as output_xml_writer:
        output_xml_writer.write(header)
        output_xml_writer.write(box)
        output_xml_writer.write(particles)
        output_xml_writer.write('\t<Constraints/>\n')
        output_xml_writer.write('\t<Forces>\n')
        output_xml_writer.write(CMmotion)
        for each_force in forces_list:
            output_xml_writer.write(each_force)
        output_xml_writer.write('\t</Forces>\n')
        output_xml_writer.write('</System>\n')


def create_cg_system_from_pdb(pdb_path, periodic=True, box_a=100, box_b=100, box_c=100, remove_cmmotion=False, remove_cmmotion_force_group=0):
    # create openmm system from pdb file
    # no need to load additional forcefield files
    # if periodic is True, then a, b, and c are the box length of each direction in unit nm
    pdb = app.PDBFile(pdb_path)
    top = pdb.getTopology()
    system = simtk.openmm.System()
    if periodic:
        # set PBC box
        box_vec_a = np.array([box_a, 0, 0])*unit.nanometer
        box_vec_b = np.array([0, box_b, 0])*unit.nanometer
        box_vec_c = np.array([0, 0, box_c])*unit.nanometer
        system.setDefaultPeriodicBoxVectors(box_vec_a, box_vec_b, box_vec_c)
    for a in top.atoms():
        atom_name = a.name
        res_name = a.residue.name
        if res_name in _proteinResidues:
            if atom_name == 'CA':
                mass = _proteinMass[res_name]
            else:
                print('Warning: atom from pdb file cannot be recognized! Set this atom mass as 0.')
                mass = 0
        elif res_name in _dnaResidues:
            if atom_name in _dnaMass.keys():
                mass = _dnaMass[atom_name]
            else:
                print('Warning: atom from pdb file cannot be recognized! Set this atom mass as 0.')
                mass = 0
        else:
            print('Warning: atom from pdb file cannot be recognized! Set this atom mass as 0.')
            mass = 0
        system.addParticle(mass)
    if remove_cmmotion:
        # add CMMotionRemover
        # system created by simtk.openmm.System() does not have CMMotionRemover by default
        force = simtk.openmm.CMMotionRemover()
        force.setForceGroup(remove_cmmotion_force_group)
        system.addForce(force)
    return system


def create_cg_system_from_df(df, periodic=True, box_a=100, box_b=100, box_c=100, remove_cmmotion=False, remove_cmmotion_force_group=0):
    # no need to load additional forcefield files
    # if periodic is True, then a, b, and c are the box length of each direction in unit nm
    system = simtk.openmm.System()
    if periodic:
        # set PBC box
        box_vec_a = np.array([box_a, 0, 0])*unit.nanometer
        box_vec_b = np.array([0, box_b, 0])*unit.nanometer
        box_vec_c = np.array([0, 0, box_c])*unit.nanometer
        system.setDefaultPeriodicBoxVectors(box_vec_a, box_vec_b, box_vec_c)
    for i, atom in df.iterrows():
        atom_name = atom['name']
        res_name = atom['resname']
        if res_name in _proteinResidues:
            if atom_name == 'CA':
                mass = _proteinMass[res_name]
            else:
                print('Warning: atom from pdb file cannot be recognized! Set this atom mass as 0.')
                mass = 0
        elif res_name in _dnaResidues:
            if atom_name in _dnaMass.keys():
                mass = _dnaMass[atom_name]
            else:
                print('Warning: atom from pdb file cannot be recognized! Set this atom mass as 0.')
                mass = 0
        else:
            print('Warning: atom from pdb file cannot be recognized! Set this atom mass as 0.')
            mass = 0
        system.addParticle(mass)
    if remove_cmmotion:
        # add CMMotionRemover
        # system created by simtk.openmm.System() does not have CMMotionRemover by default
        force = simtk.openmm.CMMotionRemover()
        force.setForceGroup(remove_cmmotion_force_group)
        system.addForce(force)
    return system


