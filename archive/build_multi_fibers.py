# %%
import sys
import numpy as np
import pandas as pd
import simtk.openmm
import simtk.unit as unit
import os
import glob
import shutil
import time
import MDAnalysis as mda
import math
import argparse
pd.set_option("display.precision", 10)

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env_main_dir', default=None, help='the directory where openSMOG3SPN2 is saved')
parser.add_argument('--n_nucl_each_fiber', required=True, type=int, help='the number of nucleosomes in each chromatin fiber')
parser.add_argument('--n_fibers', required=True, type=int, help='the number of chromatin fibers')
parser.add_argument('-s', '--scale', default=2.5, type=float, help='scale factor for protein bonds, angles, dihedrals, and native pairs')
parser.add_argument('--temp', default=300.0, type=float, help='temperature in unit K, which affects dielectric')
parser.add_argument('--salt', default=150.0, type=float, help='monovalent salt concentration in unit mM, which affects electrostatic interactions')
parser.add_argument('-p', '--platform', default='CPU', choices=['Reference', 'CPU', 'CUDA', 'OpenCL'], 
                    help='set platform')
parser.add_argument('--fibers_dcd_path', default=None, help='input multiple fiber dcd file')
parser.add_argument('--pdb_fibers_centers', default=None, help='file for assigning the coordinates of centers of each fiber in the pdb file, and coordinates are in unit nm')
parser.add_argument('-x', '--delta_x', default=10.0, type=float, help='delta x between neighboring chromatin fibers in unit nm')
parser.add_argument('-y', '--delta_y', default=10.0, type=float, help='delta y between neighboring chromatin fibers in unit nm')
parser.add_argument('-z', '--delta_z', default=10.0, type=float, help='delta z between neighboring chromatin fibers in unit nm')
parser.add_argument('-o', '--xml_output_dir', default=None, help='directory for saving the output xml files')
parser.add_argument('-m', '--mode', default='default', choices=['default', 'default_nonrigid', 'expensive', 'compare1', 'compare2'], 
                    help='determine how to build the simulation system')
parser.add_argument('--periodic', action='store_true', help='use PBC')
parser.add_argument('--cubic_box_length', type=float, default=100.0, help='cubic box length in unit nm')
args = parser.parse_args()

if args.env_main_dir is None:
    ca_sbm_3spn_openmm_path = '/Users/administrator/Documents/Projects/CA_SBM_3SPN2C_OPENMM'
else:
    ca_sbm_3spn_openmm_path = args.env_main_dir
sys.path.insert(0, ca_sbm_3spn_openmm_path)

import openSMOG3SPN2.open3SPN2.ff3SPN2 as ff3SPN2
import openSMOG3SPN2.calphaSMOG.ffCalpha as ffCalpha
import openSMOG3SPN2.openFiber as openFiber
import openSMOG3SPN2.openRigid as openRigid

# set some global parameters
n_nucl_each_fiber = args.n_nucl_each_fiber
n_fibers = args.n_fibers
scale_factor = args.scale
platform_name = args.platform
fibers_dcd_path = args.fibers_dcd_path
pdb_fibers_centers_path = args.pdb_fibers_centers
delta_x = args.delta_x
delta_y = args.delta_y
delta_z = args.delta_z
xml_output_dir = args.xml_output_dir
mode = args.mode
periodic = args.periodic
cubic_box_length = args.cubic_box_length

print('The system has %d fibers, and each fiber includes %d nucleosomes' % (n_fibers, n_nucl_each_fiber))
print('Protein bonds, angles, dihedrals, and native pairs scale factor = %.6f' % scale_factor)
print('Use platform: %s' % platform_name)
print('Use mode: %s' % mode)
if mode == 'compare1' or mode == 'compare2':
    print('Warning: compare1 and compare2 modes are only used for comparing with lammps results and debug!')

if mode == 'default':
    apply_rigid_body = True
    nb_exclude_complement_bp = False
    nb_exclude_CA_1_4 = False
elif mode == 'default_nonrigid':
    apply_rigid_body = False
    nb_exclude_complement_bp = False
    nb_exclude_CA_1_4 = False
elif mode == 'expensive':
    apply_rigid_body = False
    nb_exclude_complement_bp = True
    nb_exclude_CA_1_4 = False
elif mode == 'compare1':
    apply_rigid_body = False
    nb_exclude_complement_bp = True
    nb_exclude_CA_1_4 = True
elif mode == 'compare2':
    apply_rigid_body = False
    nb_exclude_complement_bp = False
    nb_exclude_CA_1_4 = True
else:
    print('Error: input mode cannot be recognized!')

if periodic:
    print('Use PBC')
else:
    print('Do not use PBC')

single_fiber_group_rigid_txt_path = '%s/data/chromatin-%dmer/chromatin-%dmer-rigid-group/group_rigid.txt' % (ca_sbm_3spn_openmm_path, n_nucl_each_fiber, n_nucl_each_fiber) # group_rigid.txt file with atom index starts from 1 (lammps format)
single_fiber_main_output_dir = '%s/output-files/chromatin-%dmer' % (ca_sbm_3spn_openmm_path, n_nucl_each_fiber) # the main output directory for a single chromatin fiber
fibers_main_output_dir = '%s/output-files/chromatin-%dx%dmers' % (ca_sbm_3spn_openmm_path, n_fibers, n_nucl_each_fiber) # the main output directory for multiple chromatin fibers
cg_fibers_output_dir = '%s/cg-fibers' % fibers_main_output_dir
single_fiber_smog_output_dir = '%s/smog' % single_fiber_main_output_dir # smog output directory for single chromatin fiber
if xml_output_dir == None:
    if periodic:
        xml_output_dir = '%s/mode-%s-%dmM-%dK-PBC-box-%.2fnm-init-system-state' % (fibers_main_output_dir, mode, int(args.salt), int(args.temp), cubic_box_length)
    else:
        xml_output_dir = '%s/mode-%s-%dmM-%dK-init-system-state' % (fibers_main_output_dir, mode, int(args.salt), int(args.temp))
print('Output xml files are saved in: %s' % xml_output_dir)

# build the output directories
if not os.path.exists(single_fiber_main_output_dir):
    print('%s does not exist!' % single_fiber_main_output_dir)
if not os.path.exists(single_fiber_smog_output_dir):
    print('%s does not exist!' % single_fiber_smog_output_dir)
if not os.path.exists(cg_fibers_output_dir):
    os.makedirs(cg_fibers_output_dir)
if not os.path.exists(xml_output_dir):
    os.makedirs(xml_output_dir)

# %% [markdown]
# # 1 Build multiple fiber system from single fiber

# %% [markdown]
# ## 1.1 Load the structure of single chromatin fiber

# %%
# load the pandas dataframe of single fiber structure
single_cg_fiber_unique_chainID = pd.read_csv('%s/cg-fiber/cg_fiber_unique_chainID.csv' % single_fiber_main_output_dir)
single_cg_fiber = pd.read_csv('%s/cg-fiber/cg_fiber.csv' % single_fiber_main_output_dir)

n_cg_atoms_each_fiber = len(single_cg_fiber.index)

# %% [markdown]
# ## 1.2 Build the structure for multiple chromatin fibers

# %%
# build the pandas dataframe for multiple fibers
# build two pandas dataframes, one with unique chainID and resSeq, and one without unique chainID or resSeq
# the one without unique chainID or resSeq will be converted to pdb format and later loaded by openmm
if pdb_fibers_centers_path != None:
    if pdb_fibers_centers_path[-4:] == '.npy':
        pdb_fibers_centers = np.load(pdb_fibers_centers_path)
    else:
        pdb_fibers_centers = np.loadtxt(pdb_fibers_centers_path)
    if pdb_fibers_centers.shape[0] != n_fibers:
        print('Error: the shape of pdb_fibers_centers is not consistent with the number of fibers!')
        sys.exit(1)
    if pdb_fibers_centers.shape[1] != 3:
        print('Error: the shape of pdb_fibers_centers is not consistent with 3D system!')
        sys.exit(1)
else:
    pdb_fibers_centers = None

for i in range(n_fibers):
    cg_fiber_i = single_cg_fiber.copy()
    cg_fiber_i_unique_chainID = single_cg_fiber_unique_chainID.copy()
    if pdb_fibers_centers is None:
        coord = cg_fiber_i[['x', 'y', 'z']].to_numpy()
        coord += 10*i*np.array([delta_x, delta_y, delta_z]) # convert nm to angstrom
        cg_fiber_i[['x', 'y', 'z']] = coord
        cg_fiber_i_unique_chainID[['x', 'y', 'z']] = coord
    else:
        # the coordinates in cg_fiber_i and cg_fiber_i_unique_chainID are in unit angstrom
        coord = cg_fiber_i[['x', 'y', 'z']].to_numpy()
        coord -= np.mean(coord, axis=0)
        coord += 10*pdb_fibers_centers[i] # convert nm to angstrom
        cg_fiber_i[['x', 'y', 'z']] = coord
        cg_fiber_i_unique_chainID[['x', 'y', 'z']] = coord
    if i == 0:
        cg_fibers = cg_fiber_i
        cg_fibers_unique_chainID = cg_fiber_i_unique_chainID
    else:
        cg_fibers = openFiber.combine_molecules(cg_fibers, cg_fiber_i, add_serial=False, add_resSeq=False)
        cg_fibers_unique_chainID = openFiber.combine_molecules(cg_fibers_unique_chainID, cg_fiber_i_unique_chainID, add_resSeq=False)

cg_fibers_unique_chainID = openFiber.change_unique_chainID(cg_fibers_unique_chainID)
cg_fibers.reset_index(drop=True, inplace=True)
cg_fibers_unique_chainID.reset_index(drop=True, inplace=True)

# move global center to the box center
cubic_box_center = 0.5*np.array([cubic_box_length, cubic_box_length, cubic_box_length])
cg_fibers = openFiber.move_complex_to_center(cg_fibers, 10*cubic_box_center) # use unit angstrom
cg_fibers_unique_chainID = openFiber.move_complex_to_center(cg_fibers_unique_chainID, 10*cubic_box_center) # use unit angstrom

n_cg_atoms = len(cg_fibers.index)

# replace NaN with ''
cg_fibers_unique_chainID = cg_fibers_unique_chainID.fillna('')
cg_fibers = cg_fibers.fillna('')

cg_fibers_pdb_path = '%s/cg_fibers.pdb' % cg_fibers_output_dir
ffCalpha.writePDB(cg_fibers, cg_fibers_pdb_path)
cg_fibers_unique_chainID.to_csv('%s/cg_fibers_unique_chainID.csv' % cg_fibers_output_dir, index=False)

# %% [markdown]
# # 2 Set up OpenMM simulations

# %% [markdown]
# ## 2.1 Set up the system, protein and dna objects

# %%
os.chdir(cg_fibers_output_dir)

pdb = simtk.openmm.app.PDBFile(cg_fibers_pdb_path)
top = pdb.getTopology()
fibers_coord_pdb = pdb.getPositions(asNumpy=True)

# get position from dcd file
if fibers_dcd_path != None:
    print('Load multiple chromatin dcd file: %s' % fibers_dcd_path)
    if apply_rigid_body:
        print('Warning: we will apply rigid body settings, so make sure the rigid bodies are at or close to the native configuration!')
    fibers_coord = openFiber.load_coord_from_dcd(cg_fibers_pdb_path, fibers_dcd_path)
else:
    print('Load coordinates for multiple chromatin fibers from pdb')
    fibers_coord = fibers_coord_pdb

# save the coordinate for the multi-fiber system as xyz file
xyz_file = '%s/cg-fibers/fibers.xyz' % fibers_main_output_dir
openFiber.write_openmm_coord_xyz(fibers_coord, cg_fibers, xyz_file)

s = openFiber.create_cg_system_from_pdb(cg_fibers_pdb_path, periodic, cubic_box_length, cubic_box_length, cubic_box_length)

# %%
# create the DNA and protein objects
# set dna bonds, angles, and dihedrals from the parameters of single dsDNA
# so the original open3SPN2 code will build a long DNA with sequence composed of all the bases, though convenient, this may lead to some boundary effects
# do not use ff3SPN2 to automatically set bonds, angles, and dihedrals (i.e. set compute_topology as False, then ff3SPN2.DNA.fromCoarsePandasDataFrame will not automatically get dna bonds, angles, stackings, and dihedrals)
# load dna bonds, angles, and dihedrals manually based on single chromatin fiber dna bonds, angels, and dihedrals
start_time = time.time()
dna = ff3SPN2.DNA.fromCoarsePandasDataFrame(df=cg_fibers_unique_chainID, dna_type='B_curved', compute_topology=False, 
                                            parse_config=True)
single_fiber_dna_bonds = pd.read_csv('%s/cg-fiber/dna_bonds.csv' % single_fiber_main_output_dir)
single_fiber_dna_angles = pd.read_csv('%s/cg-fiber/dna_angles.csv' % single_fiber_main_output_dir)
single_fiber_dna_stackings = pd.read_csv('%s/cg-fiber/dna_stackings.csv' % single_fiber_main_output_dir)
single_fiber_dna_dihedrals = pd.read_csv('%s/cg-fiber/dna_dihedrals.csv' % single_fiber_main_output_dir)
single_fiber_dna_topo_dict = dict(bond=single_fiber_dna_bonds, 
                                  angle=single_fiber_dna_angles, 
                                  stacking=single_fiber_dna_stackings, 
                                  dihedral=single_fiber_dna_dihedrals)
openFiber.add_topo_to_fibers_dna_from_single_fiber_dna(dna, single_fiber_dna_topo_dict, n_fibers, n_cg_atoms_each_fiber)
end_time = time.time()
delta_time = end_time - start_time
print('Adding DNA topology takes take %.6f seconds' % delta_time)

single_fiber_protein_seq_path = '%s/cg-fiber/protein_seq.txt' % single_fiber_main_output_dir
with open(single_fiber_protein_seq_path, 'r') as ps:
    single_fiber_protein_seq = ps.readlines()[0].rstrip()
fibers_protein_seq = single_fiber_protein_seq*n_fibers

protein = ffCalpha.Protein.fromCoarsePandasDataFrame(df=cg_fibers_unique_chainID, sequence=fibers_protein_seq)

dna.periodic = periodic
protein.periodic = periodic

# set monovalent salt concentration and temperature
dna.mono_salt_conc = args.salt
protein.mono_salt_conc = args.salt
dna.temp = args.temp
protein.temp = args.temp


# %%
# create rigid identity list for the fiber
if apply_rigid_body:
    single_fiber_rigid_body_identity_file = '%s/cg-fiber/rigid_body_identity.dat' % single_fiber_main_output_dir
    single_fiber_rigid_body_identity = openFiber.load_rigid_body_identity(single_fiber_rigid_body_identity_file)
    fibers_rigid_body_identity = openFiber.extend_rigid_body_identity(single_fiber_rigid_body_identity, n_fibers, n_nucl_each_fiber)
else:
    fibers_rigid_body_identity = [None]*n_cg_atoms

# get exclusions list
# get DNA exclusions list by ff3SPN2.buildDNANonBondedExclusionsList
fibers_dna_exclusions_list = ff3SPN2.buildDNANonBondedExclusionsList(dna, rigid_body_identity=fibers_rigid_body_identity, 
                                                                     OpenCLPatch=nb_exclude_complement_bp)
# get protein exclusions list by extending single fiber protein exclusion list
single_fiber_protein_exclusions_list_file = '%s/cg-fiber/mode-%s-exclusions/protein_exclusions.dat' % (single_fiber_main_output_dir, mode)
single_fiber_protein_exclusions_list = openFiber.load_exclusions_list(single_fiber_protein_exclusions_list_file)
fibers_protein_exclusions_list = openFiber.extend_exclusions(single_fiber_protein_exclusions_list, n_fibers, n_cg_atoms_each_fiber)


# %% [markdown]
# ## 2.2 Set up forces for histones and dna

# %%
# set force dictionary
forces = {}

# load smog data
single_fiber_smog_bonds_file_path = '%s/bonds.dat' % single_fiber_smog_output_dir
single_fiber_smog_angles_file_path = '%s/angles.dat' % single_fiber_smog_output_dir
single_fiber_smog_dihedrals_file_path = '%s/dihedrals_IDR_removed.dat' % single_fiber_smog_output_dir
single_fiber_smog_exclusions_file_path = '%s/exclusions_IDR_removed.dat' % single_fiber_smog_output_dir
single_fiber_smog_pairs_file_path = '%s/pairs_IDR_removed.dat' % single_fiber_smog_output_dir

single_fiber_smog_bonds_data = openFiber.load_smog_bonds(single_fiber_smog_bonds_file_path)
single_fiber_smog_angles_data = openFiber.load_smog_angles(single_fiber_smog_angles_file_path)
single_fiber_smog_dihedrals_data = openFiber.load_smog_dihedrals(single_fiber_smog_dihedrals_file_path)
single_fiber_smog_exclusions_data = openFiber.load_smog_exclusions(single_fiber_smog_exclusions_file_path)
single_fiber_smog_pairs_data = openFiber.load_smog_pairs(single_fiber_smog_pairs_file_path)

fibers_smog_bonds_data = openFiber.extend_single_fiber_to_fibers_bonds(single_fiber_smog_bonds_data, n_fibers, n_cg_atoms_each_fiber)
fibers_smog_angles_data = openFiber.extend_single_fiber_to_fibers_angles(single_fiber_smog_angles_data, n_fibers, n_cg_atoms_each_fiber)
fibers_smog_dihedrals_data = openFiber.extend_single_fiber_to_fibers_dihedrals(single_fiber_smog_dihedrals_data, n_fibers, n_cg_atoms_each_fiber)
fibers_smog_exclusions_data = openFiber.extend_single_fiber_to_fibers_exclusions(single_fiber_smog_exclusions_data, n_fibers, n_cg_atoms_each_fiber)
fibers_smog_pairs_data = openFiber.extend_single_fiber_to_fibers_pairs(single_fiber_smog_pairs_data, n_fibers, n_cg_atoms_each_fiber)

fibers_smog_data = dict(bonds=fibers_smog_bonds_data, 
                        angles=fibers_smog_angles_data, 
                        dihedrals=fibers_smog_dihedrals_data, 
                        pairs=fibers_smog_pairs_data)

# add protein and dna forces
openFiber.add_protein_dna_forces(s, protein, dna, fibers_smog_data, fibers_protein_exclusions_list, fibers_dna_exclusions_list, 
                                 fibers_rigid_body_identity, scale_factor)
if s.usesPeriodicBoundaryConditions():
    if s.usesPeriodicBoundaryConditions() != periodic:
        print('Error: system periodicity is not consistent with the setting!')
        sys.exit(1)
    print('Periodic box vectors: ')
    print(s.getDefaultPeriodicBoxVectors())
sys.stdout.flush()

# %% [markdown]
# ## 2.3 Set up rigid body

# %%

if apply_rigid_body:
    print('Apply rigid body settings to the system')
    fibers_rigid_body_list = openFiber.get_rigid_body_list_from_rigid_body_identity(fibers_rigid_body_identity)
    openRigid.createRigidBodies(s, fibers_coord, fibers_rigid_body_list)


# %% [markdown]
# ## 2.4 Run the simulation

# %%
temperature = 300*unit.kelvin
integrator = simtk.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 10*unit.femtoseconds)
platform = simtk.openmm.Platform.getPlatformByName(platform_name)
simulation = simtk.openmm.app.Simulation(top, s, integrator, platform)
simulation.context.setPositions(fibers_coord)
energy_unit = unit.kilocalories_per_mole
state = simulation.context.getState(getEnergy=True)
energy = state.getPotentialEnergy().value_in_unit(energy_unit)
print("The overall potential energy is %.6f %s" % (energy, energy_unit.get_symbol()))

# get the detailed energy after the simulation
# double check SBM pair, nonbonded, and electrostatic interactions
for force_name in openFiber.force_groups:
    group = openFiber.force_groups[force_name]
    state = simulation.context.getState(getEnergy=True, groups={group})
    energy = state.getPotentialEnergy()
    print('Name %s, group %d, energy = %.6f %s' % (force_name, group, energy.value_in_unit(energy_unit), energy_unit.get_symbol()))
sys.stdout.flush()

# %%
# save the system and the state
# system.xml contains all of the force field parameters
state = simulation.context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True, 
                                    getParameters=True, enforcePeriodicBox=periodic)

with open('%s/system.xml' % xml_output_dir, 'w') as f:
    system_xml = simtk.openmm.XmlSerializer.serialize(s) 
    f.write(system_xml)
    
with open('%s/state.xml' % xml_output_dir, 'w') as f:
    state_xml = simtk.openmm.XmlSerializer.serialize(state)
    f.write(state_xml)

print('The system and state are saved!')
sys.stdout.flush()

