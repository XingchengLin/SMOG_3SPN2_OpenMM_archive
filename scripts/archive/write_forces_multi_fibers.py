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
parser.add_argument('-n', '--n_nucl_each_fiber', required=True, type=int, help='the number of nucleosomes in each chromatin fiber')
parser.add_argument('-f', '--n_fibers', required=True, type=int, help='the number of chromatin fibers')
parser.add_argument('-s', '--scale', default=2.5, type=float, help='scale factor for protein bonds, angles, dihedrals, and native pairs')
parser.add_argument('-p', '--platform', default='CPU', choices=['Reference', 'CPU', 'CUDA', 'OpenCL'], 
                    help='set platform')
parser.add_argument('-r', '--rigid', action='store_true', help='apply rigid body')
parser.add_argument('-c', '--compare_with_lammps', action='store_true', 
                    help='MJ potential and electrostatic interactions for protein native pairs are computed')
parser.add_argument('--single_fiber_dcd_path', default=None, help='input single fiber dcd file')
parser.add_argument('-d', '--dcd', default=None, help='input multiple fiber dcd file')
parser.add_argument('-x', '--delta_x', default=100.0, type=float, help='delta x between neighboring chromatin fibers in unit angstrom')
parser.add_argument('-y', '--delta_y', default=100.0, type=float, help='delta y between neighboring chromatin fibers in unit angstrom')
parser.add_argument('-z', '--delta_z', default=100.0, type=float, help='delta z between neighboring chromatin fibers in unit angstrom')
args = parser.parse_args()

ca_sbm_3spn_openmm_path = '/Users/administrator/Documents/Projects/CA_SBM_3SPN2C_OPENMM'
sys.path.insert(0, ca_sbm_3spn_openmm_path)

import openSMOG3SPN2.open3SPN2.ff3SPN2 as ff3SPN2
import openSMOG3SPN2.calphaSMOG.ffCalpha as ffCalpha
import openSMOG3SPN2.openFiber as openFiber

# set some global parameters
n_nucl_each_fiber = args.n_nucl_each_fiber
n_fibers = args.n_fibers
scale_factor = args.scale
platform_name = args.platform
apply_rigid_body = args.rigid
single_fiber_dcd_path = args.single_fiber_dcd_path
fibers_dcd_path = args.dcd
delta_x = args.delta_x
delta_y = args.delta_y
delta_z = args.delta_z
compare_with_lammps = args.compare_with_lammps

ffCalpha_xml_path = '%s/openSMOG3SPN2/calphaSMOG/ffCalpha.xml' % ca_sbm_3spn_openmm_path
single_fiber_group_rigid_txt_path = '%s/data/chromatin-%dmer/chromatin-%dmer-rigid-group/group_rigid.txt' % (ca_sbm_3spn_openmm_path, n_nucl_each_fiber, n_nucl_each_fiber) # group_rigid.txt file with atom index starts from 1 (lammps format)
single_fiber_main_output_dir = '%s/output-files/chromatin-%dmer' % (ca_sbm_3spn_openmm_path, n_nucl_each_fiber) # the main output directory for a single chromatin fiber
fibers_main_output_dir = '%s/output-files/chromatin-%dx%dmers' % (ca_sbm_3spn_openmm_path, n_fibers, n_nucl_each_fiber) # the main output directory for multiple chromatin fibers
single_fiber_smog_output_dir = '%s/smog' % single_fiber_main_output_dir # smog output directory for single chromatin fiber
openmm_files_dir = '%s/openmm-files' % fibers_main_output_dir
sim_output_dir = '%s/sim-test-%s' % (openmm_files_dir, platform_name)
init_system_state_dir = '%s/init-system-state' % fibers_main_output_dir

# build the output directories
if not os.path.exists(single_fiber_main_output_dir):
    print('%s does not exist!' % single_fiber_main_output_dir)
if not os.path.exists(single_fiber_smog_output_dir):
    print('%s does not exist!' % single_fiber_smog_output_dir)
if not os.path.exists('%s/cg-fibers' % fibers_main_output_dir):
    os.makedirs('%s/cg-fibers' % fibers_main_output_dir)
if not os.path.exists(init_system_state_dir):
    os.makedirs(init_system_state_dir)

# %% [markdown]
# # 1 Build multiple fiber system from single fiber

# %% [markdown]
# ## 1.1 Load the structure of single chromatin fiber

# %%
# load the pandas dataframe of single fiber structure
single_cg_fiber_unique_chainID = pd.read_csv('%s/cg-fiber/cg_fiber_unique_chainID.csv' % single_fiber_main_output_dir)
single_cg_fiber = pd.read_csv('%s/cg-fiber/cg_fiber.csv' % single_fiber_main_output_dir)

n_cg_atoms_each_fiber = single_cg_fiber_unique_chainID.shape[0]

# %% [markdown]
# ## 1.2 Build the structure for multiple chromatin fibers

# %%
# build the pandas dataframe for multiple fibers
# build two pandas dataframes, one with unique chainID and resSeq, and one without unique chainID or resSeq
# the one without unique chainID or resSeq will be converted to pdb format and later loaded by openmm
delta_r = np.array([delta_x, delta_y, delta_z])*unit.angstrom
cg_fibers_unique_chainID = single_cg_fiber_unique_chainID.copy()
cg_fibers = single_cg_fiber.copy()
for i in range(1, n_fibers):
    cg_fiber_i_unique_chainID = single_cg_fiber_unique_chainID.copy()
    cg_fiber_i_unique_chainID['x'] += i*delta_x
    cg_fiber_i_unique_chainID['y'] += i*delta_y
    cg_fiber_i_unique_chainID['z'] += i*delta_z
    cg_fibers_unique_chainID = openFiber.combine_molecules(cg_fibers_unique_chainID, cg_fiber_i_unique_chainID, add_resSeq=False)
    cg_fiber_i = single_cg_fiber.copy()
    cg_fiber_i['x'] += i*delta_x
    cg_fiber_i['y'] += i*delta_y
    cg_fiber_i['z'] += i*delta_z
    cg_fibers = openFiber.combine_molecules(cg_fibers, cg_fiber_i, add_serial=False, add_resSeq=False)

# move center to (0, 0, 0)
cg_fibers = openFiber.move_complex_to_center(cg_fibers)
cg_fibers_unique_chainID = openFiber.move_complex_to_center(cg_fibers_unique_chainID)

cg_fibers_unique_chainID = openFiber.change_unique_chainID(cg_fibers_unique_chainID)
cg_fibers_unique_chainID.index = list(range(len(cg_fibers_unique_chainID.index)))
cg_fibers.index = list(range(len(cg_fibers.index)))

n_cg_atoms = len(cg_fibers.index)

# replace NaN with ''
cg_fibers_unique_chainID = cg_fibers_unique_chainID.fillna('')
cg_fibers = cg_fibers.fillna('')

cg_fibers_pdb_path = '%s/cg-fibers/cg_fibers.pdb' % fibers_main_output_dir
ffCalpha.writePDB(cg_fibers, cg_fibers_pdb_path)
cg_fibers_unique_chainID.to_csv('%s/cg-fibers/cg_fibers_unique_chainID.csv' % fibers_main_output_dir, index=False)

# %% [markdown]
# # 2 Set up OpenMM simulations

# %% [markdown]
# ## 2.1 Set up the system, protein and dna objects

# %%
os.chdir('%s/cg-fibers' % fibers_main_output_dir)

pdb = simtk.openmm.app.PDBFile(cg_fibers_pdb_path)
top = pdb.getTopology()
#coord_pdb = pdb.getPositions(asNumpy=True)

# get position from dcd file
if fibers_dcd_path != None:
    print('Load multiple chromatin dcd file: %s' % fibers_dcd_path)
    fibers_coord = openFiber.load_coord_from_dcd(cg_fibers_pdb_path, fibers_dcd_path)
elif single_fiber_dcd_path != None:
    print('Load single chromatin dcd file: %s' % single_fiber_dcd_path)
    # start from single fiber coordinate
    single_cg_fiber_pdb_path = '%s/cg-fiber/cg_fiber.pdb' % single_fiber_main_output_dir
    single_fiber_coord = openFiber.load_coord_from_dcd(single_cg_fiber_pdb_path, single_fiber_dcd_path)
    print('Get the coordinates for multiple chromatin fibers from single chromatin fiber')
    # extend single fiber coordinate to mutliple fibers
    fibers_coord = openFiber.get_fibers_coord_from_single_fiber_coord(single_fiber_coord, n_fibers, delta_r)
else:
    print('Load coordinates for multiple chromatin fibers from pdb')
    coord_pdb = pdb.getPositions(asNumpy=True)
    fibers_coord = coord_pdb
    

# save the coordinate for the multi-fiber system as xyz file
xyz_file = '%s/cg-fibers/fibers_coord_openmm.xyz' % fibers_main_output_dir
openFiber.write_openmm_coord_xyz(fibers_coord, cg_fibers, xyz_file)

forcefield = simtk.openmm.app.ForceField(ffCalpha_xml_path, ff3SPN2.xml)
s = forcefield.createSystem(top)

# %%
# create the DNA and protein objects
# set dna bonds, angles, and dihedrals from the parameters of single dsDNA
# so the original open3SPN2 code will build a long DNA with sequence composed of all the bases, though convenient, this may lead to some boundary effects
# do not use ff3SPN2 to automatically set bonds, angles, and dihedrals (i.e. set compute_topology as False, then ff3PNS2.DNA.fromCoarsePDB_thorugh_pdframe will not automatically get dna bonds, angles, stackings, and dihedrals)
# load dna bonds, angles, and dihedrals manually based on single chromatin fiber dna bonds, angels, and dihedrals
print('start loading single fiber topology, then adding topology to multiple fiber system')
start_time = time.time()
dna = ff3SPN2.DNA.fromCoarsePandasDataFrame(pd_df=cg_fibers_unique_chainID, dna_type='B_curved', compute_topology=False, parse_config=True)
single_fiber_dna_bonds = pd.read_csv('%s/cg-fiber/dna_bonds.csv' % single_fiber_main_output_dir)
single_fiber_dna_angles = pd.read_csv('%s/cg-fiber/dna_angles.csv' % single_fiber_main_output_dir)
single_fiber_dna_stackings = pd.read_csv('%s/cg-fiber/dna_stackings.csv' % single_fiber_main_output_dir)
single_fiber_dna_dihedrals = pd.read_csv('%s/cg-fiber/dna_dihedrals.csv' % single_fiber_main_output_dir)
single_fiber_dna_topo_dict = dict(bond=single_fiber_dna_bonds, 
                                  angle=single_fiber_dna_angles, 
                                  stacking=single_fiber_dna_stackings, 
                                  dihedral=single_fiber_dna_dihedrals)
openFiber.add_topo_to_fibers_from_single_fiber_dna(dna, single_fiber_dna_topo_dict, n_fibers, n_cg_atoms_each_fiber)
end_time = time.time()
delta_time = end_time - start_time
print('finish loading single fiber topology and adding topology to multiple fiber system')
print('loading single fiber topology and adding topology to multiple fiber system take %.6f seconds' % delta_time)

single_fiber_protein_seq_path = '%s/cg-fiber/protein_seq.txt' % single_fiber_main_output_dir
with open(single_fiber_protein_seq_path, 'r') as ps:
    single_fiber_protein_seq = ps.readlines()[0].rstrip()
fibers_protein_seq = single_fiber_protein_seq*n_fibers

protein = ffCalpha.Protein.fromCoarsePandasDataFrame(pd_df=cg_fibers_unique_chainID, sequence=fibers_protein_seq)

dna.periodic = False
protein.periodic = False


# %%
# create rigid identity list for the fiber
if apply_rigid_body:
    pass # to be fulfilled
else:
    fibers_rigid_identity = [None]*n_cg_atoms

# get exclusions list
print('start getting exclusions list')
start_time = time.time()
single_fiber_dna_exclusions_list = openFiber.load_exclusions_list('%s/cg-fiber/dna_exclusions.dat' % single_fiber_main_output_dir)
if compare_with_lammps:
    # if compare with lammps, then openmm needs to compute electrostatic and MJ potential for protein native pairs
    print('Compare with lammps, so we need to compute electrostatic and MJ potential for protein native pairs')
    single_fiber_protein_exclusions_list_file = '%s/cg-fiber/protein_exclusions_compare_with_lammps.dat' % single_fiber_main_output_dir
    single_fiber_protein_exclusions_list = openFiber.load_exclusions_list(single_fiber_protein_exclusions_list_file)
else:
    single_fiber_protein_exclusions_list_file = '%s/cg-fiber/protein_exclusions.dat' % single_fiber_main_output_dir
    single_fiber_protein_exclusions_list = openFiber.load_exclusions_list(single_fiber_protein_exclusions_list_file)

fibers_dna_exclusions_list = ff3SPN2.buildDNANonBondedExclusionsList(dna) # since there are exclusions between W-C paired basepairs, we cannot simply generalize exclusions from single fiber DNA exclusions
fibers_protein_exclusions_list = openFiber.extend_exclusions(single_fiber_protein_exclusions_list, n_fibers, n_cg_atoms_each_fiber)
end_time = time.time()
delta_time = end_time - start_time
print('finish getting exclusions list')
print('getting exclusions list takes %.6f seconds' % delta_time)
print('total number of exclusions between DNA atoms is %d' % len(fibers_dna_exclusions_list))
print('total number of exclusions between protein atoms is %d' % len(fibers_protein_exclusions_list))


# %% [markdown]
# ## 2.2 Set up forces for histones and dna

# %%

# load smog data
print('start loading smog data')
start_time = time.time()
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
end_time = time.time()
delta_time = end_time - start_time
print('finish loading smog data')
print('loading smog data takes %.6f seconds' % delta_time)
sys.stdout.flush()

# write protein dna interactions 
print('start writing each protein and dna force separately')
start_time = time.time()
forces = {}
openFiber.write_protein_dna_each_force(forcefield, top, forces, dna, protein, fibers_smog_data, fibers_dna_exclusions_list, fibers_protein_exclusions_list, fibers_rigid_identity, scale_factor, init_system_state_dir)
end_time = time.time()
delta_time = end_time - start_time
print('finish writing each protein and dna force separately')
print('writing each protein and dna force takes %.6f seconds' % delta_time)
print('All the forces are saved individually')
sys.stdout.flush()


