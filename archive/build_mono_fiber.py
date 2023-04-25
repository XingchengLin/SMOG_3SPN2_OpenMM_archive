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
parser.add_argument('-n', '--n_nucl', required=True, type=int, help='the number of nucleosomes')
parser.add_argument('--histone_dna_data_dir', default=None, help='the directory that saves histone dna data, note each histone has to be saved in one pdb file')
parser.add_argument('--group_rigid_txt_path', default=None, help='group_rigid.txt file with atom index starts from 1 (lammps format)')
parser.add_argument('--main_output_dir', default=None, help='main output directory path')
parser.add_argument('-s', '--scale', type=float, default=2.5, 
                    help='scale factor for protein bonds, angles, dihedrals, and native pairs')
parser.add_argument('--run_smog', action='store_true')
parser.add_argument('--smog_dir', default=None, help='directory for smog2')
parser.add_argument('--smog_output_dir', default=None, help='smog2 output directory')
parser.add_argument('--dna_seq_file', default=None, help='input dna sequence file')
parser.add_argument('--temperature', default=300.0, type=float, help='temperature in unit K, which affects dielectric')
parser.add_argument('--salt', default=150.0, type=float, help='monovalent salt concentration in unit mM, which affects electrostatic interactions')
parser.add_argument('-p', '--platform', default='CPU', choices=['Reference', 'CPU', 'CUDA', 'OpenCL'], 
                    help='set platform')
parser.add_argument('--dcd', default=None, help='input openmm dcd file')
parser.add_argument('-o', '--xml_output_dir', default=None, help='directory for saving the output xml files')
parser.add_argument('-m', '--mode', default='default', choices=['default', 'default_exclude_CA_1_4', 'default_nonrigid', 'expensive', 'compare1', 'compare2'], 
                    help='determine how to build the simulation system')
parser.add_argument('--periodic', action='store_true', help='use PBC')
parser.add_argument('--cubic_box_length', type=float, default=100.0, help='cubic box length in unit nm')
args = parser.parse_args()

if args.env_main_dir == None:
    ca_sbm_3spn_openmm_path = '/Users/administrator/Documents/Projects/CA_SBM_3SPN2C_OPENMM'
else:
    ca_sbm_3spn_openmm_path = os.path.realpath(args.env_main_dir)
sys.path.insert(0, ca_sbm_3spn_openmm_path)

import openSMOG3SPN2.open3SPN2.ff3SPN2 as ff3SPN2
import openSMOG3SPN2.calphaSMOG.ffCalpha as ffCalpha
import openSMOG3SPN2.openFiber as openFiber
import openSMOG3SPN2.openRigid as openRigid

# set some global parameters
n_nucl = args.n_nucl
scale_factor = args.scale
run_smog = args.run_smog
# smog_dir does not matter if run_smog == False
if args.smog_dir is None:
    smog_dir = '/Users/administrator/Documents/Tools/smog-2.2' # the directory where smog is installed
else:
    smog_dir = args.smog_dir
dna_seq_file = args.dna_seq_file
platform_name = args.platform
dcd_path = args.dcd
xml_output_dir = args.xml_output_dir
mode = args.mode
periodic = args.periodic
cubic_box_length = args.cubic_box_length

if mode == 'default':
    apply_rigid_body = True
    nb_exclude_complement_bp = False
    nb_exclude_native_pairs = False
    nb_exclude_CA_1_4 = False
elif mode == 'default_exclude_CA_1_4':
    apply_rigid_body = True
    nb_exclude_complement_bp = False
    nb_exclude_native_pairs = False
    nb_exclude_CA_1_4 = True
elif mode == 'default_nonrigid':
    apply_rigid_body = False
    nb_exclude_complement_bp = False
    nb_exclude_native_pairs = True
    nb_exclude_CA_1_4 = False
elif mode == 'expensive':
    apply_rigid_body = False
    nb_exclude_complement_bp = True
    nb_exclude_native_pairs = True
    nb_exclude_CA_1_4 = False
elif mode == 'compare1':
    apply_rigid_body = False
    nb_exclude_complement_bp = True
    nb_exclude_native_pairs = False
    nb_exclude_CA_1_4 = True
elif mode == 'compare2':
    apply_rigid_body = False
    nb_exclude_complement_bp = False
    nb_exclude_native_pairs = False
    nb_exclude_CA_1_4 = True
else:
    print('Error: input mode cannot be recognized!')

if periodic:
    print('Use PBC')
else:
    print('Do not use PBC')

print('The number of nucleosomes is %d' % n_nucl)
print('Protein bonds, angles, dihedrals, and native pairs scale factor = %.6f' % scale_factor)
print('Use platform: %s' % platform_name)
print('Use mode: %s' % mode)
if mode == 'compare1' or mode == 'compare2':
    print('Warning: compare1 and compare2 modes are only used for comparing with lammps results and debug!')

histone_dna_data_dir = os.path.realpath(args.histone_dna_data_dir)
if histone_dna_data_dir is None:
    # set histone_dna_data_dir as default path
    histone_dna_data_dir = '%s/data/chromatin-%dmer/separate-%dmer-output' % (ca_sbm_3spn_openmm_path, n_nucl, n_nucl) 
group_rigid_txt_path = os.path.realpath(args.group_rigid_txt_path)
if group_rigid_txt_path is None:
    # set group_rigid_txt_path as default path
    group_rigid_txt_path = '%s/data/chromatin-%dmer/chromatin-%dmer-rigid-group/group_rigid.txt' % (ca_sbm_3spn_openmm_path, n_nucl, n_nucl)
main_output_dir = args.main_output_dir
if main_output_dir is None:
    # set main_output_dir as default path
    main_output_dir = '%s/output-files/chromatin-%dmer' % (ca_sbm_3spn_openmm_path, n_nucl)
main_output_dir = os.path.realpath(main_output_dir)
print('Main output directory is: %s' % main_output_dir)
smog_output_dir = args.smog_output_dir
if smog_output_dir is None:
    # set smog_output_dir as default path
    smog_output_dir = '%s/smog' % main_output_dir # smog output directory
smog_output_dir = os.path.realpath(smog_output_dir)
all_atom_output_dir = '%s/all-atom-fiber' % main_output_dir
cg_fiber_output_dir = '%s/cg-fiber' % main_output_dir
if xml_output_dir is None:
    if periodic:
        xml_output_dir = '%s/mode-%s-%dmM-%dK-PBC-box-%.2fnm-init-system-state' % (main_output_dir, mode, int(args.salt), int(args.temperature), cubic_box_length)
    else:
        xml_output_dir = '%s/mode-%s-%dmM-%dK-init-system-state' % (main_output_dir, mode, int(args.salt), int(args.temperature))
print('Output xml files are saved in: %s' % os.path.realpath(xml_output_dir))

# build the output directories
if not os.path.exists(smog_output_dir):
    os.makedirs(smog_output_dir)
if not os.path.exists(all_atom_output_dir):
    os.makedirs(all_atom_output_dir)
if not os.path.exists(cg_fiber_output_dir):
    os.makedirs(cg_fiber_output_dir)
if not os.path.exists(xml_output_dir):
    os.makedirs(xml_output_dir)

# %% [markdown]
# # 1 Build the CG model for the chromatin

# %% [markdown]
# ## 1.1 Load PDB structures

# %%
# load each histone
all_histone_fix_list = []
for i in range(n_nucl):
    all_histone_fix_list.append(ff3SPN2.fixPDB('%s/histone-%d.pdb' % (histone_dna_data_dir, i + 1)))

# load dna
dna_fix = ff3SPN2.fixPDB('%s/dna.pdb' % histone_dna_data_dir)

# convert to pandas format tables that includes all the information of each histone and dna
# we use pandas table because there is no length limit for the entries
all_histone_atom_tables = []
for each in all_histone_fix_list:
    all_histone_atom_tables.append(ff3SPN2.pdb2table(each))

dna_atom_table = ff3SPN2.pdb2table(dna_fix)

# update serial for each histone and dna
for i in range(len(all_histone_atom_tables)):
    all_histone_atom_tables[i] = openFiber.change_serial_resSeq(all_histone_atom_tables[i], change_resSeq=False)
dna_atom_table = openFiber.change_serial_resSeq(dna_atom_table, change_resSeq=False)

# combine the tables for histones and DNA
complex_table = all_histone_atom_tables[0]
for i in range(1, len(all_histone_atom_tables)):
    complex_table = openFiber.combine_molecules(complex_table, all_histone_atom_tables[i], add_resSeq=False)
complex_table = openFiber.combine_molecules(complex_table, dna_atom_table, add_resSeq=False)

# write the data into csv file
complex_table.to_csv('%s/chromatin-%dmer.csv' % (all_atom_output_dir, n_nucl), index=False)


# %% [markdown]
# ## 1.2 Apply SMOG to histones

# %%
# write all the histones into a PDB file
ffCalpha.writePDB_protein(complex_table, '%s/histones.pdb' % smog_output_dir)

# add TER to the pdb file
input_pdb_path = '%s/histones.pdb' % smog_output_dir
output_pdb_path = '%s/histones_clean.pdb' % smog_output_dir
openFiber.add_TER_END_and_remove_OXT_for_pdb(input_pdb_path, output_pdb_path)

# %%
if run_smog:
    # perform smog on the clean protein pdb file
    cmd = 'source %s/configure.smog2; ' % smog_dir
    cmd = cmd + 'cd %s; ' % smog_output_dir
    sbm_aa_path = '%s/share/templates/SBM_AA' % smog_dir
    sbm_calpha_gaussian_path = '%s/share/templates/SBM_calpha+gaussian' % smog_dir
    cmd = cmd + 'smog2 -i histones_clean.pdb -t %s -tCG %s' % (sbm_aa_path, sbm_calpha_gaussian_path)
    #print(cmd)
    os.system(cmd)

# pick out sections from smog.top
cmd = 'cd %s; ' % smog_output_dir
py_get_section_script_path = '%s/openSMOG3SPN2/getSection.py' % ca_sbm_3spn_openmm_path
key_word_list = ['atoms', 'bonds', 'angles', 'dihedrals', 'pairs', 'exclusions', 'system']
for i in range(len(key_word_list) - 1):
    keyword1 = key_word_list[i]
    keyword2 = key_word_list[i + 1]
    cmd = cmd + 'python %s ./smog.top %s.dat "[ %s ]" "[ %s ]"; ' % (py_get_section_script_path, keyword1, keyword1, keyword2)
#print(cmd)
os.system(cmd)


# %% [markdown]
# ## 1.3 Load DNA and histone CG models separately and then combine them

# %%
# generate DNA and protein CG model from complex_table
cg_dna = ff3SPN2.DNA.CoarseGrain(complex_table)
cg_proteins = ffCalpha.Protein.CoarseGrain(complex_table)

# update the sequence for cg_dna
if dna_seq_file != None:
    print('Update DNA sequence from input DNA sequence file: %s' % dna_seq_file)
    n_bp, target_dna_seq = openFiber.load_dna_seq_file(dna_seq_file)
    cg_dna = openFiber.update_cg_dna_seq(cg_dna, target_dna_seq)
else:
    print('No input DNA sequence file. Use the DNA sequence from DNA pdb file.')

# combine CG histones and DNA
cg_fiber = pd.concat([cg_proteins, cg_dna], sort=False)
cg_fiber.index = list(range(len(cg_fiber.index)))
cg_fiber['serial'] = list(range(len(cg_fiber.index)))
n_cg_atoms = len(cg_fiber.index)

# change the chainID of the chromatin fiber
cg_fiber_unique_chainID = openFiber.change_unique_chainID(cg_fiber)

# save protein sequence
protein_seq_path = '%s/protein_seq.txt' % cg_fiber_output_dir
ffCalpha.save_protein_sequence(cg_fiber_unique_chainID, sequence_file=protein_seq_path)

# write cg_fiber to pdb format, which will later be loaded by openmm
# note we convert cg_fiber instead of cg_fiber_unique_chainID to pdb format, since cg_fiber_unique_chainID may have chainID length beyond the limit of pdb format
cg_fiber_pdb_path = '%s/cg_fiber.pdb' % cg_fiber_output_dir
ffCalpha.writePDB(cg_fiber, cg_fiber_pdb_path)
cg_fiber.to_csv('%s/cg_fiber.csv' % cg_fiber_output_dir, index=False)

# also save cg_fiber_unique_chainID.csv
cg_fiber_unique_chainID.to_csv('%s/cg_fiber_unique_chainID.csv' % cg_fiber_output_dir, index=False)

# %% [markdown]
# # 2 Set up OpenMM simulations

# %% [markdown]
# ## 2.1 Set up the system, protein and dna objects

# %%
cg_fiber_pdb_path = '%s/cg_fiber.pdb' % cg_fiber_output_dir
os.chdir('%s/cg-fiber' % main_output_dir)

pdb = simtk.openmm.app.PDBFile(cg_fiber_pdb_path)
coord_pdb = pdb.getPositions(asNumpy=True)
top = pdb.getTopology()
if dcd_path is not None:
    print('Use the coordinates from input dcd file: %s' % dcd_path)
    if apply_rigid_body:
        print('Warning: we will apply rigid body settings, so make sure the rigid bodies are at or close to the native configuration!')
    coord = openFiber.load_coord_from_dcd(cg_fiber_pdb_path, dcd_path)
else:
    print('No input dcd file, so use the coordinates from the pdb file')
    coord = coord_pdb

s = openFiber.create_cg_system_from_pdb(cg_fiber_pdb_path, periodic, cubic_box_length, cubic_box_length, cubic_box_length)

# check this initial system
# check to make sure this system only have CMMotionRemover force
with open('%s/test_init_system.xml' % xml_output_dir, 'w') as f:
    test_init_system_xml = simtk.openmm.XmlSerializer.serialize(s) 
    f.write(test_init_system_xml)

# %%
# create the DNA and protein objects
dna = ff3SPN2.DNA.fromCoarsePandasDataFrame(df=cg_fiber_unique_chainID, dna_type='B_curved')
with open(protein_seq_path, 'r') as ps:
    protein_seq = ps.readlines()[0].rstrip()
protein = ffCalpha.Protein.fromCoarsePandasDataFrame(df=cg_fiber_unique_chainID, sequence=protein_seq)

dna.periodic = periodic
protein.periodic = periodic

# set monovalent salt concentration and temperature
mono_salt_conc = args.salt*unit.millimolar
dna.mono_salt_conc = mono_salt_conc
protein.mono_salt_conc = mono_salt_conc
temperature = args.temperature*unit.kelvin
dna.temperature = temperature
protein.temperature = temperature

# %%
# save dna bonds, angles, and dihedrals
# dna bonds, angles and dihedral equilibrium values are based on template built by x3dna
dna.bonds.to_csv('%s/dna_bonds.csv' % cg_fiber_output_dir, index=False)
dna.angles.to_csv('%s/dna_angles.csv' % cg_fiber_output_dir, index=False)
dna.stackings.to_csv('%s/dna_stackings.csv' % cg_fiber_output_dir, index=False)
dna.dihedrals.to_csv('%s/dna_dihedrals.csv' % cg_fiber_output_dir, index=False)

# %%
# get DNA sequence
dna_seq = dna.getFullSequences()
dna_seq = ''.join(dna_seq.values)
dna_seq = dna_seq[:int(len(dna_seq)/2)]


# %% [markdown]
# ## 2.2 Set up forces

# %% [markdown]
# ### 2.2.1 Set up rigid body list and chain list

# %%
# set up rigid body identity
if apply_rigid_body:
    # create rigid identity list for the fiber
    print('Apply rigid body settings')
    rigid_body_array = np.loadtxt(group_rigid_txt_path, dtype=int) - 1 # atom index starts from 0
    rigid_body_identity = []
    for i in range(n_cg_atoms):
        rigid_i = None
        for j in range(n_nucl):
            if i in rigid_body_array[j]:
                rigid_i = j
                break
        rigid_body_identity.append(rigid_i)
    rigid_body_identity_output_path = '%s/rigid_body_identity.dat' % cg_fiber_output_dir
    openFiber.write_rigid_body_identity(rigid_body_identity, rigid_body_identity_output_path)
else:
    rigid_body_identity = [None]*n_cg_atoms


# %% [markdown]
# ### 2.2.2 Set up forces for histones and dna

# %%
# load the force parameters given by smog
#smog_atoms_file_path = '%s/atoms.dat' % smog_output_dir
smog_bonds_file_path = '%s/bonds.dat' % smog_output_dir
smog_angles_file_path = '%s/angles.dat' % smog_output_dir
smog_dihedrals_file_path = '%s/dihedrals.dat' % smog_output_dir
smog_exclusions_file_path = '%s/exclusions.dat' % smog_output_dir
smog_pairs_file_path = '%s/pairs.dat' % smog_output_dir

smog_bonds_data = openFiber.load_smog_bonds(smog_bonds_file_path)
smog_angles_data = openFiber.load_smog_angles(smog_angles_file_path)
smog_dihedrals_data = openFiber.load_smog_dihedrals(smog_dihedrals_file_path)
smog_exclusions_data = openFiber.load_smog_exclusions(smog_exclusions_file_path)
smog_pairs_data = openFiber.load_smog_pairs(smog_pairs_file_path)

# remove protein-protein native pairs if at least one atom is within histone tail
# also update smog_exclusions_data based on the new smog_pairs_data
smog_pairs_data, smog_exclusions_data = openFiber.remove_IDR_pairs_exclusions(smog_pairs_data, smog_exclusions_data)

# also remove dihedrals if at least one atom is within histone tail
smog_dihedrals_data = openFiber.remove_IDR_dihedrals(smog_dihedrals_data)

# save the new smog_pairs_data, smog_exclusions_data, smog_dihedrals_data
openFiber.write_smog_pairs(smog_pairs_data, '%s/pairs_IDR_removed.dat' % smog_output_dir)
openFiber.write_smog_exclusions(smog_exclusions_data, '%s/exclusions_IDR_removed.dat' % smog_output_dir)
openFiber.write_smog_dihedrals(smog_dihedrals_data, '%s/dihedrals_IDR_removed.dat' % smog_output_dir)

# save updated smog bonds, angles, dihedrals, and native pairs into a dictionary
smog_data = dict(bonds=smog_bonds_data, angles=smog_angles_data, dihedrals=smog_dihedrals_data, pairs=smog_pairs_data)

# set exclusions list and save
# label mode on the directory name
exclusions_output_dir = '%s/mode-%s-exclusions' % (cg_fiber_output_dir, mode)
if not os.path.exists(exclusions_output_dir):
    os.makedirs(exclusions_output_dir)
# for DNA, we need to decide if W-C pairs are excluded for nonbonded interactions
# for ff3SPN2.buildDNANonBondedExclusionsList, set OpenCLPatch=nb_exclude_complement_bp
dna_exclusions_list = ff3SPN2.buildDNANonBondedExclusionsList(dna, rigid_body_identity=rigid_body_identity, 
                                                              OpenCLPatch=nb_exclude_complement_bp)
dna_exclusions_list_output_path = '%s/dna_exclusions.dat' % exclusions_output_dir
openFiber.write_exclusions_list(dna_exclusions_list, dna_exclusions_list_output_path)

# for protein, we need to decide if native pairs are excluded for nonbonded interactions
# for ffCalpha.buildProteinNonBondedExclusionsList, set exclude_native_pairs=nb_exclude_native_pairs
protein_exclusions_list = ffCalpha.buildProteinNonBondedExclusionsList(protein, smog_exclusions_data, 
                                                                       rigid_body_identity=rigid_body_identity, 
                                                                       exclude_native_pairs=nb_exclude_native_pairs, 
                                                                       exclude_1_4=nb_exclude_CA_1_4)
protein_exclusions_list_output_path = '%s/protein_exclusions.dat' % exclusions_output_dir
openFiber.write_exclusions_list(protein_exclusions_list, protein_exclusions_list_output_path)

print('Total number of exclusions between DNA atoms is %d' % len(dna_exclusions_list))
print('Total number of exclusions between protein atoms is %d' % len(protein_exclusions_list))
sys.stdout.flush()

# add protein and DNA forces
openFiber.add_protein_dna_forces(s, protein, dna, smog_data, protein_exclusions_list, dna_exclusions_list, 
                                 rigid_body_identity, scale_factor)
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
    rigid_body_array = np.loadtxt(group_rigid_txt_path, dtype=int) - 1 # atom index starts from 0
    rigid_body_list = []
    for i in range(n_nucl):
        new_list = rigid_body_array[i].tolist()
        new_list = [int(each) for each in new_list]
        rigid_body_list.append(new_list)
    openRigid.createRigidBodies(s, coord, rigid_body_list)

# %% [markdown]
# ## 2.4 Run the simulation

# %%
integrator = simtk.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 10*unit.femtoseconds)
platform = simtk.openmm.Platform.getPlatformByName(platform_name)
simulation = simtk.openmm.app.Simulation(top, s, integrator, platform)
simulation.context.setPositions(coord)
energy_unit = unit.kilocalories_per_mole
state = simulation.context.getState(getEnergy=True)
energy = state.getPotentialEnergy().value_in_unit(energy_unit)
print("The overall potential energy is %.6f %s" % (energy, energy_unit.get_symbol()))

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
                                    getParameters=True, enforcePeriodicBox=False)

with open('%s/system.xml' % xml_output_dir, 'w') as f:
    system_xml = simtk.openmm.XmlSerializer.serialize(s) 
    f.write(system_xml)
    
with open('%s/state.xml' % xml_output_dir, 'w') as f:
    state_xml = simtk.openmm.XmlSerializer.serialize(state)
    f.write(state_xml)

print('The system and state are saved!')
sys.stdout.flush()

