import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
import simtk.unit as unit
import os
import pandas as pd

_ef = 1*unit.kilocalorie/unit.kilojoule  # energy scaling factor
_df = 1*unit.angstrom/unit.nanometer  # distance scaling factor
_af = 1*unit.degree/unit.radian  # angle scaling factor

__location__ = os.path.dirname(os.path.abspath(__file__))
_dnaResidues = ['DA', 'DC', 'DT', 'DG']
aa_resname_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 
                   'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']


def smog_bond_term(protein, force_group=1):
    bonds = HarmonicBondForce()
    for i, row in protein.smog_bonds.iterrows():
        aai = int(row['aai'])
        aaj = int(row['aaj'])
        r0 = float(row['r0 (nm)'])
        Kb = float(row['Kb'])*protein.smog_energy_scale
        bonds.addBond(aai, aaj, r0, Kb)
    bonds.setUsesPeriodicBoundaryConditions(protein.periodic)
    bonds.setForceGroup(force_group)
    return bonds


def smog_angle_term(protein, force_group=2):
    angles = HarmonicAngleForce()
    for i, row in protein.smog_angles.iterrows():
        aai = int(row['aai'])
        aaj = int(row['aaj'])
        aak = int(row['aak'])
        theta0 = float(row['theta0 (deg)'])*np.pi/180
        Ka = float(row['Ka'])*protein.smog_energy_scale
        angles.addAngle(aai, aaj, aak, theta0, Ka)
    angles.setUsesPeriodicBoundaryConditions(protein.periodic)
    angles.setForceGroup(force_group)
    return angles


def smog_dihedral_term(protein, force_group=3):
    dihedrals = PeriodicTorsionForce()
    for i, row in protein.smog_dihedrals.iterrows():
        aai = int(row['aai'])
        aaj = int(row['aaj'])
        aak = int(row['aak'])
        aal = int(row['aal'])
        phi0 = float(row['phi0 (deg)'])*np.pi/180
        Kd = float(row['Kd'])*protein.smog_energy_scale
        mult = int(row['mult'])
        dihedrals.addTorsion(aai, aaj, aak, aal, mult, phi0, Kd)
    dihedrals.setUsesPeriodicBoundaryConditions(protein.periodic)
    dihedrals.setForceGroup(force_group)
    return dihedrals


def smog_native_pair_term(protein, force_group=4):
    pairs = CustomBondForce('''energy;
            energy=(-epsilon*G+alpha*(1-G)/r^12-offset)*step(cutoff-r);
            offset=-epsilon*exp(-18)+alpha*(1-exp(-18))/cutoff^12;
            cutoff=mu+6*sigma;
            G=exp(-(r-mu)^2/(2*sigma^2))''')
    pairs.addPerBondParameter('epsilon')
    pairs.addPerBondParameter('mu')
    pairs.addPerBondParameter('sigma')
    pairs.addPerBondParameter('alpha')
    for i, row in protein.smog_native_pairs.iterrows():
        aai = int(row['aai'])
        aaj = int(row['aaj'])
        epsilon = float(row['epsilon'])*protein.smog_energy_scale
        mu = float(row['mu'])
        sigma = float(row['sigma'])
        alpha = float(row['alpha'])*protein.smog_energy_scale
        parameters = [epsilon, mu, sigma, alpha]
        pairs.addBond(aai, aaj, parameters)
    pairs.setUsesPeriodicBoundaryConditions(protein.periodic)
    pairs.setForceGroup(force_group)
    return pairs


'''
def nonbonded_term(protein, exclusions_list, force_group=5):
    nb = CustomNonbondedForce("sqrt(A1*A2)/(r^12)")
    nb.addPerParticleParameter("A")
    for i in range(protein.natoms):
        # protein.nres is the number of amino acids
        if (i < protein.nres): 
            A = 1.67772e-05
            nb.addParticle([A])
        else:
            A = 0.0
            nb.addParticle([A])
    # add 1-2, 1-3, 1-4 exclusions and smog exclusions
    for each in exclusions_list:
        atom1_id, atom2_id = int(each[0]), int(each[1])
        nb.addExclusion(atom1_id, atom2_id)
    
    # set PBC, cutoff, and force group
    if protein.periodic:
        nb.setNonbondedMethod(nb.CutoffPeriodic)
    else:
        nb.setNonbondedMethod(nb.CutoffNonPeriodic)
    nb.setCutoffDistance(4)
    nb.setForceGroup(force_group)
    return nb


def nonbonded_MJ_term(protein, exclusions_list, config_csv=f"{__location__}/pp_MJ.csv", force_group=14):
    # MJ potential for protein-protein pair-wise interactions
    # load configuration file
    df_pp_MJ = pd.read_csv(config_csv)
    vdwl_pairs = CustomNonbondedForce("""energy;
                 energy=4*epsilon*((sigma/r)^12-(sigma/r)^6-offset)*step(cutoff-r);
                 offset=(sigma/cutoff)^12-(sigma/cutoff)^6;
                 epsilon=epsilon_map(aa_type1, aa_type2);
                 sigma=sigma_map(aa_type1, aa_type2);
                 cutoff=cutoff_map(aa_type1, aa_type2)""")
    vdwl_pairs.addPerParticleParameter('aa_type')

    # use Discrete2DFunction to define mappings for epsilon, sigma, and cutoff
    # useful link: https://gpantel.github.io/computational-method/LJsimulation/
    # CG protein atom type index in lammps input: 15-34
    # define 21*21 matrix for the mappings
    # the 0th row and 0th column are for CG DNA atoms
    # here DNA does not have such MJ pair-wise interactions
    n_aa_types = 20
    epsilon_map = np.zeros((n_aa_types + 1, n_aa_types + 1))
    sigma_map = np.zeros((n_aa_types + 1, n_aa_types + 1))
    cutoff_map = np.zeros((n_aa_types + 1, n_aa_types + 1))
    cutoff_map += 0.01 # make sure cutoff is non-zero
    for index, row in df_pp_MJ.iterrows():
        aa_type1, aa_type2 = row['atom_type1'], row['atom_type2']
        i = aa_resname_list.index(aa_type1) + 1
        j = aa_resname_list.index(aa_type2) + 1
        epsilon_map[i, j] = row['epsilon (kj/mol)']
        epsilon_map[j, i] = epsilon_map[i, j]
        sigma_map[i, j] = row['sigma (nm)']
        sigma_map[j, i] = sigma_map[i, j]
        cutoff_map[i, j] = row['cutoff_LJ (nm)']
        cutoff_map[j, i] = cutoff_map[i, j]
    max_cutoff = np.amax(cutoff_map)
    
    # Discrete2DFunction manual: http://docs.openmm.org/7.4.0/api-c++/generated/OpenMM.Discrete2DFunction.html
    # According to Discrete2DFunction manual, in general the 2D mapping matrix should be flattened along columns
    # epsilon_map, sigma_map, and cutoff_map are all symmetric matrices 
    # For symmetric matrix, flattening along rows or columns are equivalent
    epsilon_map = epsilon_map.ravel().tolist()
    sigma_map = sigma_map.ravel().tolist()
    cutoff_map = cutoff_map.ravel().tolist()
    vdwl_pairs.addTabulatedFunction('epsilon_map', Discrete2DFunction(n_aa_types + 1, n_aa_types + 1, epsilon_map))
    vdwl_pairs.addTabulatedFunction('sigma_map', Discrete2DFunction(n_aa_types + 1, n_aa_types + 1, sigma_map))
    vdwl_pairs.addTabulatedFunction('cutoff_map', Discrete2DFunction(n_aa_types + 1, n_aa_types + 1, cutoff_map))

    # add aa_type for each CG atom
    aa_type_list = []
    for i in range(protein.natoms):
        resname_i = protein.atoms.loc[i, 'resname']
        if resname_i in aa_resname_list:
            aa_type = aa_resname_list.index(resname_i) + 1
        else:
            aa_type = 0 # for DNA
        vdwl_pairs.addParticle([aa_type])
        aa_type_list.append(aa_type)
    
    # add exclusions
    for each in exclusions_list:
        atom1_id, atom2_id = int(each[0]), int(each[1])
        vdwl_pairs.addExclusion(atom1_id, atom2_id)
    
    # set PBC, cutoff, and force group
    if protein.periodic:
        vdwl_pairs.setNonbondedMethod(vdwl_pairs.CutoffPeriodic)
    else:
        vdwl_pairs.setNonbondedMethod(vdwl_pairs.CutoffNonPeriodic)
    vdwl_pairs.setCutoffDistance(max_cutoff) # note cutoff has been encoded with step function
    vdwl_pairs.setForceGroup(force_group)
    
    return vdwl_pairs
'''


def combined_DD_PD_vdwl_PP_MJ_term(protein_dna, config_csv=f"{__location__}/pp_MJ.csv", cutoff_PD=1.425*unit.nanometer, 
                                   force_group=11):
    # combine DNA-DNA and protein-DNA Van der Waals interactions with protein-protein MJ potential
    df_pp_MJ = pd.read_csv(config_csv)
    vdwl_pairs = CustomNonbondedForce('''energy;
                 energy=4*epsilon*((sigma/r)^12-(sigma/r)^6-offset)*step(cutoff-r);
                 offset=(sigma/cutoff)^12-(sigma/cutoff)^6;
                 epsilon=epsilon_map(atom_type1, atom_type2);
                 sigma=sigma_map(atom_type1, atom_type2);
                 cutoff=cutoff_map(atom_type1, atom_type2)''')
    vdwl_pairs.addPerParticleParameter('atom_type')
    # use Discrete2DFunction to define mappings for epsilon, sigma, and cutoff
    # useful link: https://gpantel.github.io/computational-method/LJsimulation/
    # 20 types of CA atoms for 20 amino acids
    dna_atom_types = ['P', 'S', 'A', 'T', 'G', 'C']
    protein_atom_types = aa_resname_list # use amino acid names to represent protein atom types
    atom_types = dna_atom_types + protein_atom_types
    n_dna_atom_types = len(dna_atom_types)
    n_atom_types = len(atom_types)
    epsilon_map = np.zeros((n_atom_types, n_atom_types))
    sigma_map = np.zeros((n_atom_types, n_atom_types))
    cutoff_map = np.zeros((n_atom_types, n_atom_types))
    
    # add DNA-DNA interactions
    param_DD = protein_dna.particle_definition[protein_dna.particle_definition['DNA'] == protein_dna.DNAtype].copy()
    param_DD.index = param_DD['name'] # rearrange to make sure the row order is based on dna_atom_names
    param_DD = param_DD.loc[dna_atom_types]
    param_DD.index = list(range(len(param_DD.index)))
    for i in range(n_dna_atom_types):
        for j in range(i, n_dna_atom_types):
            epsilon_i = param_DD.loc[i, 'epsilon']
            epsilon_j = param_DD.loc[j, 'epsilon']
            epsilon_map[i, j] = ((epsilon_i*epsilon_j)**0.5)*_ef 
            epsilon_map[j, i] = epsilon_map[i, j]
            sigma_i = param_DD.loc[i, 'radius']
            sigma_j = param_DD.loc[j, 'radius']
            sigma_map[i, j] = 0.5*(sigma_i + sigma_j)*(2**(-1/6))*_df # be careful with sigma!
            sigma_map[j, i] = sigma_map[i, j]
            cutoff_map[i, j] = 0.5*(sigma_i + sigma_j)*_df
            cutoff_map[j, i] = cutoff_map[i, j]
    
    # add protein-protein MJ potential
    for _, row in df_pp_MJ.iterrows():
        atom_type1, atom_type2 = row['atom_type1'], row['atom_type2']
        i = atom_types.index(atom_type1)
        j = atom_types.index(atom_type2)
        epsilon_map[i, j] = row['epsilon (kj/mol)']
        epsilon_map[j, i] = epsilon_map[i, j]
        sigma_map[i, j] = row['sigma (nm)']
        sigma_map[j, i] = sigma_map[i, j]
        cutoff_map[i, j] = row['cutoff_LJ (nm)']
        cutoff_map[j, i] = cutoff_map[i, j]

    # add protein-DNA interactions
    # we need to pick out some rows from protein_dna.config['Protein-DNA particles']
    all_param_PD = protein_dna.config['Protein-DNA particles']
    param_dna_PD = all_param_PD[(all_param_PD['molecule'] == 'DNA') & (all_param_PD['DNA'] == protein_dna.DNAtype)]
    param_dna_PD.index = param_dna_PD['name']
    param_dna_PD = param_dna_PD.loc[dna_atom_types]
    param_dna_PD.index = list(range(len(param_dna_PD.index)))
    param_protein_PD =  all_param_PD[(all_param_PD['molecule'] == 'Protein')]
    param_protein_PD.index = param_protein_PD['name']
    param_protein_PD = param_protein_PD.loc[['CA']] # protein only has CA type CG atom
    param_protein_PD.index = list(range(len(param_protein_PD.index)))
    param_PD = pd.concat([param_dna_PD, param_protein_PD], ignore_index=True)
    for i in range(n_dna_atom_types):
        # view all the protein CG atoms as CA here, so no need to loop over protein atoms
        epsilon_i = param_PD.loc[i, 'epsilon']
        epsilon_j = param_PD.loc[n_dna_atom_types, 'epsilon']
        epsilon_map[i, n_dna_atom_types:] = ((epsilon_i*epsilon_j)**0.5)*_ef
        epsilon_map[n_dna_atom_types:, i] = epsilon_map[i, n_dna_atom_types:]
        sigma_i = param_PD.loc[i, 'radius']
        sigma_j = param_PD.loc[n_dna_atom_types, 'radius']
        sigma_map[i, n_dna_atom_types:] = 0.5*(sigma_i + sigma_j)*_df
        sigma_map[n_dna_atom_types:, i] = sigma_map[i, n_dna_atom_types:]
        if cutoff_PD is None:
            cutoff_i = param_PD.loc[i, 'cutoff']
            cutoff_j = param_PD.loc[n_dna_atom_types, 'cutoff']
            cutoff_map[i, n_dna_atom_types:] = ((cutoff_i*cutoff_j)**0.5)*_df
            cutoff_map[n_dna_atom_types:, i] = cutoff_map[i, n_dna_atom_types:]
        else:
            cutoff_map[i, n_dna_atom_types:] = cutoff_PD.value_in_unit(unit.nanometer)
            cutoff_map[n_dna_atom_types:, i] = cutoff_map[i, n_dna_atom_types:]
    max_cutoff = np.amax(cutoff_map)
    epsilon_map = epsilon_map.ravel().tolist()
    sigma_map = sigma_map.ravel().tolist()
    cutoff_map = cutoff_map.ravel().tolist()
    vdwl_pairs.addTabulatedFunction('epsilon_map', Discrete2DFunction(n_atom_types, n_atom_types, epsilon_map))
    vdwl_pairs.addTabulatedFunction('sigma_map', Discrete2DFunction(n_atom_types, n_atom_types, sigma_map))
    vdwl_pairs.addTabulatedFunction('cutoff_map', Discrete2DFunction(n_atom_types, n_atom_types, cutoff_map))

    # add per particle parameter
    atoms = protein_dna.atoms.copy()
    for i, row in atoms.iterrows():
        if row['resname'] in _dnaResidues:
            atom_type_i = atom_types.index(row['name'])
        elif row['resname'] in aa_resname_list:
            atom_type_i = atom_types.index(row['resname'])
        else:
            print(f'Error: type of atom {i} cannot be recognized!')
            return None
        vdwl_pairs.addParticle([atom_type_i])
    
    # add exclusions
    for _, row in protein_dna.exclusions.iterrows():
        vdwl_pairs.addExclusion(int(row['aai']), int(row['aaj']))
    
    # set PBC, cutoff, and force group
    if protein_dna.periodic:
        vdwl_pairs.setNonbondedMethod(vdwl_pairs.CutoffPeriodic)
    else:
        vdwl_pairs.setNonbondedMethod(vdwl_pairs.CutoffNonPeriodic)
    vdwl_pairs.setCutoffDistance(max_cutoff)
    vdwl_pairs.setForceGroup(force_group)
    
    return vdwl_pairs


def debye_huckel_term(protein, C=150*unit.millimolar, T=300*unit.kelvin, force_group=18, 
                      elec_cutoff=3.1415044*unit.nanometer):
    dielectric = 78
    # Debye length
    Na = unit.AVOGADRO_CONSTANT_NA  # Avogadro number
    ec = 1.60217653E-19*unit.coulomb  # proton charge
    pv = 8.8541878176E-12*unit.farad/unit.meter  # dielectric permittivity of vacuum
    kb = unit.BOLTZMANN_CONSTANT_kB  # Bolztmann constant
    ldby = (dielectric*pv*kb*T/(2.0*Na*(ec**2)*C))**0.5
    denominator = 4*np.pi*pv*dielectric/(Na*(ec**2))
    denominator = denominator.value_in_unit(unit.kilojoule_per_mole**-1*unit.nanometer**-1)
    dh = CustomNonbondedForce(f"q1*q2*exp(-r/ldby)/({denominator}*r)")
    dh.addPerParticleParameter('q')
    dh.addGlobalParameter('ldby', ldby)

    # OpenMM requires the nonbonded parameters added to every atom of the system
    for i, row in protein.atoms.iterrows():
        # Need to separate between protein and DNA data
        resname = row['resname']
        name = row['name']
        if (name == 'CA') and (resname in ['ARG', 'LYS']):
            charge = 1
        elif (name == 'CA') and (resname in ['ASP', 'GLU']):
            charge = -1
        else:
            charge = 0
        parameters = [charge]
        dh.addParticle(parameters)
        
    # add exclusions
    for i, row in protein.exclusions.iterrows():
        dh.addExclusion(int(row['aai']), int(row['aaj']))
    
    # set PBC, cutoff, and force group
    if protein.periodic:
        dh.setNonbondedMethod(dh.CutoffPeriodic)
    else:
        dh.setNonbondedMethod(dh.CutoffNonPeriodic)
    dh.setCutoffDistance(elec_cutoff)
    dh.setForceGroup(force_group)
    
    return dh


