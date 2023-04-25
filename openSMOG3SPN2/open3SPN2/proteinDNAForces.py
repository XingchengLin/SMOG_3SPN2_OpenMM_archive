import simtk.openmm.app
import simtk.openmm
import simtk.unit as unit
import configparser
import numpy as np
import itertools
import os
import pdbfixer
import pandas as pd

_complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
_dnaResidues = ['DA', 'DC', 'DT', 'DG']
_proteinResidues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                    'SER', 'THR', 'TRP', 'TYR', 'VAL']
_ef = 1*unit.kilocalorie/unit.kilojoule  # energy scaling factor
_df = 1*unit.angstrom/unit.nanometer  # distance scaling factor
_af = 1*unit.degree/unit.radian  # angle scaling factor

# define protein-DNA forces as functions instead of classes
def protein_dna_excl(protein, dna, exclusions_list, k=1, force_group=15):
    excl_force = simtk.openmm.CustomNonbondedForce(f"""{k}*energy;
                         energy=(4*epsilon*((sigma/r)^12-(sigma/r)^6)-offset)*step(cutoff-r);
                         offset=4*epsilon*((sigma/cutoff)^12-(sigma/cutoff)^6);
                         sigma=0.5*(sigma1+sigma2); 
                         epsilon=sqrt(epsilon1*epsilon2);
                         cutoff=sqrt(cutoff1*cutoff2)""")
    excl_force.addPerParticleParameter('epsilon')
    excl_force.addPerParticleParameter('sigma')
    excl_force.addPerParticleParameter('cutoff')
    excl_force.setCutoffDistance(4)
    excl_force.setForceGroup(force_group)
    if dna.periodic:
        excl_force.setNonbondedMethod(excl_force.CutoffPeriodic)
    else:
        excl_force.setNonbondedMethod(excl_force.CutoffNonPeriodic)
    particle_definition = dna.config['Protein-DNA particles']
    dna_particle_definition = particle_definition[(particle_definition['molecule'] == 'DNA') &
                                                (particle_definition['DNA'] == dna.DNAtype)]
    protein_particle_definition = particle_definition[(particle_definition['molecule'] == 'Protein')]
    
    # Merge DNA and protein particle definitions
    particle_definition = pd.concat([dna_particle_definition, protein_particle_definition], ignore_index=True)
    particle_definition.index = particle_definition.molecule + particle_definition.name

    # particle definition includes information for different types of particles
    is_dna = dna.atoms['resname'].isin(_dnaResidues)
    is_protein = dna.atoms['resname'].isin(_proteinResidues)
    atoms = dna.atoms.copy()
    atoms['is_dna'] = is_dna
    atoms['is_protein'] = is_protein
    atoms['epsilon'] = np.nan
    atoms['radius'] = np.nan
    atoms['cutoff'] = np.nan
    DNA_list = []
    protein_list = []

    # add parameters
    for i, atom in atoms.iterrows():
        if atom.is_dna:
            param = particle_definition.loc['DNA' + atom['name']]
            parameters = [param.epsilon*_ef, param.radius*_df, param.cutoff*_df]
            DNA_list += [i]
        elif atom.is_protein:
            param = particle_definition.loc['Protein' + atom['name']]
            parameters = [param.epsilon*_ef, param.radius*_df, param.cutoff*_df]
            protein_list += [i]
        else:
            print(f'Residue {i} not included in protein-DNA interactions')
            parameters = [0, 0.1, 0.1]
        atoms.loc[i, ['epsilon', 'radius', 'cutoff']] = parameters
        excl_force.addParticle(parameters)
    excl_force.addInteractionGroup(DNA_list, protein_list)

    # addExclusions
    for each in exclusions_list:
        i, j = each
        excl_force.addExclusion(i, j)
    
    return excl_force


def protein_dna_elec(protein, dna, exclusions_list, k=1, force_group=16):
    dielectric = 78
    # Debye length
    Na = unit.AVOGADRO_CONSTANT_NA  # Avogadro number
    ec = 1.60217653E-19*unit.coulomb  # proton charge
    pv = 8.8541878176E-12*unit.farad/unit.meter  # dielectric permittivity of vacuum
    kb = unit.BOLTZMANN_CONSTANT_kB  # Bolztmann constant
    C = 150*unit.millimolar
    T = 300*unit.kelvin
    ldby = (dielectric*pv*kb*T/(2.0*Na*ec**2*C))**0.5
    denominator = 4*np.pi*pv*dielectric/(Na*ec**2)
    denominator = denominator.in_units_of(unit.kilocalorie_per_mole**-1*unit.nanometer**-1)

    # For the protein-DNA interactions, the DNA charge should be recovered from -0.6 to -1.0, therefore, we need to divide the protein-dna electrostatic energy by 0.6
    elec_force = simtk.openmm.CustomNonbondedForce(f"""k_electro_protein_DNA*energy/0.6;
                            energy=q1*q2*exp(-r/inter_dh_length)/inter_denominator/r;""")
    elec_force.addPerParticleParameter('q')
    elec_force.addGlobalParameter('k_electro_protein_DNA', k)
    elec_force.addGlobalParameter('inter_dh_length', ldby)
    elec_force.addGlobalParameter('inter_denominator', denominator)
    elec_force.setCutoffDistance(4)
    if dna.periodic:
        elec_force.setNonbondedMethod(elec_force.CutoffPeriodic)
    else:
        elec_force.setNonbondedMethod(elec_force.CutoffNonPeriodic)
    elec_force.setForceGroup(force_group)

    # Merge DNA and protein particle definitions
    particle_definition = dna.config['Protein-DNA particles']
    dna_particle_definition=particle_definition[(particle_definition['molecule'] == 'DNA') &
                                                (particle_definition['DNA'] == dna.DNAtype)]
    protein_particle_definition = particle_definition[(particle_definition['molecule'] == 'Protein')]

    # Merge DNA and protein particle definitions
    particle_definition = pd.concat([dna_particle_definition, protein_particle_definition], ignore_index=True)
    particle_definition.index = particle_definition.molecule + particle_definition.name

    # Open Sequence dependent electrostatics
    sequence_electrostatics = dna.config['Sequence dependent electrostatics']
    sequence_electrostatics.index = sequence_electrostatics.resname

    # Select only dna and protein atoms
    is_dna = protein.atoms['resname'].isin(_dnaResidues)
    is_protein = protein.atoms['resname'].isin(_proteinResidues)
    atoms = protein.atoms.copy()
    atoms['is_dna'] = is_dna
    atoms['is_protein'] = is_protein
    DNA_list = []
    protein_list = []

    # add parameters
    for i, atom in atoms.iterrows():
        if atom.is_dna:
            param = particle_definition.loc['DNA' + atom['name']]
            charge = param.charge
            parameters = [charge]
            if charge != 0:
                DNA_list += [i]
        elif atom.is_protein:
            atom_param = particle_definition.loc['Protein' + atom['name']]
            seq_param = sequence_electrostatics.loc[atom.real_resname]
            charge = atom_param.charge * seq_param.charge
            parameters = [charge]
            if charge != 0:
                protein_list += [i]
        else:
            print(f'Residue {i} not included in protein-DNA electrostatics')
            parameters = [0]  # No charge if it is not DNA
        elec_force.addParticle(parameters)
    elec_force.addInteractionGroup(DNA_list, protein_list)

    # addExclusions
    for each in exclusions_list:
        i, j = each
        elec_force.addExclusion(i, j)
    
    return elec_force


    