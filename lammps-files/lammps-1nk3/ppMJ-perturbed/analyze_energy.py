import numpy as np
import sys
import os
import pandas as pd
import glob

# analyze on-the-fly output energy
# only analyze the energies of the beginning snapshot
# use "run 0" in lammps input file to run 0 step

start_protein_index = 1
end_protein_index = 63

def read_snapshot_energies(lines): # read the energies for the given snapshot
    n_lines = len(lines)
    df_bond = pd.DataFrame({'type': [], 'i': [], 'j': [], 'ebond (kcal/mol)': []})
    df_angle = pd.DataFrame({'type': [], 'i': [], 'j': [], 'k': [], 'eangle (kcal/mol)': []})
    df_dihedral = pd.DataFrame({'type': [], 'i': [], 'j': [], 'k': [], 'l': [], 'edihedral (kcal/mol)': []})
    df_pair = pd.DataFrame({'type': [], 'i': [], 'j': [], 'evdwl (kcal/mol)': [], 'ecoul (kcal/mol)': [], 'epair (kcal/mol)': []})
    for a in range(n_lines):
        row = lines[a].split()
        if 'bond' in lines[a]:
            bond_type = row[1]
            i, j = int(row[2]) + 1, int(row[3]) + 1
            if i > j:
                i, j = j, i
            ebond = float(row[4])
            df_bond.loc[len(df_bond.index)] = [bond_type, i, j, ebond]
        if 'angle' in lines[a]:
            angle_type = row[1]
            i, j, k = int(row[2]) + 1, int(row[3]) + 1, int(row[4]) + 1
            if i > k:
                i, k = k, i
            eangle = float(row[5])
            df_angle.loc[len(df_angle.index)] = [angle_type, i, j, k, eangle]
        if 'dihedral' in lines[a]:
            dihedral_type = row[1]
            i, j, k, l = int(row[2]) + 1, int(row[3]) + 1, int(row[4]) + 1, int(row[5]) + 1
            if i > l:
                i, j, k, l = l, k, j, i
            edihedral = float(row[6])
            df_dihedral.loc[len(df_dihedral.index)] = [dihedral_type, i, j, k, l, edihedral]
        if 'pair' in lines[a]:
            pair_type = row[1]
            i, j = int(row[3]) + 1, int(row[4]) + 1
            if i > j:
                i, j = j, i
            # evdwl and ecoul for the same pair are printed continuously
            if 'evdwl' in lines[a]:
                evdwl = float(row[5])
                next_row = lines[a + 1].split()
                # double check
                if next_row[:2] == row[:2] and next_row[3:5] == row[3:5] and ('ecoul' in lines[a + 1]):
                    ecoul = float(next_row[5])
                    epair = evdwl + ecoul
                    df_pair.loc[len(df_pair.index)] = [pair_type, i, j, evdwl, ecoul, epair]
                else:
                    print('issue with reading pair data')
            if 'epair' in lines[a]:
                epair = float(row[5])
                df_pair.loc[len(df_pair.index)] = [pair_type, i, j, None, None, epair]
    return df_bond, df_angle, df_dihedral, df_pair

slurm_output_path = glob.glob('sim/slurm*out')[0]
#print(slurm_output_path)

with open(slurm_output_path, 'r') as slurm_output:
    slurm_output_lines = slurm_output.readlines()
n_slurm_output_lines = len(slurm_output_lines)
for i in range(n_slurm_output_lines):
    if 'getting bdna/curv dists!' in slurm_output_lines[i]:
        start_line_index = i + 1
    if 'simulation finished' in slurm_output_lines[i]:
        end_line_index = i - 1
lines = slurm_output_lines[start_line_index:end_line_index + 1]
df_bond, df_angle, df_dihedral, df_pair = read_snapshot_energies(lines)
df_bond.to_csv('snapshot0_bond.csv', index=False)
df_angle.to_csv('snapshot0_angle.csv', index=False)
df_dihedral.to_csv('snapshot0_dihedral.csv', index=False)
df_pair.to_csv('snapshot0_pair.csv', index=False)

# compute the sum of energies for each type of bond or angle or dihedral
for bond_type in ['list', 'list/ca']:
    energy = df_bond[df_bond['type'] == bond_type]['ebond (kcal/mol)'].sum()
    print('bond type %s energy is %.6f kcal/mol' % (bond_type, energy))

for angle_type in ['stacking/3spn2', 'list', 'list/ca']:
    energy = df_angle[df_angle['type'] == angle_type]['eangle (kcal/mol)'].sum()
    print('angle type %s energy is %.6f kcal/mol' % (angle_type, energy))

for dihedral_type in ['list', 'list/ca']:
    energy = df_dihedral[df_dihedral['type'] == dihedral_type]['edihedral (kcal/mol)'].sum()
    print('dihedral type %s energy is %.6f kcal/mol' % (dihedral_type, energy))

for pair_type in ['lj/cut/coul/debye']:
    evdwl_sum = df_pair[df_pair['type'] == pair_type]['evdwl (kcal/mol)'].sum()
    ecoul_sum = df_pair[df_pair['type'] == pair_type]['ecoul (kcal/mol)'].sum()
    print('pair type %s nonbonded energy is %.6f kcal/mol' % (pair_type, evdwl_sum))
    print('pair type %s electrostatic energy is %.6f kcal/mol' % (pair_type, ecoul_sum))
    # further compute protein-protein and protein-dna nonbonded interactions
    pp_vdwl_sum, pdna_vdwl_sum, dd_vdwl_sum = 0, 0, 0
    pp_coul_sum, pdna_coul_sum, dd_coul_sum = 0, 0, 0
    for index, row in df_pair.iterrows():
        if row['type'] == pair_type:
            i, j = int(row['i']), int(row['j'])
            flag1, flag2 = False, False
            if i >= start_protein_index and i <= end_protein_index:
                flag1 = True
            if j >= start_protein_index and j <= end_protein_index:
                flag2 = True
            if flag1 and flag2: # protein-protein nonbonded interactions
                pp_vdwl_sum += row['evdwl (kcal/mol)']
                pp_coul_sum += row['ecoul (kcal/mol)']
            elif flag1 or flag2: # protein-dna nonbonded interactions
                pdna_vdwl_sum += row['evdwl (kcal/mol)']
                pdna_coul_sum += row['ecoul (kcal/mol)']
            else: # dna-dna nonbonded interactions
                dd_vdwl_sum += row['evdwl (kcal/mol)']
                dd_coul_sum += row['ecoul (kcal/mol)']
    print('pair type %s protein-protein nonbonded energy is %.6f kcal/mol' % (pair_type, pp_vdwl_sum))
    print('pair type %s protein-dna nonbonded energy is %.6f kcal/mol' % (pair_type, pdna_vdwl_sum))
    print('pair type %s dna-dna nonbonded energy is %.6f kcal/mol' % (pair_type, dd_vdwl_sum))
    print('pair type %s protein-protein electrostatic energy is %.6f kcal/mol' % (pair_type, pp_coul_sum))
    print('pair type %s protein-dna electrostatic energy is %.6f kcal/mol' % (pair_type, pdna_coul_sum))
    print('pair type %s dna-dna electrostatic energy is %.6f kcal/mol' % (pair_type, dd_coul_sum))

# compute SBM native pair interactions
epair_sum = df_pair[df_pair['type'] == 'list']['epair (kcal/mol)'].sum()
print('pair type %s (SBM native pair) energy is %.6f kcal/mol' % (pair_type, epair_sum))

