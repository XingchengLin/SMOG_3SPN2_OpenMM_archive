import numpy as np
import sys
import os
import pandas as pd
import glob

# analyze on-the-fly output energy
# only analyze the energies of the beginning snapshot
# use "run 0" in lammps input file to run 0 step
# since there are so many output pairs, we need to design a fast algorithm
# collect data into csv files
# to save space, for the pairs, we only save pairs with non-zero evdwl or ecoul or epair

slurm_output_path = glob.glob('sim/slurm*out')[0]
#print(slurm_output_path)

df_list_pair = pd.DataFrame({'type': [], 'i': [], 'j': [], 'evdwl (kcal/mol)': [], 'ecoul (kcal/mol)': [], 'epair (kcal/mol)': []})

with open(slurm_output_path, 'r') as slurm_output:
    slurm_output_lines = slurm_output.readlines()
n_slurm_output_lines = len(slurm_output_lines)

i = 0
while i < n_slurm_output_lines:
    line_i = slurm_output_lines[i]
    if line_i[:4] == 'pair':
        row = line_i.split()
        pair_type = row[1]
        name = row[2]
        atom1, atom2 = int(row[3]) + 1, int(row[4]) + 1
        if pair_type == 'list' and name == 'epair':
            epair = float(row[5])
            if epair != 0:
                df_list_pair.loc[len(df_list_pair.index)] = [pair_type, atom1, atom2, None, None, epair]
        if pair_type == 'lj/cut/coul/debye' and name == 'evdwl':
            evdwl = float(row[5])
            i += 1
            next_row = slurm_output_lines[i].split()
            next_pair_type = next_row[1]
            next_name = next_row[2]
            next_atom1, next_atom2 = int(next_row[3]) + 1, int(next_row[4]) + 1
            if next_pair_type == pair_type and next_name == 'ecoul' and next_atom1 == atom1 and next_atom2 == atom2:
                ecoul = float(next_row[5])
                if evdwl == 0 and ecoul == 0:
                    pass
                else:
                    epair = evdwl + ecoul
                    df_list_pair.loc[len(df_list_pair.index)] = [pair_type, atom1, atom2, evdwl, ecoul, epair]
        if pair_type == '3spn2' and name == 'ecoul':
            ecoul = float(row[5])
            evdwl = None
            epair = None
            df_list_pair.loc[len(df_list_pair.index)] = [pair_type, atom1, atom2, evdwl, ecoul, epair] 
    i += 1

df_list_pair.to_csv('snapshot0_pair.csv', index=False)       

