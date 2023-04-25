import numpy as np
import sys
import os
import pandas as pd
import glob

start_protein_index = 1
end_protein_index = 1948

df_list_pair = pd.read_csv('snapshot0_pair.csv')

sum_list_epair = 0 # sum of protein native contact energy
sum_dd_evdwl = 0
sum_pp_evdwl = 0
sum_dp_evdwl = 0
sum_dd_ecoul = 0
sum_pp_ecoul = 0
sum_dp_ecoul = 0

for a in range(len(df_list_pair.index)):
    row = df_list_pair.loc[a]
    pair_type = row['type']
    i, j = int(row['i']), int(row['j'])
    if pair_type == 'list':
        epair = float(row['epair (kcal/mol)'])
        sum_list_epair += epair
    if pair_type == 'lj/cut/coul/debye':
        evdwl = float(row['evdwl (kcal/mol)'])
        ecoul = float(row['ecoul (kcal/mol)'])
        flag1 = ((i >= start_protein_index) and (i <= end_protein_index))
        flag2 = ((j >= start_protein_index) and (j <= end_protein_index))
        if flag1 and flag2:
            # protein-protein interactions
            sum_pp_evdwl += evdwl
            sum_pp_ecoul += ecoul
        elif (not flag1) and (not flag2):
            # DNA-DNA interactions
            sum_dd_evdwl += evdwl
        else:
            # DNA-protein interactions
            sum_dp_evdwl += evdwl
            sum_dp_ecoul += ecoul
    if pair_type == '3spn2':
        ecoul = float(row['ecoul (kcal/mol)'])
        sum_dd_ecoul += ecoul

print('Protein native pair energy is %.6f kcal/mol' % sum_list_epair)
print('Protein-protein nonbonded energy is %.6f kcal/mol' % sum_pp_evdwl)
print('Protein-protein electrostatic energy is %.6f kcal/mol' % sum_pp_ecoul)
print('DNA-protein nonbonded energy is %.6f kcal/mol' % sum_dp_evdwl)
print('DNA-protein electrostatic energy is %.6f kcal/mol' % sum_dp_ecoul)
print('DNA-DNA nonbonded energy is %.6f kcal/mol' % sum_dd_evdwl)
print('DNA-DNA electrostatic energy is %.6f kcal/mol' % sum_dd_ecoul)

