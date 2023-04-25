import numpy as np
import shutil
import sys
import os

# with the code that build chromatin fiber model, we already have a pdb file for dsDNA and a pdb file for all the histones
# we just need to separate all the histones into multiple pdb files that each pdb file includes a single histone

n_nucl = 12 # the number of nucleosomes
dna_pdb_file_path = 'multi_nucleo-v1.0-167-12-seq-1zbb/single_nucleo/build_fiber/fiber-167-12_clean.pdb'
all_histone_file_path = 'multi_nucleo-v1.0-167-12-seq-1zbb/single_nucleo/build_fiber/histone-all.pdb'

output_dir_path = 'separate-%dmer-output' % n_nucl
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)
    
# copy the dsDNA file into the output directory
shutil.copyfile(dna_pdb_file_path, '%s/dna.pdb' % output_dir_path)

# separate each histone into a single pdb file
with open(all_histone_file_path, 'r') as histone_all_pdb:
    histone_all_lines = histone_all_pdb.readlines()

histone_index = 0
output_histone_file_lines = []
for i in range(n_nucl):
    output_histone_file_lines.append([])

for i in range(len(histone_all_lines)):
    line = histone_all_lines[i]
    if line[:4] == 'ATOM':
        output_histone_file_lines[histone_index].append(line)    
    if line[:3] == 'TER':
        curr_chain_id = histone_all_lines[i - 1][21]
        if curr_chain_id == 'H':
            output_histone_file_lines[histone_index].append('END\n')
            histone_index += 1
        else:
            output_histone_file_lines[histone_index].append(line)
    if line[:3] == 'END':
        output_histone_file_lines[histone_index].append('END\n')

for i in range(n_nucl):
    with open('%s/histone-%d.pdb' % (output_dir_path, i + 1), 'w') as each_histone_pdb:
        for each_line in output_histone_file_lines[i]:
            each_histone_pdb.write(each_line) 
    


