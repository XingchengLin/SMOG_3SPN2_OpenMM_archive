import numpy as np
import shutil
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dna', required=True, help='input pdb file with DNA')
parser.add_argument('--all_histones', required=True, help='input pdb file with all the histones')
parser.add_argument('--n_nucl', required=True, type=int, help='the number of nucleosomes')
parser.add_argument('--output_dir', required=True, help='output directory')
args = parser.parse_args()

n_nucl = args.n_nucl
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# copy the dsDNA file into the output directory
shutil.copyfile(args.dna, f'{output_dir}/dna.pdb')

# separate each histone into a single pdb file
with open(args.all_histones, 'r') as input_reader:
    all_histones_lines = input_reader.readlines()

histone_index = 0
output_histone_file_lines = []

for i in range(n_nucl):
    output_histone_file_lines.append([])

for i in range(len(all_histones_lines)):
    line = all_histones_lines[i]
    if line[:4] == 'ATOM':
        output_histone_file_lines[histone_index].append(line)    
    if line[:3] == 'TER':
        curr_chain_id = all_histones_lines[i - 1][21]
        if curr_chain_id == 'H':
            output_histone_file_lines[histone_index].append('END\n')
            histone_index += 1
        else:
            output_histone_file_lines[histone_index].append(line)
    if line[:3] == 'END':
        output_histone_file_lines[histone_index].append('END\n')

for i in range(n_nucl):
    with open(f'{output_dir}/histone-{i+1}.pdb', 'w') as each_histone_pdb:
        for each_line in output_histone_file_lines[i]:
            each_histone_pdb.write(each_line)

