import numpy as np
import shutil
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='group_rigid.txt', help='input rigid group file')
parser.add_argument('--n_nucl', required=True, type=int, help='number of nucleosomes')
parser.add_argument('--output', default=None, help='output cleaned rigid group file, and by default the output will be in place')
args = parser.parse_args()

# clean rigid group file
# by default, each line begins with "group xxx id", which we need to remove
# meanwhile, the final line may be redundant
with open(args.input, 'r') as input_reader:
    input_lines = input_reader.readlines()

new_lines = []
for i in range(args.n_nucl):
    elements = input_lines[i].split()
    if elements[0] == 'group' and elements[2] == 'id':
        elements = elements[3:]
    new_lines.append(' '.join(elements) + '\n')

output_file = args.output
if output_file is None:
    output_file = args.input
    print(f'Replace input file {args.input} with output file')
    with open(output_file, 'w') as output_writer:
        for each_line in new_lines:
            output_writer.write(each_line)





