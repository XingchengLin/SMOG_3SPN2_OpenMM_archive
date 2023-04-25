import numpy as np

# ----------Parameters----------
nrl = 167
num_nucl = 2
# ------------------------------

num_bp = nrl * (num_nucl - 1) + 147 # the total number of bps
path_data = 'data.prot_dna' # the lammps input file

data = open(path_data,'r')
data_lines = data.readlines()
n = len(data_lines)

for i in range(n):
    line = data_lines[i]
    if line[:5] == 'Atoms':
        k = i + 2
        break

# data_lines[k + 974 * num_nucl:k + 974 * num_nucl + num_bp * 3 - 1] are the information of the first ssDNA
data_seq = ''
g_group = [5,9,13]
a_group = [3,7,11]
t_group = [4,8,12]
c_group = [6,10,14]
for i in range(k + 974 * num_nucl, k + 974 * num_nucl + num_bp * 3 - 1):
    line = data_lines[i]
    values = line.split()
    if int(values[1]) == num_nucl * 8 + 1: # double check if the chain index is correct
        if int(values[3]) in g_group:
            data_seq += 'G'
        if int(values[3]) in a_group:
            data_seq += 'A'
        if int(values[3]) in t_group:
            data_seq += 'T'
        if int(values[3]) in c_group:
            data_seq += 'C'
    else:
        print('wrong alignment')
        break
data.close()

# check sequence length
if len(data_seq) != num_bp:
    print('wrong number of bps')

# load ../buildDna/dnaSeq.txt
dnaSeq = open('../buildDna/dnaSeq.txt','r')
input_seq = dnaSeq.readlines()[1][:num_bp]
input_seq = input_seq.upper()
dnaSeq.close()

if input_seq == data_seq:
    print('the sequence is correct')
else:
    print('wrong sequence')

        
    


