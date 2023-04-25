import numpy as np

n_nucl = 12

with open('group_rigid_lammps.txt', 'r') as group_rigid_lammps:
    group_rigid_lammps_lines = group_rigid_lammps.readlines()

with open('group_rigid.txt', 'w') as group_rigid:
    for each_line in group_rigid_lammps_lines[:n_nucl]:
        elements = each_line.split()
        elements = elements[3:]
        group_rigid.write(' '.join(elements))
        group_rigid.write('\n')

