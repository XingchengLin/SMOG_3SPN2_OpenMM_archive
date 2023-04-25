import numpy as np
import pandas as pd
import argparse
import math
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env_main_dir', default=None, help='the directory where openSMOG3SPN2 is saved')
parser.add_argument('--n_nucl_each_fiber', required=True, type=int, help='the number of nucleosomes in each chromatin fiber')
parser.add_argument('--gap', type=float, default=0.2, help='minimal gap between two neighboring chromatins in unit nm, if each chromatin is viewed as a cuboid')
parser.add_argument('--n_fibers', type=int, required=True, help='the number of chromatin fibers')
parser.add_argument('--cubic_box_length', type=float, required=True, help='cubic box length in unit nm')
parser.add_argument('--output', required=True, help='output path for saving the coordinates of the centers of each chromatin fiber')
args = parser.parse_args()

if args.env_main_dir is None:
    ca_sbm_3spn_openmm_path = '/Users/administrator/Documents/Projects/CA_SBM_3SPN2C_OPENMM'
else:
    ca_sbm_3spn_openmm_path = args.env_main_dir
    
n_nucl_each_fiber = args.n_nucl_each_fiber
n_fibers = args.n_fibers
gap = args.gap
cubic_box_length = args.cubic_box_length
output_path = args.output

# use unit nm
# measure the size of each chromatin
single_fiber_main_output_dir = '%s/output-files/chromatin-%dmer' % (ca_sbm_3spn_openmm_path, n_nucl_each_fiber) # the main output directory for a single chromatin fiber
single_cg_fiber_unique_chainID = pd.read_csv('%s/cg-fiber/cg_fiber_unique_chainID.csv' % single_fiber_main_output_dir)
coord = single_cg_fiber_unique_chainID[['x', 'y', 'z']].to_numpy()/10 # convert to unit nm
# view each chromatin as a cuboid
a_lattice = np.amax(coord[:, 0]) - np.amin(coord[:, 0]) + 0.5*gap
b_lattice = np.amax(coord[:, 1]) - np.amin(coord[:, 1]) + 0.5*gap
c_lattice = np.amax(coord[:, 2]) - np.amin(coord[:, 2]) + 0.5*gap
n_lattice_x = int(cubic_box_length/a_lattice)
n_lattice_y = int(cubic_box_length/b_lattice)
n_lattice_z = int(cubic_box_length/c_lattice)
n_lattice = n_lattice_x*n_lattice_y*n_lattice_z
if n_lattice < n_fibers:
    print('Error: the number of fibers exceed the maximal capacity! Try to decrease min_gap, rotate the chromatin configuartion, or enlarge the box.')
    sys.exit(1)

centers = []
for i in range(n_lattice_x):
    x = (i + 0.5)*a_lattice
    for j in range(n_lattice_y):
        y = (j + 0.5)*b_lattice
        for k in range(n_lattice_z):
            z = (k + 0.5)*c_lattice
            centers.append([x, y, z])
centers = np.array(centers)
centers = centers[:n_fibers, :]
if output_path[-4:] == '.npy':
    np.save(output_path, centers)
else:
    np.savetxt(output_path, centers, fmt='%.6f')


