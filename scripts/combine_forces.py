import sys
import numpy as np
import pandas as pd
import simtk.openmm
import simtk.unit as unit
import os
import glob
import shutil
import time
import MDAnalysis as mda
import math
import argparse
pd.set_option("display.precision", 10)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', required=True, help='path for the directory for input xml files')
parser.add_argument('-o', '--output_xml_path', default=None, 
                    help='path for the output combined system xml file, and it will be written to input_dir if this choice is not assigned')
args = parser.parse_args()

input_dir = args.input_dir
output_xml_path = args.output_xml_path
if output_xml_path == None:
    output_xml_path = '%s/system_combined.xml' % input_dir

ca_sbm_3spn_openmm_path = '/Users/administrator/Documents/Projects/CA_SBM_3SPN2C_OPENMM'
sys.path.insert(0, ca_sbm_3spn_openmm_path)

import openSMOG3SPN2.open3SPN2.ff3SPN2 as ff3SPN2
import openSMOG3SPN2.calphaSMOG.ffCalpha as ffCalpha
import openSMOG3SPN2.openFiber as openFiber

force_names = ['BondProtein', 'AngleProtein', 'DihedralProtein', 'NativePairProtein', 'NonbondedMJ', 
               'ElectrostaticsProteinProtein', 'BondDNA', 'AngleDNA', 'Stacking', 'DihedralDNA', 'BasePair', 
               'CrossStacking', 'ExclusionDNADNA', 'ElectrostaticsDNADNA', 'ExclusionProteinDNA', 
               'ElectrostaticsProteinDNA']

# combine particles with individual force into a single xml file

# check if all the files we aim to use exist
for each_force_name in force_names:
    file_path = '%s/system_%s.xml' % (input_dir, each_force_name)
    if not os.path.exists(file_path):
        print('Warning: %s does not exist!' % file_path)

# start to combine forces
xml_files = []
for each_force_name in force_names:
    xml_files.append('%s/system_%s.xml' % (input_dir, each_force_name))
openFiber.combine_forces_from_xml(xml_files, output_xml_path)


