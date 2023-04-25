import sys
import numpy as np
import pandas as pd
import simtk.openmm
import simtk.unit as unit
from openmmplumed import PlumedForce
import os
import glob
import shutil
import time
import math
import argparse
pd.set_option("display.precision", 10)

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env_main_dir', default=None, help='the directory where openSMOG3SPN2 is saved')
parser.add_argument('--system', required=True, help='path for the input system xml file')
parser.add_argument('--pdb', required=True, help='input pdb path for computing topology')
parser.add_argument('--state', default=None, help='input state xml file path')
parser.add_argument('--coord', default=None, help='coordinate in unit nm')
parser.add_argument('--move_COM_to_box_center', action='store_true', help='move center of mass to the box center')
parser.add_argument('-o', '--output_dir', required=True, help='output directory')
parser.add_argument('-p', '--platform', default='CPU', choices=['Reference', 'CPU', 'CUDA', 'OpenCL'], help='set platform')
parser.add_argument('--precision', default='mixed', choices=['single', 'mixed', 'double'], help='precision for using CUDA or OpenCL')
parser.add_argument('--temp', default=300.0, type=float, help='temperature for constant temperature simulation in unit K')
parser.add_argument('--simulation_mode', default='const_temp', choices=['const_temp', 'annealing'],  help='simulation mode, constant temperature or annealing')
parser.add_argument('--start_temp', default=320.0, type=float, help='simulation annealing start temperature in unit K')
parser.add_argument('--end_temp', default=300.0, type=float, help='simulation annealing end temperature in unit K')
parser.add_argument('--n_annealing_stages', default=2, type=int, 
                    help='the number of stages for simulation annealing')
parser.add_argument('--timestep', default=10.0, type=float, help='simulation timestep in unit fs')
parser.add_argument('-m', '--minimize', action='store_true', help='do energy minimization before running simulation')
parser.add_argument('-f', '--report_freq', default=0, type=int, 
                    help='simulation report frequency, if it is set as 0, then the report frequency is max([int(n_steps/10), 1])')
parser.add_argument('--steps', type=int, default=100, help='the number of simulation steps')
parser.add_argument('--integrator', default='NoseHoover', choices=['NoseHoover', 'Langevin'])
parser.add_argument('--collision', default=1.0, type=float, help='set collision frequency for NoseHooverIntegrator in unit 1/ps')
parser.add_argument('--friction', default=1.0, type=float, help='set friction coefficient for LangevinIntegrator in unit 1/ps')
parser.add_argument('--plumed', default=None, help='add plumed script')
parser.add_argument('--plumed_force_group', default=13, type=int, help='plumed force group index')
args = parser.parse_args()
print('command line args: ' + ' '.join(sys.argv))

system_xml_path = args.system
pdb_path = args.pdb
state_xml_path = args.state
coord_path = args.coord
output_dir = args.output_dir
platform_name = args.platform
simulation_mode = args.simulation_mode
start_temp, end_temp = args.start_temp*unit.kelvin, args.end_temp*unit.kelvin
if simulation_mode == 'annealing':
    print('Do simulation annealing')
    print(f'Set initial temperature as {start_temp} K')
    temperature = start_temp
elif simulation_mode == 'const_temp':
    print(f'Do constant temperature simulation at {args.temp} K')
    temperature = args.temp*unit.kelvin
n_annealing_stages = args.n_annealing_stages
timestep = args.timestep*unit.femtoseconds
n_steps = args.steps
report_freq = args.report_freq
if report_freq == 0:
    report_freq = max([int(n_steps/10), 1]) # make sure report_freq >= 1
print(f'Input system xml file: {system_xml_path}')
print(f'Input pdb file: {pdb_path}')
print(f'Input state file: {state_xml_path}')
print(f'Output directory: {output_dir}')
if not os.path.exists(output_dir):
    print('Create output directory')
    os.makedirs(output_dir)

if args.env_main_dir is None:
    ca_sbm_3spn_openmm_path = '/home/gridsan/sliu/Projects/CA_SBM_3SPN2C_OPENMM'
else:
    ca_sbm_3spn_openmm_path = args.env_main_dir
sys.path.insert(0, ca_sbm_3spn_openmm_path)

import openSMOG3SPN2.open3SPN2.ff3SPN2 as ff3SPN2
import openSMOG3SPN2.calphaSMOG.ffCalpha as ffCalpha
import openSMOG3SPN2.openFiber as openFiber

# load the combined system to check
s = simtk.openmm.XmlSerializer.deserialize(open(system_xml_path).read())

# check if the system is periodic
periodic = s.usesPeriodicBoundaryConditions()
if periodic:
    print('Use PBC')
    print('Periodic box vectors: ')
    print(s.getDefaultPeriodicBoxVectors())
else:
    print('Do not use PBC')

# add plumed force
force_groups = openFiber.force_groups
if args.plumed is not None:
    print(f'Use plumed with script: {args.plumed}')
    print('Be careful that plumed atom index starts from 1!')
    with open(args.plumed, 'r') as plumed_input:
        bias = PlumedForce(plumed_input.read())
    bias.setForceGroup(args.plumed_force_group)
    s.addForce(bias)
    force_groups['Plumed'] = args.plumed_force_group
else:
    print('Do not use plumed')

pdb = simtk.openmm.app.PDBFile(pdb_path)
top = pdb.getTopology()
# set integrator
if args.integrator == 'NoseHoover':
    print('Use NoseHooverIntegrator')
    collision_freq = args.collision/unit.picosecond
    integrator = simtk.openmm.NoseHooverIntegrator(temperature, collision_freq, timestep)
elif args.integrator == 'Langevin':
    print('Use LangevinIntegrator')
    friction_coeff = args.friction/unit.picosecond
    integrator = simtk.openmm.LangevinIntegrator(temperature, friction_coeff, timestep)
print(f'Set temperature as {temperature.value_in_unit(unit.kelvin)} K')
print(f'Set simulation timestep as {timestep.value_in_unit(unit.femtoseconds)} fs')
print(f'Use platform: {platform_name}')
platform = simtk.openmm.Platform.getPlatformByName(platform_name)
if platform_name in ['CUDA', 'OpenCL']:
    precision = args.precision
    properties = {'Precision': precision}
    print(f'Use precision: {precision}')
    simulation = simtk.openmm.app.Simulation(top, s, integrator, platform, properties)
else:
    simulation = simtk.openmm.app.Simulation(top, s, integrator, platform)

# set state
state = simtk.openmm.XmlSerializer.deserialize(open(state_xml_path).read())
simulation.context.setState(state)

# set coordinate
if coord_path is not None:
    if len(coord_path) >= 4 and coord_path[-4:] == '.npy':
        coord = np.load(coord_path)
    else:
        coord = np.loadtxt(coord_path)
    coord *= unit.nanometer
    simulation.context.setPositions(coord)

if args.move_COM_to_box_center:
    print('Move center of mass to the box center')
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    positions = np.array(state.getPositions().value_in_unit(unit.nanometer))
    n_atoms = s.getNumParticles()
    print(f'The number of atoms is: {n_atoms}')
    mass = []
    for i in range(n_atoms):
        mass.append(s.getParticleMass(i))
    mass = np.array(mass)
    box_vec1, box_vec2, box_vec3 = s.getDefaultPeriodicBoxVectors()
    box_vec1 = np.array(box_vec1.value_in_unit(unit.nanometer))
    box_vec2 = np.array(box_vec2.value_in_unit(unit.nanometer))
    box_vec3 = np.array(box_vec3.value_in_unit(unit.nanometer))
    box_center = 0.5*(box_vec1 + box_vec2 + box_vec3)
    center_of_mass = np.average(positions, axis=0, weights=mass/np.sum(mass))
    positions = positions - center_of_mass + box_center
    simulation.context.setPositions(positions*unit.nanometer)

# print all the forces
print('-----------------------------------')
print('Print all the forces in the system:')
print(s.getForces())
print('-----------------------------------')

state = simulation.context.getState(getEnergy=True)
energy_unit = unit.kilocalorie_per_mole
energy = state.getPotentialEnergy().value_in_unit(energy_unit)
print(f"The overall potential energy is {energy} kcal/mol")
for force_name in force_groups:
    group = force_groups[force_name]
    state = simulation.context.getState(getEnergy=True, groups={group})
    energy = state.getPotentialEnergy().value_in_unit(energy_unit)
    print(f'Name {force_name}, group {group}, energy = {energy} kcal/mol')
sys.stdout.flush()

if args.minimize:
    # do energy minimization
    print('Start doing energy minimization')
    start_time = time.time()
    simulation.minimizeEnergy()
    end_time = time.time()
    delta_time = end_time - start_time
    print('Energy minimized')
    print(f'Energy minimization takes {delta_time} seconds')
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(energy_unit)
    print(f"The overall potential energy is {energy} kcal/mol")
    for force_name in force_groups:
        group = force_groups[force_name]
        state = simulation.context.getState(getEnergy=True, groups={group})
        energy = state.getPotentialEnergy().value_in_unit(energy_unit)
        print(f'Name {force_name}, group {group}, energy = {energy} kcal/mol')
else:
    print('Do not perform energy minimization')
sys.stdout.flush()

# Reset velocity
print(f'Reset velocities based on temperature {temperature.value_in_unit(unit.kelvin)} K')
simulation.context.setVelocitiesToTemperature(temperature)

dcd_reporter = simtk.openmm.app.DCDReporter(f'{output_dir}/output.dcd', report_freq)
simulation.reporters.append(dcd_reporter)
energy_reporter = simtk.openmm.app.StateDataReporter(sys.stdout, report_freq, step=True, time=True, potentialEnergy=True, 
                                                        kineticEnergy=True, totalEnergy=True, temperature=True, speed=True)
simulation.reporters.append(energy_reporter)
start_time = time.time()
if simulation_mode == 'annealing':
    n_steps_each_stage = int(n_steps/n_annealing_stages)
    print('Do temperature annealing')
    print(f'Start temperature: {start_temp.value_in_unit(unit.kelvin)} K')
    print(f'End temperature: {end_temp.value_in_unit(unit.kelvin)} K')
    print(f'Number of annealing stages: {n_annealing_stages}')
    print(f'Number of steps in each stage: {n_steps_each_stage}')
    if n_annealing_stages > 1:
        delta_temp = (end_temp - start_temp)/(n_annealing_stages - 1)
    else:
        print('Warning: the number of annealing stages is <= 1!')
        print('Reset the number of annealing stages as 1 (equivalent to constant temperature simulation), and use the start temperature')
        n_annealing_stages = 1
        delta_temp = 0
    for i in range(n_annealing_stages):
        temp_i = start_temp + i*delta_temp
        integrator.setTemperature(temp_i)
        simulation.step(n_steps_each_stage)
elif simulation_mode == 'const_temp':
    print(f'Do constant temperature simulation at {temperature.value_in_unit(unit.kelvin)} K')
    simulation.step(n_steps)
end_time = time.time()
delta_time = end_time - start_time
print(f'The simulation takes {delta_time} seconds')

print('For the final snapshot:')
state = simulation.context.getState(getEnergy=True)
energy = state.getPotentialEnergy().value_in_unit(energy_unit)
print(f"The overall potential energy is {energy} kcal/mol")
for force_name in force_groups:
    group = force_groups[force_name]
    state = simulation.context.getState(getEnergy=True, groups={group})
    energy = state.getPotentialEnergy().value_in_unit(energy_unit)
    print(f'Name {force_name}, group {group}, energy = {energy} kcal/mol')
sys.stdout.flush()


