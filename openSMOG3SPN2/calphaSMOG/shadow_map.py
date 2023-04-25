import numpy as np
import pandas as pd
import simtk.unit as unit
import simtk.openmm
import simtk.openmm.app as app
import mdtraj
import MDAnalysis
import networkx as nx
import math
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))

__author__ = 'Shuming Liu'

def get_neighbor_pairs_and_distances(coord, cutoff=0.6, box=None, pbc=False):
    # follow: https://docs.mdanalysis.org/1.1.1/documentation_pages/lib/nsgrid.html
    # coord and cutoff should use the same length unit
    # if pbc is False, then the code can automatically produce box
    # if pbc is True, then specify box as np.array([lx, ly, lz, alpha1, alpha2, alpha3])
    if pbc:
        grid_search = MDAnalysis.lib.nsgrid.FastNS(cutoff, coord.astype(np.float32), box.astype(np.float32), pbc)
    else:
        x_min, x_max = np.amin(coord[:, 0]), np.amax(coord[:, 0])
        y_min, y_max = np.amin(coord[:, 1]), np.amax(coord[:, 1])
        z_min, z_max = np.amin(coord[:, 2]), np.amax(coord[:, 2])
        shifted_coord = coord.copy() - np.array([x_min, y_min, z_min])
        shifted_coord = shifted_coord.astype(np.float32)
        lx = 1.1*(x_max - x_min)
        ly = 1.1*(y_max - y_min)
        lz = 1.1*(z_max - z_min)
        pseudo_box = np.array([lx, ly, lz, 90, 90, 90]).astype(np.float32)
        grid_search = MDAnalysis.lib.nsgrid.FastNS(cutoff, shifted_coord, pseudo_box, pbc)
    results = grid_search.self_search()
    neighbor_pairs = results.get_pairs()
    neighbor_pair_distances = results.get_pair_distances()
    return neighbor_pairs, neighbor_pair_distances


def get_bonded_neighbor_dict(atomistic_pdb):
    traj = mdtraj.load_pdb(atomistic_pdb)
    top = traj.topology
    bond_graph = top.to_bondgraph()
    bonded_neighbor_dict = {}
    for a1 in list(bond_graph.nodes):
        bonded_neighbor_dict[a1.index] = []
        for a2 in bond_graph.neighbors(a1):
            bonded_neighbor_dict[a1.index].append(a2.index)
    return bonded_neighbor_dict
        

def find_res_pairs_from_atomistic_pdb(atomistic_pdb, frame=0, radius=0.1, bonded_radius=0.05, cutoff=0.6, box=None, pbc=False):
    # find all the native pairs between residues following the shadow algorithm
    # the contacts are searched based on all the heavy atoms in the atomistic model
    # the outputs are contacts between residues
    # the output residue index starts from 0
    # radius is the shadow radius for each atom 
    # when testing whether atom 3 blocks the light between atom 1 and 2, if atom 3 is bonded to atom 1 or 2, then atom 3 uses bonded_radius as its own radius
    # cutoff is the contact cutoff distance
    # radius and cutoff are both in unit nm, consistent with mdtraj length unit
    traj = mdtraj.load_pdb(atomistic_pdb)
    top = traj.topology
    n_atoms = top.n_atoms
    df_atoms, _bonds = top.to_dataframe()
    df_atoms.index = list(range(len(df_atoms.index)))
    # add unique_resSeq, which starts from 0
    unique_resSeq = 0
    for i, row in df_atoms.iterrows():
        if i >= 1:
            if (row['resSeq'] != df_atoms.loc[i - 1, 'resSeq']) or (row['chainID'] != df_atoms.loc[i - 1, 'chainID']):
                unique_resSeq += 1
        df_atoms.loc[i, 'unique_resSeq'] = unique_resSeq
    neighbors_no_hyd_dict = {}
    for i in range(n_atoms):
        if df_atoms.loc[i, 'element'] != 'H':
            neighbors_no_hyd_dict[i] = []
    coord = traj.xyz[frame]
    neighbor_atom_pairs, neighbor_atom_pair_distances = get_neighbor_pairs_and_distances(coord, cutoff, box, pbc) # find spatially close neighbors
    dist_matrix = np.zeros((n_atoms, n_atoms))
    dist_matrix[:, :] = -1 # initialize distance matrix elements as -1
    beyond_1_4_neighbors_no_hyd_atom_pairs = []
    for i in range(len(neighbor_atom_pairs)):
        a1, a2 = int(neighbor_atom_pairs[i, 0]), int(neighbor_atom_pairs[i, 1])
        if a1 > a2:
            a1, a2 = a2, a1
        # save the distances of all the neighboring pairs into dist_matrix
        dist_matrix[a1, a2] = neighbor_atom_pair_distances[i]
        dist_matrix[a2, a1] = neighbor_atom_pair_distances[i]
        if (df_atoms.loc[a1, 'element'] != 'H') and (df_atoms.loc[a2, 'element'] != 'H'):
            neighbors_no_hyd_dict[a1].append(a2)
            neighbors_no_hyd_dict[a2].append(a1)
            chain1, chain2 = df_atoms.loc[a1, 'chainID'], df_atoms.loc[a2, 'chainID']
            resSeq1, resSeq2 = df_atoms.loc[a1, 'resSeq'], df_atoms.loc[a2, 'resSeq']
            if (chain1 != chain2) or abs(resSeq1 - resSeq2) > 3:
                # find residue pairs that do not have bonded interactions in CG model
                beyond_1_4_neighbors_no_hyd_atom_pairs.append([a1, a2])
    res_pairs = []
    bonded_neighbor_dict = get_bonded_neighbor_dict(atomistic_pdb)
    for each in beyond_1_4_neighbors_no_hyd_atom_pairs:
        a1, a2 = each[0], each[1]
        unique_resSeq1, unique_resSeq2 = df_atoms.loc[a1, 'unique_resSeq'], df_atoms.loc[a2, 'unique_resSeq']
        # since a1 < a2 and they are beyond 1-4 interactions, thus unique_resSeq1 < unique_resSeq2
        if [unique_resSeq1, unique_resSeq2] in res_pairs:
            continue
        d12 = dist_matrix[a1, a2]
        if d12 < 0:
            sys.exit('Error, distance smaller than 0!')
        if d12 < radius:
            print(f'Distance between atom {a1} and {a2} is {d12} nm, which is smaller than the radius ({radius} nm), so we ignore this atom pair')
            print(f'This means maybe the radius is too large or atoms {a1} and {a2} are too close')
            continue
        flag = True
        # test if atom a3 blocks the contact between atom a1 and a2
        # only test atom a3 if d13 < d12 and d23 < d12
        # a1 and a2 have contact, d12 < cutoff, so d13 < cutoff and d23 < cutoff
        # so atom a3 has to be the neighbor of both atom a1 and a2
        block_candidates = [a3 for a3 in neighbors_no_hyd_dict[a1] if a3 in neighbors_no_hyd_dict[a2]]
        for a3 in block_candidates:
            # check if a3 blocks the light from a1 to a2 or the light from a2 to a1
            # (a1, a3) and (a2, a3) are both neighboring pairs, so d13 and d23 can be directly read from dist_matrix
            d13, d23 = dist_matrix[a1, a3], dist_matrix[a2, a3]
            # double check to make sure d13 and d23 can be read correctly from dist_matrix
            if d13 < 0 or d23 < 0:
                sys.exit('Error, distance smaller than 0!')
            if d12 <= d13 or d12 <= d23:
                continue
            if (a3 in bonded_neighbor_dict[a1]) or (a3 in bonded_neighbor_dict[a2]):
                radius3 = bonded_radius
            else:
                radius3 = radius
            # if radius3 is larger than d13 or d23, then recognize that a3 blocks the contact between a1 and a2
            if radius3 > d13:
                print(f'Distance between atom {a1} and {a3} is {d13} nm, which is smaller than the radius ({radius3} nm)')
                print(f'Recognize that {a3} blocks the contact between {a1} and {a2}')
                print(f'This means maybe the radius is too large or atoms {a1} and {a3} are too close')
                flag = False
                break
            if radius3 > d23:
                print(f'Distance between atom {a2} and {a3} is {d23} nm, which is smaller than the radius ({radius3} nm)')
                print(f'Recognize that {a3} blocks the contact between {a1} and {a2}')
                print(f'This means maybe the radius is too large or atoms {a2} and {a3} are too close')
                flag = False
                break
            # check if the light from a1 to a2 is blocked by a3
            angle213 = mdtraj.compute_angles(traj, np.array([[a2, a1, a3]]), pbc)[frame, 0]
            theta12 = math.asin(radius/d12)
            theta13 = math.asin(radius3/d13)
            if theta12 + theta13 >= angle213:
                flag = False
                break
            # check if the light from a2 to a1 is blocked by a3
            angle123 = mdtraj.compute_angles(traj, np.array([[a1, a2, a3]]), pbc)[frame, 0]
            theta12 = math.asin(radius/d12)
            theta23 = math.asin(radius3/d23)
            if theta12 + theta23 >= angle123:
                flag = False
                break
        if flag:
            res_pairs.append([unique_resSeq1, unique_resSeq2])
    res_pairs = np.array(sorted(res_pairs))
    return res_pairs, df_atoms


def find_ca_pairs_from_atomistic_pdb(atomistic_pdb, frame=0, radius=0.1, bonded_radius=0.05, cutoff=0.6, box=None, pbc=False):
    res_pairs, df_atoms = find_res_pairs_from_atomistic_pdb(atomistic_pdb, frame, radius, bonded_radius, cutoff, box, pbc)
    # pick out CA atoms for each residue
    dict_res_CA = {}
    for i, row in df_atoms.iterrows():
        if row['name'] == 'CA':
            unique_resSeq = df_atoms.loc[i, 'unique_resSeq']
            dict_res_CA[unique_resSeq] = i
    #df_cg_ca_pairs = pd.DataFrame(columns=['a1', 'a2', 'sigma'])
    df_cg_ca_pairs = pd.DataFrame(columns=['aai', 'aaj', 'mu']) # use aai, aaj, mu instead of a1, a2, sigma
    ca_atom_pairs = []
    for each in res_pairs:
        a1, a2 = int(each[0]), int(each[1])
        if (a1 in dict_res_CA) and (a2 in dict_res_CA):
            df_cg_ca_pairs.loc[len(df_cg_ca_pairs.index)] = [a1, a2, None]
            ca_atom_pairs.append([dict_res_CA[a1], dict_res_CA[a2]])
    ca_atom_pairs = np.array(ca_atom_pairs)
    traj = mdtraj.load_pdb(atomistic_pdb)
    df_cg_ca_pairs['mu'] = mdtraj.compute_distances(traj, ca_atom_pairs, pbc)[frame]
    df_cg_ca_pairs.loc[:, 'epsilon'] = 1
    df_cg_ca_pairs.loc[:, 'sigma'] = 0.05
    df_cg_ca_pairs.loc[:, 'alpha'] = 1.6777216e-5
    return df_cg_ca_pairs


def legacy_find_ca_pairs_from_atomistic_pdb(atomistic_pdb, frame=0, radius=0.1, bonded_radius=0.05, cutoff=0.6, box=None, pbc=False):
    # an old version used by smog previously with small bugs
    # we keep this version to compare with smog
    # find all the native pairs following the shadow algorithm
    # the contacts are searched based on all the heavy atoms in the atomistic model
    # the outputs are contacts between residues (equivalent to contacts between CA atoms)
    # the output CA atom index (equivalent to residue index) starts from 0
    # radius is the shadow radius for each atom 
    # when testing whether atom 3 blocks the light between atom 1 and 2, if atom 3 is bonded to atom 1 or 2, then atom 3 uses bonded_radius as its own radius
    # cutoff is the contact cutoff distance
    # radius and cutoff are both in unit nm, consistent with mdtraj length unit
    traj = mdtraj.load_pdb(atomistic_pdb)
    top = traj.topology
    n_atoms = top.n_atoms
    df_atoms, _bonds = top.to_dataframe()
    df_atoms.index = list(range(len(df_atoms.index)))
    # add unique resSeq, which starts from 0
    # also pick out the index of all the CA atoms
    unique_resSeq = 0
    ca_atoms = []
    for i, row in df_atoms.iterrows():
        if i >= 1:
            if row['resSeq'] != df_atoms.loc[i - 1, 'resSeq'] or row['chainID'] != df_atoms.loc[i - 1, 'chainID']:
                unique_resSeq += 1
        df_atoms.loc[i, 'unique_resSeq'] = unique_resSeq
        if row['name'] == 'CA':
            ca_atoms.append(i)
    neighbors_no_hyd_dict = {}
    for i in range(n_atoms):
        if df_atoms.loc[i, 'element'] != 'H':
            neighbors_no_hyd_dict[i] = []
    coord = traj.xyz[frame]
    neighbor_atom_pairs, neighbor_atom_pair_distances = get_neighbor_pairs_and_distances(coord, cutoff, box, pbc) # find spatially close neighbors
    dist_matrix = np.zeros((n_atoms, n_atoms))
    dist_matrix[:, :] = -1 # initialize distance matrix elements as -1
    beyond_1_4_neighbors_no_hyd_atom_pairs = []
    for i in range(len(neighbor_atom_pairs)):
        a1, a2 = int(neighbor_atom_pairs[i, 0]), int(neighbor_atom_pairs[i, 1])
        if a1 > a2:
            a1, a2 = a2, a1
        # save the distances of all the neighboring pairs into dist_matrix
        dist_matrix[a1, a2] = neighbor_atom_pair_distances[i]
        dist_matrix[a2, a1] = neighbor_atom_pair_distances[i]
        if (df_atoms.loc[a1, 'element'] != 'H') and (df_atoms.loc[a2, 'element'] != 'H'):
            neighbors_no_hyd_dict[a1].append(a2)
            neighbors_no_hyd_dict[a2].append(a1)
            chain1, chain2 = df_atoms.loc[a1, 'chainID'], df_atoms.loc[a2, 'chainID']
            resSeq1, resSeq2 = df_atoms.loc[a1, 'resSeq'], df_atoms.loc[a2, 'resSeq']
            if (chain1 != chain2) or abs(resSeq1 - resSeq2) > 3:
                # find atom pairs that are involved in residues that do not have bonded interactions in CA model
                beyond_1_4_neighbors_no_hyd_atom_pairs.append([a1, a2])
    res_pairs = []
    bonded_neighbor_dict = get_bonded_neighbor_dict(atomistic_pdb)
    for each in beyond_1_4_neighbors_no_hyd_atom_pairs:
        a1, a2 = each[0], each[1]
        unique_resSeq1, unique_resSeq2 = df_atoms.loc[a1, 'unique_resSeq'], df_atoms.loc[a2, 'unique_resSeq']
        # since a1 < a2 and they are beyond 1-4 interactions, thus unique_resSeq1 < unique_resSeq2
        if [unique_resSeq1, unique_resSeq2] in res_pairs:
            continue
        flag = True
        # test if atom a3 blocks the contact between atom a1 and a2
        # only test atom a3 if d13 < d12 and d23 < d12
        # because a1 and a2 have contact, d12 < cutoff, thus d13 < cutoff and d23 < cutoff
        # so atom a3 has to be the neighbor of both atom a1 and a2
        block_candidates = [a3 for a3 in neighbors_no_hyd_dict[a1] if a3 in neighbors_no_hyd_dict[a2]]
        for a3 in block_candidates:
            # check if a3 blocks the light from a1 to a2 or the light from a2 to a1
            # if dist_matrix element is less than 0, then we need to compute them
            # because a1 and a2, a2 and a3, a1 and a3 are all neighboring pairs, thus d12, d13, and d23 can be directly read from dist_matrix
            d12, d13, d23 = dist_matrix[a1, a2], dist_matrix[a1, a3], dist_matrix[a2, a3]
            if d12 <= d13 or d12 <= d23:
                continue
            if (a3 in bonded_neighbor_dict[a1]) or (a3 in bonded_neighbor_dict[a2]):
                radius3 = bonded_radius
            else:
                radius3 = radius
            # check if the light from a1 to a2 is blocked by a3
            angle213 = mdtraj.compute_angles(traj, np.array([[a2, a1, a3]]), pbc)[frame, 0]
            # this is where the old version has bug
            # the old version uses atan instead of asin to compute theta12 and theta13
            theta12 = math.atan(radius/d12)
            theta13 = math.atan(radius3/d13)
            if theta12 + theta13 >= angle213:
                flag = False
                break
            # check if the light from a2 to a1 is blocked by a3
            angle123 = mdtraj.compute_angles(traj, np.array([[a1, a2, a3]]), pbc)[frame, 0]
            # this is where the old version has bug
            # the old version uses atan instead of asin to compute theta12 and theta23
            theta12 = math.atan(radius/d12)
            theta23 = math.atan(radius3/d23)
            if theta12 + theta23 >= angle123:
                flag = False
                break
        if flag:
            res_pairs.append([unique_resSeq1, unique_resSeq2])
    res_pairs = np.array(sorted(res_pairs))
    if res_pairs.ndim > 1:
        df_ca_pairs = pd.DataFrame(res_pairs, columns=['a1', 'a2'])
        # residue pairs are equivalent to CA atom pairs in the CA model
        ca_atom_pairs = []
        for each in res_pairs:
            ca1, ca2 = ca_atoms[int(each[0])], ca_atoms[int(each[1])]
            ca_atom_pairs.append([ca1, ca2])
        ca_atom_pairs = np.array(ca_atom_pairs)
        sigma = mdtraj.compute_distances(traj, ca_atom_pairs, pbc)[frame]
        df_ca_pairs['sigma'] = sigma
    else:
        df_ca_pairs = pd.DataFrame(columns=['a1', 'a2', 'sigma'])
    return df_ca_pairs


def load_ca_pairs_from_top(top_file, ca_pdb, frame=0, pbc=False):
    # use this function to load native pairs from gromacs topology file
    # this can be used to check our code
    with open(top_file, 'r') as input_reader:
        top_file_lines = input_reader.readlines()
    flag = False
    for i in range(len(top_file_lines)):
        if '[ pairs ]' in top_file_lines[i]:
            start_index = i + 2
            flag = True
        if flag and len(top_file_lines[i].split()) == 0:
            end_index = i - 1
            break
    ca_pairs = []
    for i in range(start_index, end_index + 1):
        elements = top_file_lines[i].split()
        a1 = int(elements[0]) - 1
        a2 = int(elements[1]) - 1
        if a1 > a2: 
            a1, a2 = a2, a1
        ca_pairs.append([a1, a2])
    ca_pairs = np.array(sorted(ca_pairs))
    df_ca_pairs = pd.DataFrame(ca_pairs, columns=['a1', 'a2'])
    traj = mdtraj.load_pdb(ca_pdb)
    sigma = mdtraj.compute_distances(traj, ca_pairs, pbc)[frame]
    df_ca_pairs['sigma'] = sigma
    return df_ca_pairs



