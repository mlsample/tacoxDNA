from sys import argv
import os
from os import chdir, path
import numpy as np
from oxDNA_analysis_tools import duplex_finder, output_bonds
from oxDNA_analysis_tools.UTILS.RyeReader import strand_describe, describe, get_confs, inbox
from copy import deepcopy
import json as js
import sys
from contextlib import contextmanager
from scipy import stats
from collections import Counter

def oxDNA_to_nNxB(
    particles_per_course_bead: int, path_to_conf: str, path_to_top: str, path_to_input: str,
    path_to_traj: str, material: str, remainder_modifer: float, force_stiff: float, system_name: str, n_cpus=1
    ):
    """Converts oxDNA simulation output to nNxB format for coarse-grained analysis.

    Args:
        particles_per_course_bead (int): Number of particles per coarse-grained bead.
        path_to_conf (str): Path to the configuration file.
        path_to_top (str): Path to the topology file.
        path_to_input (str): Path to the input file for oxDNA.
        path_to_traj (str): Path to the trajectory file from oxDNA simulation.
        n_cpus (int): Number of CPUs to use for parallel processing (default: 1).
        system_name (str): Name of the system for output files (default: None).

    Returns:
        None, but prints the status of nNxB file creation and any issues encountered.
    """
    
    # Extract nucleotide and position information from configuration and topology files
    monomer_id_info, positions, ox_conf = nucleotide_and_position_info(path_to_conf, path_to_top)
    # Run the duplex finder and output bonds analysis using oxDNA Analysis Tools
    path_to_duplex_file, path_to_hb_file = run_oat_duplex_output_bonds(path_to_input, path_to_traj, n_cpus)
    
    print('Course-graining the particles')
    # Use the duplex finder output to create a mapping from duplexes to particles
    duplex_to_particle, particle_to_duplex = associate_particle_idx_to_unique_duiplex(path_to_duplex_file)
    # Run error correction for the duplex finder by handling edge cases and non-bonded fixes
    duplex_to_particle, all_edge_cases = run_duplex_finder_error_correction(duplex_to_particle, particle_to_duplex, monomer_id_info, positions, path_to_hb_file)
    
    # Create coarse-grained particle information formated as unordered dicts
    coarse_particles_positions, coarse_particles_nucleotides, coarse_particle_indexes, course_particle_strands = create_coarse_particle_info(
       duplex_to_particle, positions, particles_per_course_bead, monomer_id_info, remainder_modifer=remainder_modifer 
    )
    # Write the coarse-grained particle information to nNxB files
    coarse_particles_nucleotides_ordered, coarse_particles_positions_ordered, bead_pair_dict, coarse_particle_indexes_ordered, formatted_strand_list = write_course_particle_files_functional(
       coarse_particles_nucleotides, coarse_particles_positions, coarse_particle_indexes, course_particle_strands, system_name, particles_per_course_bead, material, ox_conf, force_stiff
    )
    
    print_results(all_edge_cases, ox_conf, coarse_particle_indexes_ordered)
    
    return None
    

def print_results(all_edge_cases, ox_conf, coarse_particle_indexes_ordered):
    if all_edge_cases:
        print('Able to create nNxB files, but unable to assign all nucleotides to duplexes.')
    else:
        print('nNxB files created successfully.')
    flat_edge_cases = [item for sublist in all_edge_cases for item in sublist]

    n_starting_particles = ox_conf.positions.shape[0]
    flat_indexes = [item for sublist in coarse_particle_indexes_ordered.values() for item in sublist]
    n_course_particles = len(flat_indexes)
    n_particles_for_each_bead = [len(value) for value in flat_indexes]
    unique_particles_per_bead = set(n_particles_for_each_bead)
    particle_per_bead_statistic = {value: n_particles_for_each_bead.count(value) for value in unique_particles_per_bead}
    
    print(f'\nConversion statistics:\n')
    print(f'Percentage of particles included in nNxB file: {(1 - (len(flat_edge_cases)/n_starting_particles))*100:.2f}%')
    print(f'Original particles: {n_starting_particles}, Course particles: {n_course_particles}, Compression factor: {n_starting_particles/n_course_particles:.2f} particles per bead')
    print('Percent of course beads with N particles:')
    for key, value in particle_per_bead_statistic.items():
        print(f' {key} = {value / n_course_particles * 100:.2f}%')

    
def nucleotide_and_position_info(
    path_to_conf: str, path_to_top: str
    ):
    """Extracts nucleotide and position information from configuration and topology files.

    Args:
        path_to_conf (str): Path to the configuration file.
        path_to_top (str): Path to the topology file.

    Returns:
        Tuple containing monomer ID information and positions.
    """
    top_info, traj_info = describe(None, path_to_conf)
    system, monomer_id_info = strand_describe(path_to_top)
    
    ox_conf = get_confs(top_info, traj_info, 0, 1)[0]
    ox_conf = inbox(ox_conf, center=True)
    positions = ox_conf.positions
    
    return monomer_id_info, positions, ox_conf


def run_oat_duplex_output_bonds(
    path_to_input: str, path_to_traj: str, n_cpus: int
    ):
    """Runs the duplex finder and output bonds analysis using oxDNA Analysis Tools.

    Parameters:
        path_to_input (str): Path to the input file for oxDNA.
        path_to_traj (str): Path to the trajectory file from oxDNA simulation.
        n_cpus (str): Number of CPUs to use for parallel processing.

    Returns:
    - Paths to the duplex info file and hydrogen bonds info file.
    """
    path_to_files = '/'.join(path_to_traj.split('/')[:-1])
    path_to_duplex_file = create_duplex_info_file(path_to_files, path_to_input, path_to_traj, n_cpus=n_cpus)
    path_to_hb_file = create_output_bonds_info_file(path_to_files, path_to_input, path_to_traj, n_cpus=n_cpus)
    
    return path_to_duplex_file, path_to_hb_file 


def create_duplex_info_file(
    abs_path_to_files: str, input_file_name:str, conf_file_name:str, n_cpus=1
    ):
    """Creates a duplex info file using the duplex finder from oxDNA Analysis Tools.

    Args:
        abs_path_to_files (str): Absolute path to the directory containing the input and trajectory files.
        input_file_name (str): Name of the input file.
        conf_file_name (str): Name of the conformation file.
        n_cpus (int, optional): Numeber of CPUs to use. Defaults to 1.

    Returns:
        path_to_duplex_info_file (str): Path to duplex info file
    """
    print('Creating duplex info file')
    chdir(abs_path_to_files)
    argv.clear()
    argv.extend(['duplex_finder.py', input_file_name, conf_file_name, '-p', str(n_cpus)])
    duplex_finder.main()
    path_to_duplex_info_file = path.join(abs_path_to_files, 'angles.txt')
    return path_to_duplex_info_file


def create_output_bonds_info_file(
    abs_path_to_files: str, input_file_name: str, conf_file_name: str, n_cpus=1
    ):
    """Create a hydrogen bonds info file using the output bonds analysis from oxDNA Analysis Tools.

    Args:
        abs_path_to_files (str): Absolute path to the directory containing the input and trajectory files.
        input_file_name (str): Input file name.
        conf_file_name (str): Conformation file name.
        n_cpus (int, optional): Number of CPUs to use. Defaults to 1.

    Returns:
        path_to_hb_info_file (str): Path to hydrogen bonds info file.
    """
    print('Creating hydrogen bonds info file')
    chdir(abs_path_to_files)
    argv.clear()
    argv.extend(['output_bonds.py', '-v', 'bonds.json',input_file_name, conf_file_name, '-p', str(n_cpus)])
    output_bonds.main()
    path_to_hb_info_file = path.join(abs_path_to_files, 'bonds_HB.json')
    return path_to_hb_info_file

def read_hb_energy_file(
    path_to_hb_energy_file: str
    ):
    """Read the hb energy file and return the data as a dictionary.

    Args:
        path_to_hb_energy_file (str): Path to the hydrogen bond energy file.

    Returns:
        hb_energy (dict): Dict containing the hydrogen bond energy data for each nucleotide.
    """
    with open(path_to_hb_energy_file, 'r') as f:
        hb_energy = js.load(f)
    return hb_energy

def associate_particle_idx_to_unique_duiplex(
    path_to_duplex_info: str
    ):
    """Map the particle indices to unique duplexes using the duplex info file.

    Args:
        path_to_duplex_info (str): Absolute path to the duplex info file.

    Returns:
        d_to_p (dict): Dictonary with duplex id as key, and as values we have a list of 2 lists where the first
                list is the particle idxes of the duplex in 3` -> 5` and 2nd list is particle idex in 5` -> 3`
        p_to_d (dict): Dictionary with particle idx as key and duplex id as value.
    """
    with open(path_to_duplex_info, 'r') as f:
        duplex_ends = f.readlines()

    n_conf = int(duplex_ends[-1].split('\t')[0])
    d1_ends = {}
    d2_ends = {}
    
    split_lines = [line.split('\t') for line in duplex_ends[1:]]
    conf_split = np.array([ends[0] for ends in split_lines], dtype=int)
    d1_split = np.array([ends[2:4] for ends in split_lines], dtype=int)
    d2_split = np.array([ends[4:6] for ends in split_lines], dtype=int)

    # make a dict called d1_ends with the conf as the key de
    d1_ends = split_by_conf(conf_split, d1_split)
    d2_ends = split_by_conf(conf_split, d2_split)
    
    d1_most_nucs = [np.sum(d[:, 1] - d[:, 0] + 1) for d in d1_ends]
    d2_most_nucs = [np.sum(d[:, 1] - d[:, 0] + 1) for d in d2_ends]
    
    d1_argmax = np.argmax(d1_most_nucs)
    d2_argmax = np.argmax(d2_most_nucs)
    
    if d1_argmax == d2_argmax:
        d1_argmax_ends = d1_ends[d1_argmax]
        d2_argmax_ends = d2_ends[d2_argmax]
    else:
        d1_argmax_ends = d1_ends[d1_argmax]
        d2_argmax_ends = d2_ends[d1_argmax]
    
    d1s = [np.arange(d1[0], d1[1]+1) for d1 in d1_argmax_ends]
    d2s = [np.arange(d2[0], d2[1]+1) for d2 in d2_argmax_ends]
    


    p_to_d = {}
    for i, (d1, d2) in enumerate(zip(d1s, d2s)):
        for p in d1:
            p_to_d[p] = i+1
        for p in d2:
            p_to_d[p] = -(i+1)

    # d_to_p = {duplex: [[],[]] for duplex in range(1, len(d1s)+1,)}
    d_to_p = {}
    for duplex, (d1, d2) in enumerate(zip(d1s, d2s), start=1):
        d_to_p[duplex] = []
        d_to_p[duplex].append(d1.tolist())
        d_to_p[duplex].append(d2.tolist()) 


    # Now d_to_p will contain lists of particles for each duplex
    for duplex, strands in d_to_p.items():
        strands[1] = strands[1][::-1]
        
    return d_to_p, p_to_d


def split_by_conf(conf_split, d1_splits):
    # Find the indices where the value changes in conf_split
    change_indices = np.where(np.diff(conf_split) != 0)[0] + 1

    # Compute the counts of each consecutive integer
    counts = np.diff(np.append(0, change_indices))

    # Split d1_splits based on these counts
    d1_ends = np.split(d1_splits, np.cumsum(counts))

    return d1_ends


def run_duplex_finder_error_correction(
    duplex_to_particle: dict, particle_to_duplex: dict, nucleotides_in_duplex, positions, path_to_hb_file
    ):
    """Runs error correction for the duplex finder by handling edge cases and non-bonded fixes.

    Args:
        duplex_to_particle (dict): Mapping from duplexes to particles.
        particle_to_duplex (dict): Mapping from particles to duplexes.
        nucleotides_in_duplex: Information about nucleotides in each duplex.
        positions (np.array): Positions of particles.
        path_to_hb_file (str): Path to the hydrogen bond information file.

    Returns:
        Updated duplex_to_particle mapping and a list of all edge cases.
    """
    all_edge_cases, len_one_parts = get_nuc_not_included_in_d_to_p(particle_to_duplex, nucleotides_in_duplex)

    if len_one_parts.size > 0:
        duplex_to_particle, single_nucs_dealt_with = deal_with_single_nuc_edge_cases(positions, len_one_parts, duplex_to_particle)

        if type(single_nucs_dealt_with) != bool:
            for idx, nucs in enumerate(all_edge_cases):
                for i, val in enumerate(nucs):
                    if val in single_nucs_dealt_with:
                        all_edge_cases[idx].pop(i)

            all_edge_cases = [sublist for sublist in all_edge_cases if sublist]
    
    if all_edge_cases:
        duplex_to_particle, fixed = fully_complementary_sequential_fix(nucleotides_in_duplex, positions, all_edge_cases, duplex_to_particle)
        if fixed:
            values_to_remove = [ele for sublist in fixed for item in sublist for ele in item]
            all_edge_cases = [[val for val in sublist if val not in values_to_remove] for sublist in all_edge_cases]
            all_edge_cases = [sublist for sublist in all_edge_cases if sublist]   
        
    # duplex_to_particle, fixed = fully_complementary_sequential_fixs(nucleotides_in_duplex, positions, all_edge_cases, duplex_to_particle, path_to_hb_info_file)
    
    if all_edge_cases:
        duplex_to_particle, fixed = non_bonded_fixes(path_to_hb_file, all_edge_cases, duplex_to_particle)
        if fixed:
            values_to_remove = np.concatenate(fixed)
            all_edge_cases = [[val for val in sublist if val not in values_to_remove] for sublist in all_edge_cases]
            all_edge_cases = [sublist for sublist in all_edge_cases if sublist]

    if all_edge_cases:
        print('\nUnable to assign all nucleotides to duplexes, continuing with the ones that were assigned. Unassigned nucleotides:\n', all_edge_cases,'\n')
    
    return duplex_to_particle, all_edge_cases


def get_nuc_not_included_in_d_to_p(p_to_d, nucleotides_in_duplex):
    
    nucs_in_duplex = list(p_to_d.keys())
    nucs_in_duplex.sort()
    ids = [nuc.id for nuc in nucleotides_in_duplex]
    ids.sort()
    ids = set(ids)
    nucs_in_duplex = set(nucs_in_duplex)
    difference = ids.difference(nucs_in_duplex)
    difference_list = sorted(list(difference))

    sequential_parts = []
    current_sequence = [difference_list[0]]

    for i in range(1, len(difference_list)):
        if difference_list[i] == difference_list[i - 1] + 1:
            # The current element is sequential to the previous one
            current_sequence.append(difference_list[i])
        else:
            # The current element is not sequential to the previous one,
            # store the current sequence and start a new one
            sequential_parts.append(current_sequence)
            current_sequence = [difference_list[i]]

    # Append the last sequence if it exists
    if current_sequence:
        sequential_parts.append(current_sequence)

    len_parts = [len(part) for part in sequential_parts]

    len_one_parts = np.array([part for part in sequential_parts if len(part) == 1]).flatten()
    
    return sequential_parts, len_one_parts


def calculate_distance_matrix(points):
    # Ensure the input is a NumPy array
    points = np.asarray(points)

    # Calculate differences in each dimension (x, y, z) between points
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]

    # Calculate the squared distances
    squared_distances = np.sum(diff**2, axis=-1)

    # Take the square root to get the Euclidean distance
    distance_matrix = np.sqrt(squared_distances)

    return distance_matrix


def deal_with_single_nuc_edge_cases(positions, len_one_parts, d_to_p):
    
    len_one_parts_pos = positions[len_one_parts]

    len_one_parts_pos_ma = calculate_distance_matrix(len_one_parts_pos)
    len_one_parts_pos_ma[len_one_parts_pos_ma == 0] = 'inf'
    idx = np.argmin(len_one_parts_pos_ma, axis=1)
    mins = np.min(len_one_parts_pos_ma, axis=1)

    idx_pairs = [[i,val] for i,val in enumerate(idx)]

    idx_pair_set = []
    for pair in idx_pairs:
        pair_inv = [pair[1], pair[0]]
        if pair_inv not in idx_pair_set:
            idx_pair_set.append(pair)

    len_one_parts_pairs = [[len_one_parts[pair[0]], len_one_parts[pair[1]]] for pair in idx_pair_set]
    removed = []

    for key, value in d_to_p.items():
        ends = [value[0][-1], value[1][-1]]
        starts = [value[0][0], value[1][0]]
        for pairs in  len_one_parts_pairs:
            look_0 = [pairs[0] - 1, pairs[1] + 1]
            look_1 = [pairs[0] + 1, pairs[1] - 1]

            if ends == look_0:
                d_to_p[key][0].append(pairs[0])
                d_to_p[key][1].append(pairs[1])
                removed.append(pairs)

            elif starts == look_0:
                d_to_p[key][0].insert(0, pairs[0])
                d_to_p[key][1].insert(0, pairs[1])
                removed.append(pairs)

            elif ends == look_1:
                d_to_p[key][0].append(pairs[0])
                d_to_p[key][1].append(pairs[1])
                removed.append(pairs)

            elif starts == look_1:
                d_to_p[key][0].insert(0, pairs[0])
                d_to_p[key][1].insert(0, pairs[1])
                removed.append(pairs)

    if removed:
        single_nucs_dealt_with = np.concatenate(removed)
    else:
        single_nucs_dealt_with = False
    
    return d_to_p, single_nucs_dealt_with


def full_sequential_group_distance_check(positions, sequential_parts):
    pos_edge_cases = [positions[part] for part in sequential_parts]
    cms_edge_cases = np.array([np.mean(part, axis=0) for part in pos_edge_cases])

    distance_matrix = calculate_distance_matrix(cms_edge_cases)
    distance_matrix[distance_matrix == 0] = 'inf'

    nearest_edge_case = np.argmin(distance_matrix, axis=1)
    nearest_edge_case_min = np.min(distance_matrix, axis=1)

    nearest_edge_case_pairs = [[idx,pair] for idx,pair in enumerate(nearest_edge_case)]

    pair_set = []
    for pair in nearest_edge_case_pairs:
        pair_inv = [pair[1], pair[0]]
        if pair_inv not in pair_set:
            pair_set.append(pair)

    indexes = [[sequential_parts[pair[0]], sequential_parts[pair[1]]] for pair in pair_set]

    return pair_set, indexes, nearest_edge_case_min


def fully_complementary_sequential_fix(nucleotides_in_duplex, positions, all_edge_cases, duplex_to_particle):
    all_edge_cases = [sublist for sublist in all_edge_cases if len(sublist) > 1]
    pair_set, indexes, nearest_edge_case_min = full_sequential_group_distance_check(positions, all_edge_cases)

    easy_fix = []
    monomer_info = np.array(nucleotides_in_duplex)
    for idxes in indexes:
        nuc_dict = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
        idx_1 = idxes[0]
        idx_2 = idxes[1]

        monomers_1 = monomer_info[idx_1]
        monomers_1 = [mono.btype for mono in monomers_1]

        monomers_2 = monomer_info[idx_2]
        monomers_2 = [nuc_dict[mono.btype] for mono in monomers_2][::-1]

        if monomers_1 == monomers_2:
            new_key = len(duplex_to_particle) + 1
            duplex_to_particle[int(new_key)] = [idxes[0], idxes[1][::-1]]
            easy_fix.append([idxes[0], idxes[1][::-1]])
    
    return duplex_to_particle, easy_fix


def non_bonded_fixes(path_to_hb_info_file, all_edge_cases, duplex_to_particle):
    fixed = []
    hb_energy = read_hb_energy_file(path_to_hb_info_file)
    hb_energy = np.array(hb_energy['HB (oxDNA su)'])
    hb_boolean = np.array(hb_energy < -0.2)

    hb_boolean_edge_cases = [hb_boolean[case] for case in all_edge_cases]
    new_boolean_criteria = [np.mean(~boolean) for boolean in hb_boolean_edge_cases]
    hb_edge_cases_idx = [edge_case for edge_case, boolean in zip(all_edge_cases, new_boolean_criteria) if boolean >= 0.3]
    
    for case in hb_edge_cases_idx:
        if len(case) > 1:
            new_key = len(duplex_to_particle) + 1
            duplex_to_particle[int(new_key)] = [case]

            fixed.append(case)
    
        
    return duplex_to_particle, fixed


def create_coarse_particle_info(d_to_p, positions, particles_per_course_bead, nucleotides_in_duplex, remainder_modifer=0.25):
    remainder_modifer = np.floor(particles_per_course_bead * remainder_modifer)
    coarse_particles_positions = {}
    coarse_particles_nucleotides = {}
    coarse_particle_indexes = {}
    course_particle_strands = {}
    
    # Iterate through each duplex
    # duplex is the int starting at 1, strand is each strand in the duplex
    for duplex, strands in d_to_p.items():
        # strands[0] contains the particle indices of the first strand
        # strands[1] contains the particle indices of the second strand
        
        for strand_idx, strand in enumerate(strands):
            #start by pulling the first strand from strands
            coarse_positions = []
            coarse_nucleotides = []
            course_indexes = []
            course_strands = []
            
            # Iterate through the particles in the strand, taking N at a time
            for i in range(0, len(strand), particles_per_course_bead):
                # Get the particle indices of the current group
                particle_indices = strand[i:i+particles_per_course_bead]

                if (len(particle_indices) > remainder_modifer) or (not course_indexes):
                    course_indexes.append(particle_indices)
                    # Compute the center of mass for the current group
                    center_of_mass = np.mean(positions[particle_indices], axis=0)
                    coarse_positions.append(center_of_mass)

                    # Accumulate the nucleotide types for the current group
                    nucleotide_types = ''.join([nucleotides_in_duplex[idx].btype for idx in particle_indices])
                    coarse_nucleotides.append(nucleotide_types)
                    
                    strand_ids = Counter([nucleotides_in_duplex[idx].strand.id for idx in particle_indices]).most_common(1)[0][0]
                    course_strands.append(strand_ids)
                    
                    
                elif len(particle_indices) <= remainder_modifer:
                    course_indexes[-1].extend(particle_indices)

                    # Accumulate the nucleotide types for the current group
                    nucleotide_types = ''.join([nucleotides_in_duplex[idx].btype for idx in particle_indices])
                    coarse_nucleotides[-1] += nucleotide_types
    
                    # Compute the center of mass for the current group
                    particle_indices = strand[i-particles_per_course_bead:]
                    center_of_mass = np.mean(positions[particle_indices], axis=0)
                    coarse_positions[-1] = center_of_mass
                    
                    strand_ids = Counter([nucleotides_in_duplex[idx].strand.id for idx in particle_indices]).most_common(1)[0][0]
                    course_strands[-1] = (strand_ids)
                else:
                    print(f'Error: Something went wrong with the coarse-graining process. Duplex idx: {duplex}')
                    
                    
            # Store the coarse-grained positions for the current strand in the duplex
            key = (duplex, strand_idx)
            coarse_particles_positions[key] = coarse_positions
            coarse_particles_nucleotides[key] = coarse_nucleotides
            coarse_particle_indexes[key] = course_indexes
            course_particle_strands[key] = course_strands
    return coarse_particles_positions, coarse_particles_nucleotides, coarse_particle_indexes, course_particle_strands


def write_course_particle_files_functional(
    coarse_particles_nucleotides, coarse_particles_positions,
    coarse_particle_indexes, course_particle_strands,
    system_name, particles_per_course_bead,
    material, ox_conf, force_stiff):
    
    # Write the nucleotide sequence to a file with one strand being in 3` to 5` order and the other being in 5` to 3` order for visual inspection
    write_sanity_check(coarse_particles_nucleotides, system_name)
    
    # Find the order to call the keys in the course_particle dictionaries to place beads in sequential order
    keys_ordered_acending, indexes_ordered_acending = order_indexes_and_keys_acending(coarse_particle_indexes)
    
    # Write the nucleotide sequence of each course bead with the cooresponding course particle index in 5` to 3` order
    coarse_particles_nucleotides_ordered = write_course_particle_nucleotides(
        coarse_particles_nucleotides, coarse_particle_indexes, keys_ordered_acending, system_name, particles_per_course_bead)
    
    # Write the list of bonded course particles
    bead_pair_dict, coarse_particle_indexes_ordered = write_course_particle_bonded_pairs(
        coarse_particle_indexes, coarse_particles_nucleotides_ordered, indexes_ordered_acending, system_name, particles_per_course_bead, course_particle_strands)
    
    # Write the course-grained conformation position and direction info to .dat file
    coarse_particles_positions_ordered = write_course_particles_positions(
        coarse_particles_positions, keys_ordered_acending, system_name, particles_per_course_bead, indexes_ordered_acending, ox_conf)
    
    # Write the annamo topology and interaction matrix files
    strand_list = write_strand_info(
        course_particle_strands, coarse_particles_nucleotides, keys_ordered_acending, system_name, particles_per_course_bead, material)
    
    write_course_top_for_visualization(
        coarse_particles_nucleotides_ordered, bead_pair_dict, course_particle_strands, particles_per_course_bead, system_name, material, keys_ordered_acending)
    
    write_mutual_trap_file(particles_per_course_bead, system_name, bead_pair_dict, force_stiff)
        
    return coarse_particles_nucleotides_ordered, coarse_particles_positions_ordered, bead_pair_dict, coarse_particle_indexes_ordered, strand_list


def order_indexes_and_keys_acending(coarse_particle_indexes):
    sorted_values_keys = sorted(
        [(sorted(values), key) for key, values in coarse_particle_indexes.items()]
    )
    indexes_ordered_acending = [item[0] for item in sorted_values_keys]
    keys_ordered_acending = [item[1] for item in sorted_values_keys]
                
    return keys_ordered_acending, indexes_ordered_acending


def write_strand_info(course_particle_strands, coarse_particles_nucleotides, keys_ordered_acending, system_name, particles_per_course_bead, material):  
    ordered_nucs = deepcopy(coarse_particles_nucleotides)
    
    # Reverse the nucleotides in each group to put the nucleotides back
    # in 3` to 5` order after I reversed them in d_to_p
    course_keys = list(coarse_particles_nucleotides.keys())
    num_course_duplex = len([key for key in course_keys if key[1] == 1])
    
    for idx in range(num_course_duplex):
        for lists in range(len(coarse_particles_nucleotides[(idx+1,1)])):
            ordered_nucs[(idx+1,1)][lists] = coarse_particles_nucleotides[(idx+1,1)][lists][::-1]

    # Reverse the order of the groups to put them back in 3` to 5` order
    for idx in range(num_course_duplex):
        ordered_nucs[(idx+1,1)] = ordered_nucs[(idx+1,1)][::-1]
    
    sort_particles_based_on_strand = {}
    
    for key in keys_ordered_acending:
        strands = course_particle_strands[key]
        nucs = ordered_nucs[key]
        
        for strand, nuc in zip(strands, nucs):
            if strand not in sort_particles_based_on_strand:
                sort_particles_based_on_strand[strand] = []
            sort_particles_based_on_strand[strand].append(nuc)
    
    sorted_keys = sorted(sort_particles_based_on_strand.keys())
    formatted_list = []
    for key in sorted_keys:
        list_of_nucs = sort_particles_based_on_strand[key]
        formatted_list.extend(['x',])
        formatted_list.extend(list_of_nucs)
        formatted_list.extend(['x',])
    
    five_to_three_formatted_list = []
    for nucs in formatted_list:
        five_to_three_formatted_list.append(nucs[::-1])
    five_to_three_formatted_list = five_to_three_formatted_list[::-1]

    print(material)
    write_annamo_topology(five_to_three_formatted_list, system_name, particles_per_course_bead)
    write_annamo_interaction_matrix(five_to_three_formatted_list, material)

    
    with open(f'{system_name}_{particles_per_course_bead}_nuc_beads_strands_list.txt', 'w') as f:
        f.write(f'{five_to_three_formatted_list}')
    
    return five_to_three_formatted_list

def write_course_particle_nucleotides(coarse_particles_nucleotides, coarse_particle_indexes, keys_ordered_acending, system_name, particles_per_course_bead):
    ordered_nucs = deepcopy(coarse_particles_nucleotides)
    
    # Reverse the nucleotides in each group to put the nucleotides back
    # in 5` to 3` order after I reversed them in d_to_p
    course_keys = list(coarse_particles_nucleotides.keys())
    num_course_duplex = len([key for key in course_keys if key[1] == 1])
    
    for idx in range(num_course_duplex):
        for lists in range(len(coarse_particles_nucleotides[(idx+1,0)])):
            ordered_nucs[(idx+1,0)][lists] = coarse_particles_nucleotides[(idx+1,0)][lists][::-1]

    # Reverse the order of the groups to put them back in 5` to 3` order
    for idx in range(num_course_duplex):
        ordered_nucs[(idx+1,0)] = ordered_nucs[(idx+1,0)][::-1]

    # Order the groups by acending particle index
    coarse_particles_nucleotides_ordered = [ordered_nucs[order] for order in keys_ordered_acending]   
    coarse_particles_nucleotides_ordered = coarse_particles_nucleotides_ordered[::-1] # Reverse the order of the groups to put them back in 5` to 3` order
    nuc_beads_dic = []
    
    for lists in coarse_particles_nucleotides_ordered:
        for string in lists:
            nuc_beads_dic.append(string) # Reverse the order of the groups to put them back in 5` to 3` order
    # nuc_beads_dic = nuc_beads_dic[::-1]
    
    nuc_beads_dic = {f'{key}':string for key, string in enumerate(nuc_beads_dic)}

    with open(f'{system_name}_{particles_per_course_bead}_nuc_beads_sequence.txt', 'w') as f:
        for key, value in nuc_beads_dic.items():
            f.write(f'{key} {value}\n')

    return coarse_particles_nucleotides_ordered


def compute_course_a1_a3_vectors(ox_conf, indexes_ordered_acending):
    flat_indexes = [item for sublist in indexes_ordered_acending for item in sublist][::-1]
    
    a1s_in_beads = [ox_conf.a1s[idx] for idx in flat_indexes]
    a3s_in_beads = [ox_conf.a3s[idx] for idx in flat_indexes]
    
    course_a1s = [np.mean(a1, axis=0) for a1 in a1s_in_beads]
    course_a3s = [np.mean(a3, axis=0) for a3 in a3s_in_beads]
    
    return course_a1s, course_a3s
    

def write_course_particles_positions(coarse_particles_positions, keys_ordered_acending, system_name, particles_per_course_bead, indexes_ordered_acending, ox_conf):
    ordered_positions = deepcopy(coarse_particles_positions)    
    
    course_keys = list(coarse_particles_positions.keys())
    num_course_duplex = len([key for key in course_keys if key[1] == 1])
    # Reverse the order of the groups to put them back in 3` to 5` order
    for idx in range(num_course_duplex):
        ordered_positions[(idx+1,1)] = coarse_particles_positions[(idx+1,1)][::-1]

    # I think this will correctly order the postions in 5` to 3` order, however I need to test it but am yet to create a means to test it.
    coarse_particles_positions_ordered = [ordered_positions[order][::-1] for order in keys_ordered_acending]    
    coarse_particles_positions_ordered = coarse_particles_positions_ordered[::-1] # Reverse the order of the groups to put them back in 5` to 3` order
    
    course_a1s, course_a3s = compute_course_a1_a3_vectors(ox_conf, indexes_ordered_acending)
    flat_course_particles_positions_ordered = [item for sublist in coarse_particles_positions_ordered for item in sublist]
    
    with open(f'{system_name}_{particles_per_course_bead}_nuc_beads_positions.dat', 'w') as file:
        file.write("t = 0\n")
        file.write(f"b = {ox_conf.box[0]} {ox_conf.box[1]} {ox_conf.box[2]}\n")
        file.write(f"E = 0 0 0\n")
        
        for position, a1, a3 in zip(flat_course_particles_positions_ordered, course_a1s, course_a3s):
            # Write x, y, z, a1_x, ..., a3_x, ..., vel_x, ..., angular_vel_x, ..., angular_vel_z separated by spaces
            file.write(f"{position[0]} {position[1]} {position[2]} {a1[0]} {a1[1]} {a1[2]} {a3[0]} {a3[1]} {a3[2]} 0 0 0 0 0 0\n")
            
    with open(f'{system_name}_{particles_per_course_bead}_nuc_beads_positions.xyz', 'w') as file:
        for position in flat_course_particles_positions_ordered:
            # Write x, y, z positions separated by spaces
            file.write(f"{position[0]} {position[1]} {position[2]}\n")
    
    return coarse_particles_positions_ordered


def write_course_particle_bonded_pairs(
    coarse_particle_indexes, coarse_particles_nucleotides_ordered, indexes_ordered_acending, system_name, particles_per_course_bead, course_particle_strands
    ):
    ordered_indexes = deepcopy(coarse_particle_indexes)
    
    course_keys = list(coarse_particle_indexes.keys())
    num_course_duplex = len([key for key in course_keys if key[1] == 1])
    
    # paired_indexes = {k:v for idx,(k,v) in enumerate(ordered_indexes.items()) if idx < num_course_duplex*2}
    # particles_idx_pairs = []
    # for (n, _), paired_lists in paired_indexes.items():
    #     for k in range(len(paired_lists)):  # Assuming both lists in each pair are of the same length

    #         pair = (paired_indexes[(n, 0)][k], paired_indexes[(n, 1)][k])
    #         particles_idx_pairs.append(pair)

    # print('here')
    ordered_indexes_values = list(ordered_indexes.values())
    particles_idx_pair_dict = {}
    i = 0
    for values in ordered_indexes_values[:num_course_duplex*2:2]:
        for bead in values:
            particles_idx_pair_dict[i] = [bead]
            i += 1
            
    i = 0
    for values in ordered_indexes_values[1:num_course_duplex*2:2]:
        for bead in values:
            particles_idx_pair_dict[i].append(bead)
            i += 1
    
    start = len(particles_idx_pair_dict)
    for values in ordered_indexes_values[num_course_duplex*2:]:
        for bead in values:
            particles_idx_pair_dict[start] = [bead]
            start += 1
    
    # print(particles_idx_pair_dict)
    
    indexes_acending = []
    for lists in indexes_ordered_acending:
        for embedded_list in lists:
            indexes_acending.append(embedded_list)
    
    
    ordered_indexes_acending = {f'{key}':ind for key,ind in enumerate(indexes_acending)}
    
    dict_values = list(ordered_indexes_acending.values())
    dict_values = dict_values[::-1] # Reverse the order of the groups to put them back in 5` to 3` order
    
    course_bead_idx_pair_dict = {}
    i = 0
    for beads in particles_idx_pair_dict.values():
        try:
            course_bead_idx_pair_dict[i] = (dict_values.index(beads[1]), dict_values.index(beads[0]))
        except:
            course_bead_idx_pair_dict[i] = (dict_values.index(beads[0]),)
        i +=1
    
    course_bead_idx_pair_dict = dict(sorted(course_bead_idx_pair_dict.items(), key=lambda item: item[1][0]))
    
    nucs = [item for sublist in coarse_particles_nucleotides_ordered for item in sublist]
    
    with open(f'{system_name}_{particles_per_course_bead}_nuc_beads_bonds.txt', 'w') as f:
        for key, value in course_bead_idx_pair_dict.items():
            try:
                f.write(f'({value[0]}, {value[1]}): {nucs[value[0]]} {nucs[value[1]]}\n')
            except:
                f.write(f'({value[0]}): {nucs[value[0]]}\n')
    
    return course_bead_idx_pair_dict, ordered_indexes


def write_course_top_for_visualization(
    coarse_particles_nucleotides_ordered, course_bead_idx_pair_dict, course_particle_strands, particles_per_course_bead, system_name, material, keys_ordered_acending
    ):
    ordered_course_particle_strands = []
    for key in keys_ordered_acending:
        ordered_course_particle_strands.append(course_particle_strands[key])
        
    bead_strand = [item for sublist in ordered_course_particle_strands for item in sublist]
    new_bead_strand = swap_counts(bead_strand)
    unqiue_strands = set(new_bead_strand)
    n_unqiue_strands = len(unqiue_strands)
    
    pair_dict_for_top = {}
    for key, value in course_bead_idx_pair_dict.items():
        if len(value) > 1:
            pair_dict_for_top[value[0]] = True
            pair_dict_for_top[value[1]] = False
        else:
            pair_dict_for_top[value[0]] = True

    nucs = [item for sublist in coarse_particles_nucleotides_ordered for item in sublist]
    pair_dict_for_top = dict(sorted(pair_dict_for_top.items()))
    nucs_for_top = [nucs[idx][0] if pair_bool else nucs[idx][-1] for idx, pair_bool in pair_dict_for_top.items()] 
    
    nuc_to_strand = {strand_idx:[] for strand_idx in unqiue_strands}
    for strand_idx, nuc in zip(new_bead_strand, nucs_for_top):
        nuc_to_strand[strand_idx].append(nuc)
    
    with open(f'{system_name}_{particles_per_course_bead}_visualization.top', 'w') as f:
        f.write(f'{len(nucs_for_top)} {n_unqiue_strands} 5->3\n')
        for nuc_strand, nucleotides in nuc_to_strand.items():
            f.write(f'{"".join(nucleotides)} id={nuc_strand} type={material} circular=false\n')
        
    return None

def swap_counts(nums):
    # Get the unique integers and their counts
    unique_ints = sorted(set(nums))
    counts = [nums.count(num) for num in unique_ints]

    # Build the new list with swapped counts
    new_nums = []
    for i in range(len(unique_ints)):
        new_nums.extend([unique_ints[i]] * counts[-(i + 1)])

    return new_nums


def write_mutual_trap_file(particles_per_course_bead, system_name, course_bead_idx_pair_dict, force_stiff):
    
    with open(f'{system_name}_{particles_per_course_bead}_force.txt', 'w') as f:
        for value in course_bead_idx_pair_dict.values():
            if len(value) > 1:
                f.write('{\n')
                f.write('    type = mutual_trap\n')
                f.write(f'    particle = {value[0]}\n')
                f.write(f'    ref_particle = {value[1]}\n')
                f.write(f'    stiff = {force_stiff}\n')
                f.write('    r0 = 1.2\n')
                f.write('    PBC = 1\n')
                f.write('}\n')
                f.write('\n')
                
                f.write('{\n')
                f.write('    type = mutual_trap\n')
                f.write(f'    particle = {value[1]}\n')
                f.write(f'    ref_particle = {value[0]}\n')
                f.write(f'    stiff = {force_stiff}\n')
                f.write('    r0 = 1.2\n')
                f.write('    PBC = 1\n')
                f.write('}\n')
                f.write('\n')
                
                
            
    return None
    


def write_sanity_check(coarse_particles_nucleotides, system_name):
    sanity_check = deepcopy(coarse_particles_nucleotides)
    sanity_check = {f'{key}':value for key, value in sanity_check.items()}
    stringify = js.dumps(sanity_check)
    stringify = stringify.split(', "(')
    stringify = ',\n "('.join(stringify)
    with open(f'{system_name}_sanity_check.json', 'w') as f:
        f.write(stringify)


def write_annamo_topology(five_to_three_formatted_list, system_name, particles_per_course_bead):
    x_idx = [i for i,e in enumerate(five_to_three_formatted_list) if e=='x']
    n_beads_in_each_strand = [x_idx[i+1]-x_idx[i]-1 for i in range(len(x_idx)-1)]
    number_of_strands = len(n_beads_in_each_strand)

    with open(f'{system_name}_{particles_per_course_bead}_ANNaMo.top', 'w') as topo_file:
        topo_file.write("{} {}\n".format(sum(n_beads_in_each_strand), number_of_strands))
        idx = 0
        for ctrl,n in enumerate(n_beads_in_each_strand):
            if ctrl == 0:
                typ = 0
            else: typ = 1
            for i in range(n):
                if i == 0 or i == n-1:
                    topo_file.write("{} {} 1\n".format(idx, idx+1+ctrl))
                    if i == 0:
                        topo_file.write("{}\n".format(idx+1))
                    elif i == n-1:
                        topo_file.write("{}\n".format(idx-1))
                else:
                    topo_file.write("{} {} 2\n".format(idx, idx+1+ctrl))
                    topo_file.write("{} {}\n".format(idx-1, idx+1))

                idx += 1

            topo_file.write("\n")


def write_annamo_interaction_matrix(beads_sequences, material):
    beads_sequences = check_material(beads_sequences, material)
    H,S = interaction_matrix(beads_sequences, material, 1)
    x_idx = [i for i,e in enumerate(beads_sequences) if e=='x']

    with open("dHdS_annamo.dat","w") as f_matrix:
        for i in range(1, H.shape[0]-1):
            for j in range(i, H.shape[1]-1):
                if i not in x_idx and j not in x_idx:
                    dH_init = 0
                    dS_init = 0

                    multi_strand_system = len(x_idx) > 2

                    if multi_strand_system:
                        dH_init, dS_init = initiation_contrib(i, j, x_idx)

                    terminal_info = check_terminal(beads_sequences, i, j, x_idx) 
                    if terminal_info["is_terminal"]:
                        dH_terminal, dS_terminal = terminal_penalties(beads_sequences, i, j, terminal_info["condition"], material)
                    else:
                        dH_terminal = 0
                        dS_terminal = 0

                    dH_tot = np.round(H[i][j] + dH_init + dH_terminal, 2)
                    dS_tot = np.round(S[i][j] + dS_init + dS_terminal, 2)

                    f_matrix.write("dH[{}][{}] = {}\n".format(i, j, dH_tot))
                    f_matrix.write("dS[{}][{}] = {}\n".format(i, j, dS_tot))


########################################################
#### Functions to calculate the interaction matrix #####
########################################################

def dH_dS(seq1, seq2):
    dH = 0
    dS = 0
    weight = 0.5

    for k in range(len(seq1)-1):
        if seq1[k:k+2]+seq2[::-1][k:k+2] in dH_stack.keys():
            if k == 0 or k == len(seq1)-2:
                weight = 0.5
                dH += weight * dH_stack[ seq1[k:k+2]+seq2[::-1][k:k+2] ]
                dS += weight * dS_stack[ seq1[k:k+2]+seq2[::-1][k:k+2] ]

            else:
                weight = 1
                dH += dH_stack[ seq1[k:k+2]+seq2[::-1][k:k+2] ]
                dS += dS_stack[ seq1[k:k+2]+seq2[::-1][k:k+2] ]

        else:
            pass

    return dH, dS


def sliding_window(seqs, i, j):
    slide_time = len(seqs[i]) - len(seqs[j])
    HS = []
    seq_j = seqs[j+1][0]+seqs[j][::-1]+seqs[j-1][-1]
    for sl in range(slide_time + 1):
        if sl == 0:
            if sl == slide_time:
                HS.append( dH_dS(seqs[i-1][-1]+seqs[i]+seqs[i+1][0],  seq_j[::-1] ) )
            else:
                HS.append( dH_dS(seqs[i-1][-1]+seqs[i][:len(seqs[j])+1],  seq_j[::-1] ) )

        elif sl == slide_time:
            HS.append( dH_dS(seqs[i][sl-1:sl+len(seqs[j])]+seqs[i+1][0],  seq_j[::-1] ) )

        else:
            HS.append( dH_dS(seqs[i][sl-1:sl+len(seqs[j])+1], seq_j[::-1] ) )

    HS.sort(key=lambda x: x[0])

    return HS[0][0], HS[0][1]


def ordering_seqs(seq1, seq2, idx1, idx2):
    if len(seq1)>=len(seq2):
        return idx1, idx2
    else:
        return idx2, idx1


def interaction_matrix(seqs,material,salt_conc):
    dH = np.zeros((len(seqs),len(seqs)))
    dS = np.zeros((len(seqs),len(seqs)))

    for i in range(1, len(seqs)-1):
        for j in range(i+1, len(seqs)-1):

            longer, shorter = ordering_seqs(seqs[i], seqs[j], i, j)

            H, S = sliding_window(seqs, longer, shorter)
            dH[i][j] = H
            dS[i][j] = S

            #salt_correction = 0.368 * (len(seqs[i]) - 1.0)  * math.log(salt_conc)
            #dS[i][j] += salt_correction

    return dH, dS


def initiation_contrib(i, j, x_idx):
    lower_x = [e for e in x_idx if e<i][-1]     
    upper_x = [e for e in x_idx if e>i][0] 

    ij_not_same_strand = j > upper_x
    if ij_not_same_strand:
        size_strand_i = upper_x - lower_x - 1
        lower_x_j = [e for e in x_idx if e<j][-1]
        upper_x_j = [e for e in x_idx if e>j][0]
        size_strand_j = upper_x_j - lower_x_j - 1

        dH_init = dH_initiation/min(size_strand_i, size_strand_j)
        dS_init = dS_initiation/min(size_strand_i, size_strand_j)

        return dH_init, dS_init
    
    else:
        return 0, 0


def check_terminal(tris, i, j, x_idx):
    i_is_initial = i in [e+1 for e in x_idx]
    j_is_final = j in [e-1 for e in x_idx]
    terminal_1 = tris[i][0]+tris[j][-1]
    condition_1 = i_is_initial and j_is_final and terminal_1 in dH_terminal_penalties.keys()

    i_is_final = i in [e-1 for e in x_idx]
    j_is_initial = j in [e+1 for e in x_idx]
    terminal_2 = tris[j][0]+tris[i][-1]
    condition_2 = i_is_final and j_is_initial and terminal_2 in dH_terminal_penalties.keys()

    cond = "condition_i_init_j_fin" if condition_1 else "condition_i_fin_j_init" if condition_2 else None

    terminal_info = {
        "is_terminal": i_is_final and j_is_initial or i_is_final and j_is_initial,
        "condition": cond
        }
    
    return terminal_info


def terminal_penalties(beads_sequences, i, j, condition, material):
    if condition == "condition_i_init_j_fin":
        terminal = beads_sequences[i][0] + beads_sequences[j][-1]
        if material == "DNA":
            dH_term = dH_terminal_penalties[terminal]
            dS_term = dS_terminal_penalties[terminal]
            return dH_term, dS_term

        else:
            pre_terminal = beads_sequences[1][0] + beads_sequences[j][-2]
            if pre_terminal in dH_terminal_penalties[terminal]:
                dH_term = dH_terminal_penalties[terminal][pre_terminal]
                dS_term = dS_terminal_penalties[terminal][pre_terminal]
                return dH_term, dS_term
            else:
                return 0, 0
            
    elif condition == "condition_i_fin_j_init":
        terminal = beads_sequences[j][0] + beads_sequences[i][-1]
        if material == "DNA":
            dH_term = dH_terminal_penalties[terminal]
            dS_term = dS_terminal_penalties[terminal]
            return dH_term, dS_term

        else:
            pre_terminal = beads_sequences[j][1] + beads_sequences[i][-2]
            if pre_terminal in dH_terminal_penalties[terminal]:
                dH_term = dH_terminal_penalties[terminal][pre_terminal]
                dS_term = dS_terminal_penalties[terminal][pre_terminal]
                return dH_term, dS_term
            else:
                return 0, 0


def check_material(beads_sequences, material):
    material = material.upper()
    if material == 'DNA':
        beads_sequences = [beads_sequences[i].upper().replace('U','T') for i in range(len(beads_sequences))]
        from libs.DNA_SL import dH_stack, dS_stack, dH_initiation, dS_initiation, dH_terminal_penalties, dS_terminal_penalties
    elif material =='RNA':
        beads_sequences = [beads_sequences[i].upper().replace('T','U') for i in range(len(beads_sequences))]
        from libs.RNA_22 import dH_stack, dS_stack, dH_initiation, dS_initiation, dH_terminal_penalties, dS_terminal_penalties
    else:
        print("Error: material option must be specified! Choose between RNA and DNA")
        exit(0)

    global dH_stack
    global dH_initiation
    global dH_terminal_penalties
    global dS_stack
    global dS_initiation
    global dS_terminal_penalties

    return beads_sequences
  
  
########################################################
#### Depreciated functions #############################
########################################################
  
def fully_complementary_sequential_fixs(nucleotides_in_duplex, positions, all_edge_cases, duplex_to_particle, path_to_hb_info_file):
    pair_set, indexes, nearest_edge_case_min = full_sequential_group_distance_check(positions, all_edge_cases)
    nuc_dict = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    easy_fix = []
    alignments1 = []
    alignments2 = []
    monomer_info = np.array(nucleotides_in_duplex)
    
    hb_energy = read_hb_energy_file(path_to_hb_info_file)
    hb_energy = np.array(hb_energy['HB (oxDNA su)'])
    for idxes in indexes:
        idx_1 = idxes[0]
        idx_2 = idxes[1]

        monomers_1 = monomer_info[idx_1]
        monomers_1 = ''.join([mono.btype for mono in monomers_1])

        monomers_2 = monomer_info[idx_2]
        monomers_2 = ''.join([nuc_dict[mono.btype] for mono in monomers_2][::-1])
        
        alignment1, alignment2, seq_1_indexs, seq_2_indexs = smith_waterman(monomers_1, monomers_2)
        
        # if len(seq_1_indexs) > 1:
        # print(seq_1_indexs, seq_2_indexs)
        # print(idx_1, idx_2)
        print(alignment1)
        print(monomers_1, monomers_2)
        
        aligned_idxes_1 = [idx_1[i] for i in seq_1_indexs]
        aligned_idxes_2 = [idx_2[i] for i in seq_2_indexs]
        
        energies_1 = hb_energy[aligned_idxes_1]
        energies_2 = hb_energy[aligned_idxes_2]
        print(aligned_idxes_1, aligned_idxes_2)
        print(energies_1, energies_2)
        
        # new_key = len(duplex_to_particle) + 1
        # duplex_to_particle[int(new_key)] = [aligned_idxes_1, aligned_idxes_2[::-1]]
        # easy_fix.append([aligned_idxes_1, aligned_idxes_2[::-1]])


    return duplex_to_particle, easy_fix



def smith_waterman(seq1, seq2, match_score=1, mismatch_penalty=-1, gap_penalty=-2):
    # Initialize the matrix
    m, n = len(seq1), len(seq2)
    score_matrix = np.zeros((m+1, n+1))

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                diag_score = score_matrix[i-1][j-1] + match_score
            else:
                diag_score = score_matrix[i-1][j-1] + mismatch_penalty

            score_matrix[i][j] = max(
                0,
                diag_score,
                score_matrix[i-1][j] + gap_penalty,
                score_matrix[i][j-1] + gap_penalty
            )

    # Trace back from the highest scoring cell
    alignment1, alignment2 = '', ''
    i, j = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    seq_1_indexs = []
    seq_2_indexs = []

    while score_matrix[i][j] > 0:
        if i > 0 and score_matrix[i][j] == score_matrix[i-1][j] + gap_penalty:
            alignment1 = seq1[i-1] + alignment1
            alignment2 = '-' + alignment2
            seq_1_indexs.append(i-1)
            i -= 1
        elif j > 0 and score_matrix[i][j] == score_matrix[i][j-1] + gap_penalty:
            alignment1 = '-' + alignment1
            alignment2 = seq2[j-1] + alignment2
            seq_2_indexs.append(j-1)
            j -= 1
        else:
            alignment1 = seq1[i-1] + alignment1
            alignment2 = seq2[j-1] + alignment2
            seq_1_indexs.append(i-1)
            seq_2_indexs.append(j-1)
            i -= 1
            j -= 1
    seq_1_indexs = seq_1_indexs[::-1]
    seq_2_indexs = seq_2_indexs[::-1]
    
    return alignment1, alignment2, seq_1_indexs, seq_2_indexs