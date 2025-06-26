import MDAnalysis as mda
import re
import os
import datetime
import glob
import json
import numpy as np
from MDAnalysis.coordinates.PDB import PDBReader
from MDAnalysis.coordinates.LAMMPS import DumpReader
from collections import defaultdict
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from MDAnalysis.lib.distances import distance_array
from multiprocessing import Pool


ALL_TYPE_MAP = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og"
]

RADII = {
    "Na": 1.02,
    "K": 1.38,
    "Rb": 1.52,
    "NH4": 1.40,
    "O": 1.4,
}

def calculate_distances(atom_group1, atom_group2):
    box = atom_group1.universe.dimensions
    dists = distance_array(atom_group1.positions, atom_group2.positions, box=box, backend='OpenMP')
    return dists

def identify_ammonium(universe, type_map):
    if isinstance(universe.trajectory, PDBReader):
        nitrogens = universe.select_atoms('name N')
        hydrogens = universe.select_atoms('name H')
    elif isinstance(universe.trajectory, DumpReader):
        nitrogens = universe.select_atoms(f'type {type_map["N"]}')
        hydrogens = universe.select_atoms(f'type {type_map["H"]}')

    nh_dists = calculate_distances(nitrogens, hydrogens)
    ammonium_indices = np.sum(nh_dists < 1.27, axis=1) == 4
    ammonium_atoms = nitrogens[ammonium_indices]
    return ammonium_atoms

def calculate_XB_collisions(universe, type_map, B_element, start_time, end_time):
    print(len(universe.trajectory))
    if isinstance(universe.trajectory, PDBReader):
        oxygens = universe.select_atoms('name O')
        if B_element == 'NH4':
            B_atoms = identify_ammonium(universe, type_map)
        else:
            B_atoms = universe.select_atoms(f'name {B_element}')
    elif isinstance(universe.trajectory, DumpReader):
        oxygens = universe.select_atoms(f'type {type_map["O"]}')
        if B_element == 'NH4':
            B_atoms = identify_ammonium(universe, type_map)
        else:
            B_atoms = universe.select_atoms(f'type {type_map[B_element]}')

    # Define collision distance threshold
    collision_threshold = 1.2
    d_cutoff = (RADII[B_element] + RADII["O"]) * collision_threshold

    N_X = len(oxygens)
    T_sim = (end_time - start_time) * universe.trajectory.dt # Simulation time for the current interval
    print(T_sim)
    N_coll = 0
    total_coll_time = 0
    prev_collisions = set()

    # Iterate over the trajectory for the current time interval
    for ts in tqdm(universe.trajectory[start_time:end_time:2]):
        dists = calculate_distances(oxygens, B_atoms)
        collisions = np.argwhere(dists <= d_cutoff)
        current_collisions = set([tuple(c) for c in collisions])

        new_collisions = current_collisions - prev_collisions
        N_coll += len(new_collisions)

        if len(current_collisions) > 0:
            if len(prev_collisions) == 0:
                start_time_ts = ts.time
            elif len(current_collisions.intersection(prev_collisions)) == 0:
                end_time_ts = ts.time
                total_coll_time += end_time_ts - start_time_ts
                start_time_ts = ts.time
        elif len(prev_collisions) > 0:
            end_time_ts = ts.time
            total_coll_time += end_time_ts - start_time_ts

        prev_collisions = current_collisions

    if len(prev_collisions) > 0:
        end_time_ts = universe.trajectory.totaltime
        total_coll_time += end_time_ts - start_time_ts

    CF_XB = N_coll / (T_sim * N_X)

    return CF_XB

COLOR_DICT = {
    'dap-1': "#D9B426",
    "dap-2": "#5B338E",
    "dap-3": "#BC358F",
    "dap-4": "#41AA35"
}

def process_single_traj(args):
    dpmd_traj, type_map, TIMESTEP, INTERVAL = args
    _temperature = dpmd_traj.split('/')[0]
    _system = dpmd_traj.split('/')[1]
    print(f"Processing system: {_system}")
    if 'dap-1' in _system:
        B_element = 'Na'
    elif _system == 'dap-2':
        B_element = "K"
    elif _system == 'dap-3':
        B_element = 'Rb'
    elif _system == 'dap-4':
        B_element = 'NH4'
    else:
        return None

    universe = mda.Universe(dpmd_traj, format='LAMMPSDUMP', dt=TIMESTEP * 1e-3 * INTERVAL, atom_style='ATOMS element id type x y z')

    # Segment the trajectory and calculate CF_XB for each interval
    step_intervals = [
        (1, 2), (2, 4), (4, 12), (12, 28), (28, 60), (60, 124), (124, 252), (252, 508), (508, 999)
    ]
    CF_XB_values = []
    times = []

    for start, end in step_intervals:
        CF_XB = calculate_XB_collisions(universe, type_map, B_element, start, end)
        CF_XB_values.append(CF_XB)
        times.append(start * TIMESTEP * INTERVAL * 1e-3) 

    # Store results
    result = {
        f'{_temperature}_{_system}_{B_element}': {
            'times': times,
            'CF_XB': CF_XB_values
        },
        'system': _system
    }
    return result


if __name__ == '__main__':
    TIMESTEP = 0.1  # fs
    INTERVAL = 100
    type_map = {element: idx + 1 for idx, element in enumerate(ALL_TYPE_MAP)}
    traj_temp = glob.glob(f'1666/*/dump_nvt_freq-100.lammpstrj')
    traj_temp = sorted(traj_temp, key=lambda x: int(x.split('/')[0]))

    results = {}
    # Plotting setup
    plt.figure(figsize=(4, 4))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    with Pool() as pool:
        args_list = [(dpmd_traj, type_map, TIMESTEP, INTERVAL) for dpmd_traj in traj_temp]
        pool_results = pool.map(process_single_traj, args_list)

    for res in pool_results:
        if res is not None:
            key = list(res.keys())[0]
            if key != 'system':
                results[key] = res[key]
                _system = res['system']
                times = res[key]['times']
                CF_XB_values = res[key]['CF_XB']
                plt.plot(times, CF_XB_values, c=COLOR_DICT[_system], marker='s', label=f'{_system.upper()}')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(8e-3, 1e2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.xlabel('Time (ps)')
    plt.ylabel('C.F.$_\mathrm{X-B}$ (ps$^{-1}$)')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('collision.png', dpi=300)
    plt.savefig('collision.eps')

    # Output results
    print(json.dumps(results, indent=4))
