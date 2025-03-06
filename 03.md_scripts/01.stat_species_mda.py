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

def calculate_distances(atom_group1, atom_group2):    
    box = atom_group1.universe.dimensions
    dists = distance_array(atom_group1.positions, atom_group2.positions, box=box, backend='OpenMP')
    return dists


def identify_molecules(universe, type_map):
    water_counts = []
    hcl_counts = []
    ammonium_counts = []
    ammonia_counts = []
    perchlorate_counts = []
    nitrogen_counts = []
    CO2_counts = []
    CO_counts = []

    frames = []

    for ts in tqdm(universe.trajectory[::10]):
        if isinstance(universe.trajectory, PDBReader):
            oxygens = universe.select_atoms('name O')
            hydrogens = universe.select_atoms('name H')
            chlorines = universe.select_atoms('name Cl')
            nitrogens = universe.select_atoms('name N')
            carbons = universe.select_atoms('name C')
        elif isinstance(universe.trajectory, DumpReader):
            oxygens = universe.select_atoms(f'type {type_map["O"]}')
            hydrogens = universe.select_atoms(f'type {type_map["H"]}')
            chlorines = universe.select_atoms(f'type {type_map["Cl"]}')
            nitrogens = universe.select_atoms(f'type {type_map["N"]}')
            carbons = universe.select_atoms(f'type {type_map["C"]}')
        
        start = time()
        oh_dists = calculate_distances(oxygens, hydrogens)
        nh_dists = calculate_distances(nitrogens, hydrogens)
        cl_h_distances = calculate_distances(chlorines, hydrogens)
        cl_o_distances = calculate_distances(chlorines, oxygens)
        co_dists = calculate_distances(carbons, oxygens)
        cc_dists = calculate_distances(carbons, carbons)
        ch_dists = calculate_distances(carbons, hydrogens)
        nn_dists = calculate_distances(nitrogens, nitrogens)
        end = time()
        
        water_indices = np.sum((oh_dists < 1.26), axis=1) == 2
        water_count = np.sum(water_indices)

        hcl_indices = np.sum(cl_h_distances < 1.56, axis=1) == 1
        hcl_count = np.sum(hcl_indices)

        ammonium_indices = np.sum(nh_dists < 1.27, axis=1) == 4
        ammonium_count = np.sum(ammonium_indices)

        ammonia_indices = np.sum(nh_dists < 1.27, axis=1) == 3
        ammonia_count = np.sum(ammonia_indices)

        perchlorate_atoms = np.sum(cl_o_distances < 1.7, axis=1) == 4
        perchlorate_count = np.sum(perchlorate_atoms)

        nitrogen_atoms = np.sum((nn_dists < 1.15) & (nn_dists > 0.01), axis=1) == 1
        nitrogen_count = np.sum(nitrogen_atoms) // 2

        end = time()

        min_shape = min(co_dists.shape[1], cc_dists.shape[1], ch_dists.shape[1])
        co_dists = co_dists[:, :min_shape]
        cc_dists = cc_dists[:, :min_shape]
        ch_dists = ch_dists[:, :min_shape]

        CO2_condition = (co_dists < 1.3) & (cc_dists > 1.2) & (ch_dists > 1.0)
        CO2_atoms = np.sum(CO2_condition, axis=1) == 2
        CO2_count = np.sum(CO2_atoms)

        CO_condition = (co_dists < 1.2) & (cc_dists > 1.2) & (ch_dists > 1.0)
        CO_atoms = np.sum(CO_condition, axis=1) == 1
        CO_count = np.sum(CO_atoms)

        end = time()

        # Store the data
        frames.append(ts.frame)
        water_counts.append(water_count)
        hcl_counts.append(hcl_count)
        ammonium_counts.append(ammonium_count)
        ammonia_counts.append(ammonia_count)
        perchlorate_counts.append(perchlorate_count)
        nitrogen_counts.append(nitrogen_count)
        CO2_counts.append(CO2_count)
        CO_counts.append(CO_count)

        end = time()

    return {
        "frames": frames,
        "water": water_counts,
        "hcl": hcl_counts,
        "ammonium": ammonium_counts,
        "ammonia": ammonia_counts,
        "perchlorate": perchlorate_counts,
        "nitrogen": nitrogen_counts,
        "carbon_dioxide": CO2_counts,
        "carbon_monoxide": CO_counts
    }


def plot_species(stat_dict, species_list, label, linestyle, **kwargs):
    times = np.array(stat_dict["frames"]) * TIMESTEP * INTERVAL / 1000  # ps
    for i, (species, data_key, color) in enumerate(species_list):
        ax1 = plt.subplot(2, 4, i + 1)

        data = np.array(stat_dict[data_key])
        #smoothed_data = savgol_filter(data, window_length=10, polyorder=1)
        smoothed_data = data
        ax1.set_xlim(1e-1, 1e2)
        ax1.set_xscale('log')
        plot_text = kwargs.get('plot_text', False)
        alpha = kwargs.get('alpha', 1)
        '''if species == 'Ammonium' or species == 'Perchlorate':
            _x = 0.95
            _ha = 'right'
        else:
            _x = 0.05
            _ha = 'left'
        '''
 
        ax1.plot(times, smoothed_data, linestyle=linestyle, label=label) 
        if plot_text:
            ax1.text(0.02, 1.02, f'{species}', 
                transform=plt.gca().transAxes, 
                horizontalalignment="left", 
                verticalalignment='bottom')

        if i >= 4:
            ax1.set_xlabel('Time (ps)')      
        if i % 4 == 0:
            ax1.set_ylabel('Count')
        #ax1.grid(True)
        ax1.legend(ncols=1, fontsize=10, handlelength=1.0, frameon=False, handletextpad=0.2, columnspacing=0.2)

def convert_numpy_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(item) for item in obj]
    return obj


excludes = ['dap-1', 'dap-2', 'dap-3', 'dap-4']

if __name__ == "__main__":    
    TIMESTEP = 0.1 #fs
    INTERVAL = 100
    stat_dict_batch = defaultdict(dict)
    species_list = [
        ("Water", "water", '#cb4455'),
        ("Hydrogen chloride", "hcl", '#7A1C99'),
        ("Ammonia", "ammonia", '#EEbA22'),
        ("Ammonium", "ammonium", '#293FAA'),
        ("Perchlorate", "perchlorate", '#34ab56'),
        ("Nitrogen", "nitrogen", '#456bfa'),
        ("Carbon dioxide", "carbon_dioxide", '#111122'),
        ("Carbon monoxide", "carbon_monoxide", '#106059')
    ]
    
    type_map = {element: idx + 1 for idx, element in enumerate(ALL_TYPE_MAP)}    
    traj_temp = glob.glob(f'*/*/dump_nvt_freq-100.lammpstrj')
    traj_temp = sorted(traj_temp, key=lambda x: int(x.split('/')[0]))
    for idx, dpmd_traj in enumerate(traj_temp):
        if os.path.basename(os.path.dirname(dpmd_traj)) not in excludes:
            continue
        _temperature = dpmd_traj.split('/')[0]
        _system = dpmd_traj.split('/')[1]
        if (int(_temperature) > 2600):
            continue
        universe = mda.Universe(dpmd_traj, format='LAMMPSDUMP', dt=TIMESTEP * 1e-3 * INTERVAL, atom_style='ATOMS element id type x y z')
        stat_dict = identify_molecules(universe, type_map)
        
        # Convert numpy arrays to lists
        for key, value in stat_dict.items():
            if isinstance(value, np.ndarray):
                stat_dict[key] = value.tolist()
        
        stat_dict_batch[_system][_temperature] = convert_numpy_to_native(stat_dict)

    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Arial'
    plt.style.use('seaborn-v0_8-deep')
    plt.figure(figsize=(12, 6))
    for idx, system in enumerate(list(stat_dict_batch.keys())):
        for jdx, temperature in enumerate(list(stat_dict_batch[system].keys())):
            plot_text = True
            stat_dict = stat_dict_batch[system][temperature]
            plot_species(stat_dict, species_list, label=temperature, linestyle='-', plot_text=plot_text)
    
        plt.suptitle(f'{system.upper()}')
        plt.tight_layout()  
        plt.savefig(f'products_{system}.png', dpi=300)  
        plt.savefig(f'products_{system}.eps') 
        plt.clf()

    print('Done.')
