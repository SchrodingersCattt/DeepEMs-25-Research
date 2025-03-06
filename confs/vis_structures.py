
import os
import glob
import matplotlib.pyplot as plt
import ase
import re
print(ase.__version__)
from ase.io import read
from ase.visualize.plot import plot_atoms

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def get_space_group(cif_path):
    structure = Structure.from_file(cif_path)
    analyzer = SpacegroupAnalyzer(structure)
    
    space_group_symbol = analyzer.get_space_group_symbol()
    space_group_number = analyzer.get_space_group_number()
    
    return space_group_symbol, space_group_number


def find_supercell_string(system_name):
    match = re.search(r'(?<!\d)\d{3}(?!\d)', system_name)
    if match:
        return match.group(0)
    else:
        return "111"

custom_colors = {
    'C': '#686868',
    'H': '#eeeeee',
    'N': '#3787C0',
    'O': '#C13838',
    'Cl': '#71C14C',
    'Na': '#D9B426',  
    'K': '#6A2590',
    'Rb': '#cF50a4',
    'Cu': '#00ADFF',
    'Ag': '#cccccc',
    'Hg': '#ff0f00',
    'Tl': '#C1E573',
    'Pb': '#e48300',
}

def plot_poscar(poscar):
    sg_symbol, sg_number = get_space_group(poscar)
    atoms = read(poscar)
    rotation = '-90x' 
    
    colors = [custom_colors.get(symbol, 'blue') for symbol in atoms.get_chemical_symbols()]
    plot_atoms(atoms, axs[i], 
            rotation=rotation,
            radii=1.0, 
            scale=1.0,  
            offset=(0.5, 0.5),  
            colors=colors               
    )
    
    _system_name = os.path.basename(os.path.dirname(poscar)).replace(
        "DAP4-order", "DAP-4"
    ).replace(
        "DAP-7_222-order", "DAP-7"
    )
    dir_name = "_".join(_system_name.split('_')[:2]) if not 'icsd' in _system_name else "_".join(_system_name.split('_')[:3])
    if len(dir_name.split('_', 1)) > 1:
        sysname, _sysid = dir_name.split('_', 1)
    else:
        sysname, _sysid = dir_name, ' '
    sys_id = _sysid.replace('_', '-')
    axs[i].axis('off')
    at = ""
    if "DAP-4" in _system_name:
        sg_symbol, sg_number = "Pa-3", 205
    elif 'DAP-7' in _system_name:
        sg_symbol, sg_number =  "P2_1/m", 11   
    if "relaxed" in _system_name:
        at += f"\n  Relaxed"
    elif "DAP-4" in _system_name or 'DAP-7' in _system_name:
        at += f"\n  Removed disorder"
    else:
        at += f"None."
        
    supercell = "×".join(find_supercell_string(_system_name))
    if 'DAP-7' in _system_name:
        supercell = "×".join("121")
    axs[i].text(1.0, 1.02, f'{sysname}, {sys_id}\nSpace group: ${sg_symbol}$ ({sg_number})\nSupercell: {supercell}\n\nAtom numb: {len(atoms)}\nAdditional treatment: {at}', 
                transform=axs[i].transAxes,
                verticalalignment='top', 
                horizontalalignment='left', 
                fontsize=18, 
                color='black')
    axs[i].text(0.05, 1.05, f'{chr(ord("a") + i)}',  
                transform=axs[i].transAxes,
                fontsize=24,
                fontweight='bold',
                color='black',
                verticalalignment='bottom',
                horizontalalignment='left')
            

if __name__ == '__main__':
    _systems_list = glob.glob("*/POSCAR")
    structures = sorted(
        _systems_list,
        key=lambda x: (
            "_".join(
                os.path.basename(os.path.dirname(x)).replace("DAP4-order", "DAP-4").replace("DAP-7_222-order", "DAP-7").split('_')[:2]
            ) if "icsd" not in os.path.basename(os.path.dirname(x)) else 
            "_".join(os.path.basename(os.path.dirname(x)).replace("DAP4-order", "DAP-4").replace("DAP-7_222-order", "DAP-7").split('_')[:3])
        )
    )
    n = len(structures)
    ncols = 4  
    nrows = (n + ncols - 1) // ncols
    figsize = (ncols * 6, nrows * 3) 
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    plt.rcParams['font.family'] = 'Arial'
    axs = axs.flat

    for i, poscar in enumerate(structures):
        try:
            plot_poscar(poscar)
        except Exception as e:
            print(f"Error processing {poscar}: {str(e)}")
            axs[i].axis('off')

    for j in range(n, nrows * ncols):
        axs[j].axis('off')

    plt.tight_layout(pad=2.0, w_pad=4.0, h_pad=0.0)
    plt.savefig('crystal_structures.png', dpi=300)