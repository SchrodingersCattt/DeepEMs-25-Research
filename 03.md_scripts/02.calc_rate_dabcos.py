import os
import glob
import numpy as np
import pandas as pd
import ase.geometry
import ase.units
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from time import sleep
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union
from scipy.stats import linregress
from scipy.stats import t
from tqdm import tqdm

def read_species(
    specfile: Union[str, Path],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    step_idx = []
    n_species = {}
    
    with open(specfile) as f:
        lines = f.readlines()
        nsteps = len(lines)
        
        step_idx = np.empty(nsteps, dtype=int)
        
        for ii, line in enumerate(lines):
            s = line.split()
            step_idx[ii] = int(s[1].strip(":"))
            
            species = s[2::2]
            counts = [int(x) for x in s[3::2]]
            
            for ss, nn in zip(species, counts):
                if ss not in n_species:
                    n_species[ss] = np.zeros(nsteps, dtype=int)
                n_species[ss][ii] = nn
    return step_idx, n_species


def read_reactions(reacfile: Union[str, Path]) -> List[Tuple[int, Counter, str]]:
    occs = []
    with open(reacfile) as f:
        for line in f:
            s = line.split()
            occs.append((int(s[0]), Counter(s[1].split("->")[0].split("+")), s[1]))
    return occs


def calculate_rate(
    specfile: Union[str, Path], reacfile: Union[str, Path], 
    cell: np.ndarray, timestep: float, max_stat:int=10, **kwargs
) -> Dict[str, Tuple[float, int]]:
    ase_cell = ase.geometry.Cell(cell)

    timestep *= 10**-15  # fs to s
    step_idx, n_species = read_species(specfile)
    occs = read_reactions(reacfile)

    time_int = (step_idx[1] - step_idx[0]) * timestep
    print(f"Time interval: {step_idx[1] - step_idx[0]}")
    volume = ase_cell.volume
    volume *= 10**-24  # Ang^3 to cm^3
    volume_times_na = volume * ase.units.mol  # V * NA

    rates = {}
    for occ, reacts, reactions in occs:
        if occ <= max_stat:
            continue
        n_react = np.array([n_species.get(kk, None) for kk in reacts.keys()])
        nu = np.array(list(reacts.values()))
        c_po = np.power(
            n_react / volume_times_na,
            np.repeat(nu, n_react.shape[1]).reshape(n_react.shape),
        )
        c_tot = np.sum(np.prod(c_po, axis=0))
        k = occ / (volume_times_na * time_int * c_tot)
        rates[reactions] = (k, occ)
    return rates
    

def get_cell(dump_file):
    with open(dump_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("BOX BOUNDS not found in the file")
            if line.startswith('ITEM: BOX BOUNDS'):
                break

        x_lo, x_hi, xy = map(float, f.readline().split())
        y_lo, y_hi, xz = map(float, f.readline().split())
        z_lo, z_hi, yz = map(float, f.readline().split())

    lx = x_hi - x_lo
    ly = y_hi - y_lo
    lz = z_hi - z_lo

    return np.array([
        [lx,  0.0, 0.0],
        [xy,  ly,  0.0],
        [xz,  yz,  lz]
    ])


def reg_dabco(smiles):
    if "[H][C]([H])[C]([H])([H])[N]1([H])[C]([H])([H])[C]([H])([H])[N]([H])[C]([H])([H])[C]1([H])[H]" in smiles:
        smiles = smiles.replace("[H][C]([H])[C]([H])([H])[N]1([H])[C]([H])([H])[C]([H])([H])[N]([H])[C]([H])([H])[C]1([H])[H]", "H2DABCO_bridge-opened-on-C-N")
    if "[H][C]1([H])[C]([H])([H])[N]2([H])[C]([H])([H])[C]([H])([H])[N]1([H])[C]([H])([H])[C]2([H])[H]" in smiles:
        smiles = smiles.replace("[H][C]1([H])[C]([H])([H])[N]2([H])[C]([H])([H])[C]([H])([H])[N]1([H])[C]([H])([H])[C]2([H])[H]", "H2DABCO")
    if "[H][C]1([H])[N]2[C]([H])([H])[C]([H])([H])[N]([H])([C]1([H])[H])[C]([H])([H])[C]2([H])[H]" in smiles:
        smiles = smiles.replace("[H][C]1([H])[N]2[C]([H])([H])[C]([H])([H])[N]([H])([C]1([H])[H])[C]([H])([H])[C]2([H])[H]", "H2DABCO_deprotonated-on-N")
    if "[H][C]1[C]([H])([H])[N]2([H])[C]([H])([H])[C]([H])([H])[N]1([H])[C]([H])([H])[C]2([H])[H]" in smiles:
        smiles = smiles.replace("[H][C]1[C]([H])([H])[N]2([H])[C]([H])([H])[C]([H])([H])[N]1([H])[C]([H])([H])[C]2([H])[H]", "H2DABCO_deprotonated-on-C")
    if "[H][C]([H])[N]1([H])[C]([H])([H])[C]([H])([H])[N]([H])([C]([H])[H])[C]([H])([H])[C]1([H])[H]" in smiles:
        smiles = smiles.replace("[H][C]([H])[N]1([H])[C]([H])([H])[C]([H])([H])[N]([H])([C]([H])[H])[C]([H])([H])[C]1([H])[H]", "HDABCO_bridge-opened-on-C-C")
    if "[H][O][C]1([H])[C]([H])([H])[N]2([H])[C]([H])([H])[C]([H])([H])[N]1([H])[C]([H])([H])[C]2([H])[H]" in smiles:
        smiles = smiles.replace("[H][O][C]1([H])[C]([H])([H])[N]2([H])[C]([H])([H])[C]([H])([H])[N]1([H])[C]([H])([H])[C]2([H])[H]", "H2DABCO-OH")
    return smiles


def visualize_smiles(smiles, save_dir='imgs/'):
    from rdkit.Chem import rdchem
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return
        try:
            mols = [mol]
            d = rdMolDraw2D.MolDraw2DCairo(1200, 800)
            d.DrawMolecules(mols)
            d.FinishDrawing()
            img = d.GetDrawingText()

            with open(f'{save_dir}{reg_dabco(smiles)}.png', 'wb') as f:
                f.write(img)
        except rdchem.AtomValenceException as e:
            print(f"Valence error in SMILES {smiles}: {e}")
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
    
def visualize_reactions(rates_dict):
    all_smiles = set()
    for reaction in rates_dict.keys():
        reactants, products = reaction.split('->')
        reactant_list = reactants.split('+')
        product_list = products.split('+')
        all_smiles.update(reactant_list)
        all_smiles.update(product_list)

    for smiles in all_smiles:
        visualize_smiles(smiles)


def calc_activative_ener_from_df(df):
    activation_energies = []
    df['Temperature'] = df['Temperature'].astype(float)
    df['Rate Constant'] = df['Rate Constant'].astype(float)
    for system, supergroup in df.groupby('System'):
        for reaction, group in supergroup.groupby('Reaction'):
            temperature = group['Temperature'].values
            rate = group['Rate Constant'].values
            Ea = calc_activative_energy(temperature, rate)
            activation_energies.append({'System': system, 'Reaction': reaction, 'Activation Energy (kJ/mol)': Ea})
    return pd.DataFrame(activation_energies)

COLOR_DICT = {
    'dap-1': "#D9B426",
    "dap-2": "#5B338E",
    "dap-3": "#BC358F",
    "dap-4": "#41AA35"
}

def plot_lnK_1T(df):
    df['Temperature'] = df['Temperature'].astype(float)
    df['Rate Constant'] = df['Rate Constant'].astype(float)
    
    reactions = df['Reaction'].unique()
    systems = df['System'].unique()
    nrows = len(reactions)
    ncols = len(systems)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows), squeeze=False)
    
    reaction_limits = {}
    
    # First pass: collect axis limits
    for i, reaction in enumerate(reactions):
        if "H2DABCO_" in reaction.split('->')[0]:
            continue
        
        all_inv_temp = []
        all_ln_rate = []
        for system in systems:
            group = df[(df['Reaction'] == reaction) & (df['System'] == system)]
            if not group.empty:
                temperature = group['Temperature'].values
                rate = group['Rate Constant'].values
                sorted_pairs = sorted(zip(temperature, rate), key=lambda x: x[0])
                temperature, rate = zip(*sorted_pairs)
                inv_temp = 1000 / np.array(temperature)
                ln_rate = np.log(rate)
                all_inv_temp.extend(inv_temp)
                all_ln_rate.extend(ln_rate)
        
        if all_inv_temp:
            xmin, xmax = min(all_inv_temp), max(all_inv_temp)
            ymin, ymax = min(all_ln_rate), max(all_ln_rate)
            xpad = (xmax - xmin) * 0.1
            ypad = (ymax - ymin) * 0.1
            reaction_limits[reaction] = ((xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))
        else:
            reaction_limits[reaction] = (None, None)
    
    # Second pass: plot with uncertainties
    for i, reaction in enumerate(reactions):
        if "H2DABCO_" in reaction.split('->')[0]:
            continue
        
        xlim, ylim = reaction_limits.get(reaction, (None, None))
        systems = sorted(systems, key=lambda x: int(x.split('-')[1]))
        for j, system in enumerate(systems):
            ax = axes[i, j]
            group = df[(df['Reaction'] == reaction) & (df['System'] == system)]
            
            if group.empty or xlim is None or ylim is None:
                ax.axis('off')
                continue
                
            temperature = group['Temperature'].values
            rate = group['Rate Constant'].values
            sorted_pairs = sorted(zip(temperature, rate), key=lambda x: x[0])
            temperature, rate = zip(*sorted_pairs)
            inv_temp = 1000 / np.array(temperature)
            ln_rate = np.log(rate)
            
            ax.scatter(inv_temp, ln_rate, 
                marker='o',
                s=150,
                facecolors='none', 
                edgecolors=COLOR_DICT[system],
                linewidths=5
            )

            _x_min = inv_temp.min()
            _x_max = inv_temp.max()
            _x_pad = (_x_max - _x_min) * 0.1 
            _x_est = np.linspace(_x_min - _x_pad, _x_max + _x_pad, 100)

            # Perform linear regression with error metrics
            slope, intercept, r_value, p_value, slope_stderr = linregress(inv_temp, ln_rate)
            _y_est = slope * _x_est + intercept

            # Calculate intercept uncertainty
            n = len(inv_temp)
            if n > 2:
                residuals = ln_rate - (slope * inv_temp + intercept)
                dof = n - 2
                mse = np.sum(residuals**2) / dof
                s_err = np.sqrt(mse)
                x_mean = np.mean(inv_temp)
                SS_xx = np.sum((inv_temp - x_mean)**2)
                intercept_stderr = s_err * np.sqrt(1/n + x_mean**2 / SS_xx)
                
                # Confidence bands calculation
                se = s_err * np.sqrt(1/n + (_x_est - x_mean)**2 / SS_xx)
                t_val = t.ppf(0.975, dof)
                upper = _y_est + t_val * se
                lower = _y_est - t_val * se
                ax.fill_between(_x_est, lower, upper, color=COLOR_DICT[system], alpha=0.3, linewidth=0, zorder=0)
            
            ax.plot(_x_est, _y_est, linestyle='--', color='k')
            
            # Format text with uncertainties
            R = 8.31446261815324  # in kJ/(mol*K)
            Ea = -slope * R
            Ea_stderr = slope_stderr * R
            text = (f"{system.upper()}\n"
                    f"ln$A$ = {intercept:.2f}±{intercept_stderr:.2f}\n"
                    f"$E_a$ = {Ea:.2f}±{Ea_stderr:.2f} kJ/mol\n"
                    f"$R^2$ = {r_value**2:.3f}")
            ax.text(0.05, 0.05, text, transform=ax.transAxes, 
                    fontsize=12, verticalalignment='bottom')
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
            if j == 3:
                ax.set_title(f"{reaction.replace('DABCO', 'A').replace('deprotonated-on', 'depro.')}", pad=20)
            ax.set_xlabel(r"1000/$T$ (K$^{-1}$)")
            if j == 0:
                ax.set_ylabel(r"ln($k$)")

    plt.tight_layout()



def calc_activative_energy(temperature, rate):
    R = 8.31446261815324  # kJ/(mol*K)
    ln_rate = np.log(rate)
    inv_temp = 1 / np.array(temperature)
    slope, _, _, _, _ = linregress(inv_temp, ln_rate)
    Ea = -slope * R
    return Ea


if __name__ == "__main__":
    wkplaces = glob.glob("*/*/")
    timestep = 0.1  # in unit fs
    summarized_df = pd.DataFrame(columns=['System', 'Temperature', 'Reaction', 'Rate Constant', 'Occurrence'])
    to_stat = ["dap-1", "dap-2", "dap-3", "dap-4"]
    for ww in wkplaces:
        system = ww.split('/')[1]
        if not system in to_stat:
            continue
        temperature = ww.split('/')[0]
        if 'imgs' in ww or '_' in ww or int(temperature) >= 2600:
            continue
        dump_file = os.path.join(ww, "dump_nvt_freq-100.lammpstrj")
        species_file = os.path.join(ww, "dump_nvt_freq-100.lammpstrj.species")
        rxn_file = os.path.join(ww, "dump_nvt_freq-100.lammpstrj.reactionabcd")
        if not os.path.exists(dump_file) or not os.path.exists(species_file) or not os.path.exists(rxn_file):
            continue
        cell = get_cell(dump_file)
        rates = calculate_rate(species_file, rxn_file, cell, timestep, max_stat=10)
        filtered_rates = sorted(
            ((k, v) for k, v in rates.items() if "[H][C]1([H])[C]([H])([H])[N]2([H])[C]([H])([H])[C]([H])([H])[N]1([H])[C]([H])([H])[C]2([H])[H]" in k),
            key=lambda item: item[1][0],
            reverse=True
        )
        formatted = [(reg_dabco(k), f'{v[0]:.5e}', v[1]) for k, v in filtered_rates]
        _df = pd.DataFrame([(system, temperature, reaction, rate, occurrence) for reaction, rate, occurrence in formatted],
                           columns=['System', 'Temperature', 'Reaction', 'Rate Constant', 'Occurrence'])
        summarized_df = pd.concat([summarized_df, _df], ignore_index=True)
        tmp_df = pd.DataFrame(formatted, columns=['Reaction', 'Rate Constant', 'Occurrence'])
        print(system, temperature)
        print(tmp_df.to_markdown(index=False))
        print()
    activation_energy_df = calc_activative_ener_from_df(summarized_df)
    for reaction, group in activation_energy_df.groupby('Reaction'):
        if not "deprotonated" in reaction and not 'bridge - opened' in reaction:
            continue
        print(f"Reaction: {reaction}")
        print(group.to_markdown(index=False))
        print()
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Arial'
    plt.style.use('seaborn-v0_8-deep')
    plot_lnK_1T(summarized_df)
    plt.tight_layout()
    plt.savefig('lnk-1T.pdf', dpi=300)
    plt.close()