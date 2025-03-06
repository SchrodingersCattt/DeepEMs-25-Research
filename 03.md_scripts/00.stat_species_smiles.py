import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


COLOR_DICT = {
    'dap-1': "#D9B426",
    "dap-2": "#A57BC9",
    "dap-3": "#C66FA9",
    "dap-4": "#6DBA6A",
    "dap-5": "#848484",
    "dap-6": "#BB6655",
    "dap-7": "#5599EE",
    "dap-m4": "#12569b"
}

def adjust_lightness(color_hex, factor=0.5):
    rgb = plt.cm.colors.to_rgb(color_hex)
    h, s, v = rgb_to_hsv(rgb)
    new_v = min(1.0, v * factor)
    return plt.cm.colors.to_hex(hsv_to_rgb((h, s, new_v)))
    
def integrate_species(times, species):
    return np.trapz(species, times)

def stat_species(species_file, to_stat=None):
    times = []
    species = []
    with open(species_file, 'r') as ff:
        ll = ff.readlines()
    
    if isinstance(to_stat, str):
        for l in ll:
            info = l.split(': ')
            time_info = float(info[0].split('Timestep')[-1])
            times.append(time_info)
            species_info = info[1].split(' ')
            for idx, item in enumerate(species_info):
                if item == to_stat:
                    stat = float(species_info[idx+1])
                    break
                else:
                    stat = 0
            species.append(stat)

    if isinstance(to_stat, list):  
        for l in ll:
            stat = 0
            info = l.split(': ')
            time_info = float(info[0].split('Timestep')[-1])
            times.append(time_info)
            species_info = info[1].split(' ')
            for idx, item in enumerate(species_info):
                for x in to_stat:
                    if item == x:
                        stat += float(species_info[idx+1])
            species.append(stat)
    times = [x - times[0] for x in times]
    return times, species

if __name__ == "__main__":
    TIMESTEP = 0.1  # fs
    species_files = glob.glob(f'*/*/dump_nvt_freq-100.lammpstrj.species')
    systems = set()
    temperatures = set()
    for species_file in species_files:
        _temperature = species_file.split('/')[0]
        _system = species_file.split('/')[1]
        if (int(_temperature) > 2500) or (_system in ["dap-5", "dap-6", "dap-7", "dap-m4"]):
            continue
        systems.add(_system)
        temperatures.add(_temperature)

    plt.figure(figsize=(3 * len(temperatures), 2 * len(systems)))
    subplot_idx = 1
    systems = sorted(systems, key=lambda x: int(x.split('-')[-1]))
    for system in systems:
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 12
        base_color = COLOR_DICT[system]
        gradient_factors = [1.2, 0.9, 0.6]
        num_subplots = len(temperatures)
        species_list_smiles = [
            (r"H$_2$DABCO",  
            "[H][C]1([H])[C]([H])([H])[N]2([H])[C]([H])([H])[C]([H])([H])[N]1([H])[C]([H])([H])[C]2([H])[H]", 
            '#848484'),
            
            (r"HDABCO, deprotonated on N",
            "[H][C]1([H])[N]2[C]([H])([H])[C]([H])([H])[N]([H])([C]1([H])[H])[C]([H])([H])[C]2([H])[H]",
            adjust_lightness(base_color, gradient_factors[0])),
            
            (r"HDABCO, deprotonated on C",
            "[H][C]1[C]([H])([H])[N]2([H])[C]([H])([H])[C]([H])([H])[N]1([H])[C]([H])([H])[C]2([H])[H]",
            adjust_lightness(base_color, gradient_factors[1])),
            
            
            (r"H$_2$DABCO, bridge-opened-CN",
            "[H][C]([H])[C]([H])([H])[N]1([H])[C]([H])([H])[C]([H])([H])[N]([H])[C]([H])([H])[C]1([H])[H]",
            adjust_lightness(base_color, gradient_factors[2]))
        ]
        for temperature in sorted(temperatures):
            ax = plt.subplot(len(systems), 5, subplot_idx)
            subplot_idx += 1
            product_counts = {species_name: 0 for species_name, _, _ in species_list_smiles[1:]}
            for species_name, species_smiles, color in species_list_smiles[1:]:
                for species_file in species_files:
                    _current_temperature = species_file.split('/')[0]
                    _current_system = species_file.split('/')[1]
                    if _current_system == system and _current_temperature == temperature:
                        times, species = stat_species(species_file=species_file, to_stat=species_smiles)
                        product_counts[species_name] += integrate_species(times, species)  # 使用积分值

            for species_name, species_smiles, color in species_list_smiles:
                for species_file in species_files:
                    _current_temperature = species_file.split('/')[0]
                    _current_system = species_file.split('/')[1]
                    if _current_system == system and _current_temperature == temperature:
                        times, species = stat_species(species_file=species_file, to_stat=species_smiles)
                        times_ps = np.array(times) * 1e-3 * TIMESTEP
                        ax.plot(times_ps, species, label=species_name.replace("DABCO", "A"), c=color)
            
            if subplot_idx >= 17:
                ax.set_xlabel('Time (ps)')
            ax.text(0.95, 0.95, f'{system.upper()} at {temperature} K', ha='right', va='top', transform=ax.transAxes)
            ax.set_xscale('log')
            ax.set_xlim(1e-1, 1e2)
            ax.set_ylim(0, 250)
            if subplot_idx % 5 == 2:
                ax.set_ylabel('Count')
            else:
                ax.yaxis.set_ticks([])  
            inset_loc = [0.05, 0.1, 0.5, 0.5] if (subplot_idx % 5 >= 2 and subplot_idx % 5 <= 3) else [0.5, 0.1, 0.5, 0.5]
            ax_inset = ax.inset_axes(inset_loc)  # [x0, y0, width, height]
            sizes = list(product_counts.values())

            wedges, texts, autotexts = ax_inset.pie(
                sizes, 
                colors=[color for _, _, color in species_list_smiles[1:]], 
                autopct='%1.1f%%',
                pctdistance=1.2,
                startangle=90,
                textprops={
                    'color': 'black',
                    'ha': 'center', 
                    'va': 'center',
                    #'fontsize': 10 
                }
            )

            ax_inset.axis('equal')  
            plt.subplots_adjust(wspace=0)
        plt.tight_layout()
    plt.savefig(f'dabco_merged.eps')
    plt.savefig(f'dabco_merged.png', dpi=300)
    plt.close()
    print('Done.')
