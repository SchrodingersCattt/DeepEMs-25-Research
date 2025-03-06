import os
import glob
import logging
from concurrent.futures import ProcessPoolExecutor
from subprocess import call
import traceback

logging.basicConfig(level=logging.INFO, filename='run_react_gen.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ALL_TYPE_MAP_str = " ".join([
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
])

def run_command(ww):
    cwd = os.getcwd()
    try:
        logging.info(f"Running in {ww}")
        os.chdir(ww)
        return_code = call(cmd, shell=True)
        if return_code != 0:
            logging.error(f"Command {cmd} failed in {ww} with return code {return_code}")
            logging.error(traceback.format_exc())
        os.chdir(cwd)
        logging.info(f"Done in {ww}")
    except Exception as e:
        logging.error(f"Error in {ww}: {e}")
        logging.error(traceback.format_exc())

def par_run_command(wk_places):
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_command, ww) for ww in wk_places]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Unexpected error: {e}")

if __name__ == '__main__':    
    dump_filename = "dump_nvt_freq-100.lammpstrj"
    # 2, 10 or 20 are both ok. Smaller steps means more resolution but more noise, and vice versa.    
    stepinterval = 2 
    maxspecies = 50
    sel_atoms = "C O N"
    systems = ['dap-1', 'dap-2', 'dap-3', 'dap-4']
    for system in systems:
        wk_places = glob.glob(f"*/{system}/")
        wk_places = [x for x in wk_places if ('imgs' not in x) and ('_' not in x) and (int(x.split('/')[0])<2800)]
        wk_places = sorted(wk_places, key=lambda x: int(x.split('/')[0]))
        cmd = f"reacnetgenerator -i {dump_filename} --type dump -a {ALL_TYPE_MAP_str} -n 16 --nohmm --stepinterval {stepinterval} --maxspecies {maxspecies}"
        par_run_command(wk_places)