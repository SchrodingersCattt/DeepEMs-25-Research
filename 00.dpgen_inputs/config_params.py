import json
import glob
confs = [[x] for x in glob.glob('confs/*/POSCAR')]
inits = glob.glob('init*/*/')
configs = []

max_iter = 24
sys_idx_em = list(range(26))

for ii in range(max_iter):
    if ii < 1:
        model_devi_config = {
            "sys_idx": sys_idx_em, "trj_freq": 50, "ensemble": "npt-t",
            "nsteps": 5000, "temps": [100, 500], "press": [0, 100000, 500000, 1200000], "_idx":ii}
        configs.append(model_devi_config)
    elif ii < 2:
        model_devi_config = {
            "sys_idx": sys_idx_em, "trj_freq": 50, "ensemble": "npt-t",
            "nsteps": 10000, "temps": [100, 500], "press": [0, 100000, 500000, 1200000], "_idx":ii}
        configs.append(model_devi_config)
    
    elif ii < 4:
        model_devi_config = {
            "sys_idx": sys_idx_em, "trj_freq": 500, "ensemble": "nvt",
            "nsteps": 50000, "temps": [100, 300, 500, 900, 1500, 2500, 3000, 4000], "_idx":ii}
        configs.append(model_devi_config)
    
    elif ii < 6:
        model_devi_config = {
            "sys_idx": sys_idx_em, "trj_freq": 500, "ensemble": "nvt",
            "nsteps": 50000, "temps": [1500, 2000, 2500, 3000, 3500, 4000], "_idx":ii}
        configs.append(model_devi_config)
    
    elif ii < 8:
        model_devi_config = {
            "sys_idx": sys_idx_em, "trj_freq": 2000, "ensemble": "nvt",
            "nsteps": 100000, "temps": [1500, 2000, 2500, 3000, 3500, 4000], "_idx":ii}
        configs.append(model_devi_config)

    elif ii < 9:
        model_devi_config = {
            "sys_idx": sys_idx_em, "trj_freq": 50, "ensemble": "npt-i",
            "nsteps": 100000, "temps": [100, 500], "press": [0, 100000, 500000, 1200000], "_idx":ii}
        configs.append(model_devi_config)

    elif ii < 10:
        model_devi_config = {
            "sys_idx": sys_idx_em, "trj_freq": 50, "ensemble": "npt-a",
            "nsteps": 100000, "temps": [100, 500], "press": [0, 100000, 500000, 1200000], "_idx":ii}
        configs.append(model_devi_config)

    elif ii < 18:
        model_devi_config = {
            "sys_idx": sys_idx_em, "trj_freq": 200, "ensemble": "npt-t",
            "nsteps": 100000, "temps": [100, 500], "press": [0, 100000, 500000, 1200000], "_idx":ii}
        configs.append(model_devi_config)
        
    elif ii < 21:
        model_devi_config = {
            "sys_idx": sys_idx_em, "trj_freq": 1000, "ensemble": "nvt",
            "nsteps": 500000, "temps": [2000, 2500, 3000, 4000], "_idx":ii}
        configs.append(model_devi_config)

    elif ii < 23:
        model_devi_config = {
            "sys_idx": sys_idx_em, "trj_freq": 200, "ensemble": "npt-t",
            "nsteps": 100000, "temps": [100, 500], "press": [0, 100000, 500000, 1200000], "_idx":ii}
        configs.append(model_devi_config)


with open('params.json', 'r') as ff:
    data = json.load(ff)

data["model_devi_jobs"] = configs
data["init_data_sys"] = inits
data["sys_configs"] = confs

with open('params_2.json', 'w') as ff:
    json.dump(data, ff, indent=4)
