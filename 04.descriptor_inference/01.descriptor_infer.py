import os
import torch
import dpdata
import numpy as np
from tqdm import tqdm
import glob
import gc


DATA_PATHs = glob.glob("../id-testset/*")
for dd in DATA_PATHs:
    SAVE_NAME = dd.split('test_')[-1].split('_')[0]
    SAVE_DIR = "descriptors/"

    with open("_descriptor_infer.py", "r") as f:
        tmp_code = f.read()
    tmp_code = tmp_code.replace("SAVE_NAME_PLACEHOLDER", SAVE_NAME)
    tmp_code = tmp_code.replace("DATASET_PLACEHOLDER", dd)
    with open("_tmp_descriptor_infer.py", "w") as f:
        f.write(tmp_code)

    os.system(f"python _tmp_descriptor_infer.py")

