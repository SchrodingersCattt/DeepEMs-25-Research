from deepmd.infer.deep_eval import (
    DeepEval,
)
import torch
import dpdata
import numpy as np
from tqdm import tqdm
import glob
import gc

torch.cuda.is_available = lambda: False 
SAVE_NAME = "SAVE_NAME_PLACEHOLDER"
SAVE_DIR = "descriptors/"
MODEL_PATH = "dpa1_l0/model.ckpt-4000000.pt"
# MODEL_PATH = "/mnt/data_nas/guomingyu/personal/dap-4/msst_roomtemp/Model/iter23_1.pb"


dd = "DATASET_PLACEHOLDER"
model = torch.load(MODEL_PATH,map_location=torch.device('cpu'))
dp = DeepEval(MODEL_PATH, device='cpu')
print(model['model']["_extra_state"]["model_params"]["type_map"])
type_map = model["model"]["_extra_state"]["model_params"]["type_map"] 
ls_multi = dpdata.LabeledSystem(dd, fmt='deepmd/npy')
all_descriptor = np.zeros((1, 1, 1208))  
for _, ls in enumerate(ls_multi[:10]):
    print(ls)
    old_type_map = ls.data["atom_names"].copy()
    assert isinstance(type_map, list)
    missing_type = [i for i in old_type_map if i not in type_map]
    assert not missing_type, f"Thes types are missing in model's type_map: {missing_type}!"
    _atype = np.array([type_map.index(old_type_map[ii]) for ii in ls.data["atom_types"]])
    assert list(_atype) == ls.data["atom_types"].tolist()
    # descriptor = dp.eval_descriptor(ls.data["coords"], ls.data["cells"], _atype)

    # model["model"]["Default"].set_eval_descriptor_hook(True)
    descriptor = dp.eval_descriptor(ls.data["coords"], ls.data["cells"], _atype)
    # model["model"]["Default"].set_eval_descriptor_hook(False)


    print(ls.data["coords"].shape, ls.data["cells"].shape, _atype.shape)
    all_descriptor = np.concatenate((all_descriptor.reshape(-1, all_descriptor.shape[-1]), descriptor.reshape(-1, descriptor.shape[-1])), axis=0)
    # print(descriptor.shape)
    print(all_descriptor.shape)

print(all_descriptor.shape)
np.save(f"{SAVE_DIR}/{SAVE_NAME}.npy", all_descriptor)

del dp
del ls_multi
del model
del descriptor
del all_descriptor
torch.cuda.empty_cache()
gc.collect()


print(glob.glob("../id-testset/*"))
