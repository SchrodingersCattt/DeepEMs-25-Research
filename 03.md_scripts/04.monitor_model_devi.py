import glob
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


COLOR_DICT = {
    'dap-1': "#D9B426",
    "dap-2": "#957Bf9",
    "dap-3": "#f66F99",
    "dap-4": "#6DBA6A"
}

def adjust_lightness(color_hex, factor=0.5):
    rgb = plt.cm.colors.to_rgb(color_hex)
    h, s, v = rgb_to_hsv(rgb)
    new_v = min(1.0, v * factor)
    return plt.cm.colors.to_hex(hsv_to_rgb((h, s, new_v)))


_files = glob.glob("*/dap-*/*model_devi.out")
files = sorted(
    [
        x for x in _files if (
        int(x.split('/')[0])<=2500)
    ],
    key=lambda x: int(x.split('/')[0])
)

plt.figure(figsize=(10, 5))
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'

for file in files:
    temp = int(file.split('/')[0])
    system = file.split('/')[1]
    model_devi = np.loadtxt(file)
    
    ## first 10 are npt eq steps.
    step = model_devi[:, 0][10:] # monitor nvt only
    time = step/ 1e4 # 1e3 fs / 0.1 fs 
    max_devi_f = model_devi[:, 4][10:] # monitor nvt only
    avg_devi_f = model_devi[:, 6][10:] # monitor nvt only
    
    plt.subplot(211)
    plt.plot(time, max_devi_f, color=adjust_lightness(COLOR_DICT[system], temp/2000))
    plt.ylabel('Max. Devi. $F$ (rel.)')
    plt.text(0.02, 1.05, 'a', transform=plt.gca().transAxes,
            fontsize=16, fontweight='bold', va='bottom', ha='left')
    plt.xlim(0, 100)

    plt.subplot(212)
    plt.plot(time, avg_devi_f, color=adjust_lightness(COLOR_DICT[system], temp/2000), label=f"{system.upper()} at {temp} K")
    plt.xlabel('Time (ps)')
    plt.ylabel('Avg. Devi. $F$ (rel.)')
    plt.text(0.02, 1.05, 'b', transform=plt.gca().transAxes,
            fontsize=16, fontweight='bold', va='bottom', ha='left')
    plt.xlim(0, 100)

plt.legend(frameon=False, ncols=5, fontsize=8)
plt.tight_layout()
plt.savefig('model_devi.eps', dpi=300)