# DPA-EMs
Energetic Materials Research based on Deep Potential.


This repository contains scripts, configurations, and data for molecular dynamics (MD) simulations of energetic materials, specifically focusing on the DAP (1,4-**d**iazabicyclo[2.2.2]octane-1,4-diium ($\mathrm{H_2dabco}$)-**a**lkali/**a**mmonium  **p**erchlorate) series and other related compounds. The repository is organized into several directories, each serving a specific purpose. Below is a detailed description of the contents.

Relevant citation: https://arxiv.org/pdf/2503.04540


Links to datasets: https://aissquare.com/datasets/detail?pageType=datasets&name=EnergeticMaterials-v1&id=311


Links to DeePMD models: https://aissquare.com/models/detail?pageType=models&name=DeepEMs-25__DPA1-L0&id=312


---

## Repository Structure

### **1. `01.dpgen_inputs`**
Contains configuration files and scripts for generating input files for machine learning potential (MLP) training and testing.
- `machine.json`: The contents of this file WOULD NOT BE PUBLISHED due to privacy consideration. Readers could refer to the repo of dpgen (https://github.com/deepmodeling/dpgen).
- `params.json`: JSON file containing parameters for DP-GEN running.
- `config_params.py`: Python script for quick configuring `params.json` MLP training parameters.

---

### **2. `02.dptest_scripts`**
Includes scripts and output files for testing the trained MLP.
- `plot_dptest.py`: Python script for plotting test results.
- `test.e.out`: Output file containing energy predictions.
- `test.e_peratom.out`: Output file containing per-atom energy predictions.
- `test.f.out`: Output file containing force predictions.
- `test.v.out`: Output file containing virial predictions.
- `test.v_peratom.out`: Output file containing per-atom virial predictions.

---

### **3. `03.md_scripts`**
Contains scripts for analyzing molecular dynamics simulations and statistical properties.
- `00.stat_species_smiles.py`: Script for tracking and analyzing chemical species using SMILES encoding.
- `01.stat_species_mda.py`: Script for analyzing species defined by coordinate numbers and distance cutoff using MDAnalysis.
- `02.calc_rate_dabcos.py`: Script for calculating reaction rates of DABCOs.
- `03.stat_collision.py`: Script for statistical analysis of molecular collisions.
- `04.monitor_model_devi.py`: Script for monitoring model deviations during MD simulations.
- `in_template.lmp`: Template input file for LAMMPS simulations.
- `run_react_gen.py`: Script for running reaction generation and analysis.

---

### **4. `04.descriptor_inference`**
Contains scripts for analyzing molecular dynamics simulations and statistical properties.
- `00.stat_species_smiles.py`: Script for tracking and analyzing chemical species using SMILES encoding.
- `01.stat_species_mda.py`: Script for analyzing species defined by coordinate numbers and distance cutoff using MDAnalysis.
- `02.calc_rate_dabcos.py`: Script for calculating reaction rates of DABCOs.
- `03.stat_collision.py`: Script for statistical analysis of molecular collisions.
- `04.monitor_model_devi.py`: Script for monitoring model deviations during MD simulations.
- `in_template.lmp`: Template input file for LAMMPS simulations.
- `run_react_gen.py`: Script for running reaction generation and analysis.

---

### **5. `confs`**
Contains configuration files (`POSCAR`) for various energetic materials, including the DAP series and other compounds. Each subdirectory corresponds to a specific material and its relaxed structure.
- **DAP Series**:
  - `DAP-1`, `DAP-2`, `DAP-3`, `DAP4-order`, `DAP-5`, `DAP-6`, `DAP-7_222-order`, `DAP-M4`: Configuration files for different DAP variants.
- **Other Energetic Materials**:
  - `AN_JACFOM01_333_relaxed`, `AP_SUXRUA_222_relaxed`, `Az2Cu_icsd_28171_152`, `Az2Hg_icsd_21029_222`, `Az2Pb_icsd_1298_211`, `Az2Pb_icsd_16887_211`, `AzAg_icsd_27135_222`, `AzCu_icsd_420051_221`, `AzTl_icsd_25009_222`, `CL-20_PUBMUU03_211`, `CL-20_PUBMUU20_211_relaxed`, `CL-20_PUBMUU26_211`, `HMX_OCHTET_112`, `HMX_OCHTET12_222`, `HMX_OCHTET13_222`, `RDX_CTMTNA04_121`, `TNT_ZZZMUC01_121`, `TNT_ZZZMUC06_112`: Configuration files for various energetic materials.

---
