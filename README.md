# DPA-EMs
Energetic Materials Research based on Deep Potential.

Here’s a well-structured `README.md` for your repository, explaining the purpose of each directory and file:

---

# DPA-EMs: Energetic Materials Molecular Dynamics Simulations

This repository contains scripts, configurations, and data for molecular dynamics (MD) simulations of energetic materials, specifically focusing on the DAP (Diamine Perchlorate) series and other related compounds. The repository is organized into several directories, each serving a specific purpose. Below is a detailed description of the contents.

---

## Repository Structure

### **1. `01.dpgen_inputs`**
Contains configuration files and scripts for generating input files for machine learning potential (MLP) training and testing.
- `config_params.py`: Python script for defining MLP training parameters.
- `machine.json`: Configuration file specifying computational resources for MLP training.
- `params.json`: JSON file containing parameters for MLP training.

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
- `01.stat_species_mda.py`: Script for analyzing species using MDAnalysis.
- `02.calc_rate_dabcos.py`: Script for calculating reaction rates of DABCO molecules.
- `03.stat_collision.py`: Script for statistical analysis of molecular collisions.
- `04.monitor_model_devi.py`: Script for monitoring model deviations during MD simulations.
- `in_template.lmp`: Template input file for LAMMPS simulations.
- `run_react_gen.py`: Script for running reaction generation and analysis.

---

### **4. `confs`**
Contains configuration files (`POSCAR`) for various energetic materials, including the DAP series and other compounds. Each subdirectory corresponds to a specific material and its relaxed structure.
- **DAP Series**:
  - `DAP-1`, `DAP-2`, `DAP-3`, `DAP4-order`, `DAP-5`, `DAP-6`, `DAP-7_222-order`, `DAP-M4`: Configuration files for different DAP variants.
- **Other Energetic Materials**:
  - `AN_JACFOM01_333_relaxed`, `AP_SUXRUA_222_relaxed`, `Az2Cu_icsd_28171_152`, `Az2Hg_icsd_21029_222`, `Az2Pb_icsd_1298_211`, `Az2Pb_icsd_16887_211`, `AzAg_icsd_27135_222`, `AzCu_icsd_420051_221`, `AzTl_icsd_25009_222`, `CL-20_PUBMUU03_211`, `CL-20_PUBMUU20_211_relaxed`, `CL-20_PUBMUU26_211`, `HMX_OCHTET_112`, `HMX_OCHTET12_222`, `HMX_OCHTET13_222`, `RDX_CTMTNA04_121`, `TNT_ZZZMUC01_121`, `TNT_ZZZMUC06_112`: Configuration files for various energetic materials.

---

### **5. `LICENSE`**
Specifies the license under which this repository is distributed. By default, this repository uses the MIT License.

---

### **6. `README.md`**
This file, providing an overview of the repository and its contents.

---

## Getting Started

### Prerequisites
- Python 3.x
- LAMMPS
- MDAnalysis
- Deep Potential (DP) tools

### Installation
Clone the repository:
```bash
git clone https://github.com/SchrodingersCattt/DPA-EMs.git
cd DPA-EMs
```

### Usage
1. **MLP Training**: Use the scripts in `01.dpgen_inputs` to generate input files and train the MLP.
2. **MLP Testing**: Use the scripts in `02.dptest_scripts` to test the trained MLP.
3. **MD Simulations**: Use the `POSCAR` files in `confs` and the scripts in `03.md_scripts` to run and analyze MD simulations.

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear and concise messages.
4. Submit a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- This work was supported by [Your Funding Source].
- Special thanks to the contributors and maintainers of the Deep Potential and MDAnalysis libraries.

---

## Contact
For questions or feedback, please contact [Your Name] at [Your Email].

---
