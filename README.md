# Efficient energy guided sampling of realistic amino acid side-chain conformations via latent space representations
**Master Thesis Rens den Braber**

**University of Amsterdam**

https://github.com/user-attachments/assets/fe12926d-32c7-4d39-aaf9-ee6963b60332

## Overview
This repository contains the code associated with the thesis Efficient energy guided sampling of realistic amino acid side-chain conformations via latent space representations.
```
📦 Sidechain-auto-encoder
├─ amino (Main code)
├─ checkpoints (Trainend model checkpoints)
├─ configs (Config files with network training hyperparameters)
├─ jobfiles (Jobfiles to be used when training on Snellius)
└─ results (stores results + code to generate figures/tabes)
```
Some naming conventions in the code are slightly different from the thesis:

Code: Thesis

full dataset: PDB dataset 

synth dataset: unfiltered synthetic dataset

synth/high_energy: filtered synthetic dataset

mapping network: hybrid model

## Instalation
First, clone this repository, then run the following commands.
A Snellius installation job file can be found [here](jobfiles/install_env.job)
1. **Create a new conda environment**:  
   ```sh
   conda create --yes --name sidechain python=3.10 numpy matplotlib 
   conda activate sidechain
   conda install wandb scikit-learn plotly openmm pandas seaborn --channel conda-forge 
   pip install typing_extensions
   pip install -e .
   ```

2. **Install PyTorch with CUDA support**:  
   ```sh
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install PyTorch Lightning**:  
   ```sh
   conda install lightning -c conda-forge
   ```

4. **Install Torch KD-Tree**: 
   ```sh
   git clone https://github.com/thomgrand/torch_kdtree
   cd torch_kdtree
   git submodule init
   git submodule update
   pip install .
   ```
   (Instructions for compiling extra dimensions can be found [here](https://github.com/thomgrand/torch_kdtree)
   
6. **Install PDB fixer**: 
   ```sh
   git clone https://github.com/openmm/pdbfixer.git
   cd pdbfixer/
   python setup.py install
   ```
## Dataset
Unfortunately, the side-chain dataset is not yet publicly available, but it will be shared later.
If you're interested, please feel free to reach out.

Generating the synthetic dataset can be done by running:
```sh
python amino/data/synth_data.py
```

Calculating the openMM energy for the synthetic and PDB datasets is done using these commands:
With amino_idx specifying the index of the amino acid in this list ["ARG", "LYS", "MET", "GLU", "GLN"] to be ran.
```sh
python amino/clustering/grid.py --amino_idx 0
python amino/energy/struct_to_energy_multi.py --amino_idx 0
```
On Snellius, these scripts can easily be run using the following commands: 
```sh
   sbatch --array=0-2 jobfiles/energy/grid_energy.job
   sbatch --array=0-2 jobfiles/energy/synth_energy.job
```
### Filtering the synthetic dataset
Filtering the synthetic dataset to exclude the lowest energy conformations can be done by running:
```sh
python amino/data/high_energy_synth_data.py
```
### Extract HAE latents & interpolate energy
After training the HAE encoder-decoder model (see next section), the latents extracted using these models are needed to train the second decoder. They can then also be used to extend the energy calculated dataset using interpolation.
For this run the following commands:
```sh
# Generate latents
python amino/data/generate_latents.py
# Interpolate energy
python amino/energy/interpolate_energy.py
# Evaluate interpolation
python amino/energy/test_interpolation.py
```
## Training the models
Training these models is best done on the Snellius supercomputer using an H100; the following commands can be used, and the job files referenced in these commands contain the Python script to run when not on Snellius. These commands will train the Arginine, Lysine, and Methionine models.
```sh
# HAE encoder-decoder:
sbatch --array=0-2 jobfiles/full/HAE.job
sbatch --array=0-2 jobfiles/synth/HAE.job

# HAE encoder-decoder no uniformity loss:
sbatch --array=0-2 jobfiles/full/HAE_no_uni.job
sbatch --array=0-2 jobfiles/synth/HAE_no_uni.job

# HAE second decoder (requires latents to be extracted, see previous section):
sbatch --array=0-2 jobfiles/full/HAE_decoder_energy.job
sbatch --array=0-2 jobfiles/synth/HAE_decoder_energy.job

# Torsion decoder:
sbatch --array=0-2 jobfiles/full/torsion_decoder.job
sbatch --array=0-2 jobfiles/synth/torsion_decoder.job

# Torsion energy predictor with periodicity:
sbatch --array=0-2 jobfiles/full/torsion_energy_predictor_dim2.job
sbatch --array=0-2 jobfiles/synth/torsion_energy_predictor_dim2.job

# Torsion energy predictor without periodicity:
sbatch --array=0-2 jobfiles/full/torsion_energy_predictor_dim1.job
sbatch --array=0-2 jobfiles/synth/torsion_energy_predictor_dim1.job

# Hybrid network:
sbatch --array=0-2 jobfiles/full/mapping/mapping.job
sbatch --array=0-2 jobfiles/synth/mapping/mapping.job
```
## Evaluating the trained models
The Hybrid and Torsion angle networks will already be evaluated during the training runs. The HAE networks, on the other hand, as they are also evaluated on random latents, require the following script to be run:
```sh
python amino/eval_model.py
```
## Sampling energy-minimized side-chain conformations:
The following script will generate samples and evaluate the sampling process:
```sh
# Samples 5 examples using each model and writes them to a PDB
python amino/sampling/sample_lr.py
# Samples 2000 structures and evaluates their energy using openMM
python amino/sampling/evaluate_energy.py
# Finds the lowest energy structures and writes them to PDB files
python amino/sampling/get_lowest_energy_struct.py
```
## Generate figures/tables
The code for generating the figures and tables used in the thesis can be found in the results folder.
