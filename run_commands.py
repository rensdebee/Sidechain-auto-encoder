import argparse
import itertools
import os
import subprocess

# Usage: python run_commands.py --config_file search.yaml --cli_options lr=0.001*0.0005*0.0001 batch_size=4096*512*1024 hidden_dims=[1024,256,128]*[256,128,32]*[256,32]


def run_commands(base_command, config_file, cli_options):
    # Extract CLI argument names and their values
    cli_keys = cli_options.keys()
    cli_values = cli_options.values()

    # Generate all possible combinations of CLI arguments
    combinations = list(itertools.product(*cli_values))

    for combination in combinations:
        # Construct the command with the current combination of arguments
        args = " ".join(f"--{key} {value}" for key, value in zip(cli_keys, combination))
        command = f"{base_command} {args}"
        if config_file is not None:
            command = f"{base_command} --config_file {config_file} {args}"
        print(f"Running command: {command}")
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")


if __name__ == "__main__":
    base_commands = [
        # "python amino/train_energy_predictor.py --config configs/full/torsion_energy_predictor_dim1.yaml",
        # "python amino/train_energy_predictor.py --config configs/synth/torsion_energy_predictor_dim2.yaml",
        # "python amino/train_energy_predictor.py --config configs/full/torsion_energy_predictor_dim2.yaml",
        # "python amino/train_energy_predictor.py --config configs/synth/torsion_energy_predictor_dim1.yaml",
        # "python amino/train_decoder.py --config configs/full/HAE_decoder_energy.yaml",
        # "python amino/train_decoder.py --config configs/synth/HAE_decoder_energy.yaml",
        # "python amino/train_decoder.py --config configs/full/HAE_decoder.yaml",
        # "python amino/train_decoder.py --config configs/synth/HAE_decoder.yaml",
        # "python amino/train_hae_energy_predictor.py --config configs/full/HAE_energy_predictor.yaml",
        # "python amino/train_hae_energy_predictor.py --config configs/synth/HAE_energy_predictor.yaml",
        "python amino/train_mapping_network.py --config configs/backbone.yaml",
    ]
    # Parse CLI options into a dictionary
    #  ["ARG", "GLN", "GLU", "LYS", "MET"],
    options = {
        "amino_acid": ["ARG", "LYS", "MET"],
        "amino_acid": ["GLY"],
    }

    for base_command in base_commands:
        # run_commands(base_command, "./configs/search.yaml", options)
        run_commands(base_command, None, options)
