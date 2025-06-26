import os

import torch

from amino.data.datasets import SidechainDataset
from amino.utils.utils import generate_latents


def main():
    amino_acids = ["ARG", "LYS", "MET"]

    data_paths = ["full", "synth/high_energy"]
    for data_path in data_paths:
        for amino_acid in amino_acids:
            print(amino_acid)
            data_set = SidechainDataset(
                amino_acid, f"dataset/{data_path}/data", fixed_O=True
            )

            checkpoint_path = f"checkpoints/{data_path}/HAE/_0.1"
            latents = generate_latents(amino_acid, checkpoint_path, data_set)
            os.makedirs(f"dataset/{data_path}/HAE_latents", exist_ok=True)
            torch.save(latents, f"dataset/{data_path}/HAE_latents/{amino_acid}.pt")

            checkpoint_path = f"checkpoints/{data_path}/HAE/_0"
            os.makedirs(f"dataset/{data_path}/HAE_latents", exist_ok=True)
            latents = generate_latents(amino_acid, checkpoint_path, data_set)
            os.makedirs(f"dataset/{data_path}/HAE_latents_no_uni", exist_ok=True)
            torch.save(
                latents, f"dataset/{data_path}/HAE_latents_no_uni/{amino_acid}.pt"
            )


if __name__ == "__main__":
    main()
