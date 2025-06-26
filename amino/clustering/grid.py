import argparse
import time
from collections import defaultdict

import numpy as np
import torch

from amino.clustering.kmeans import plot_angle_distributions
from amino.data.datasets import SidechainDataset
from amino.energy.struct_to_energy_multi import calculate_energy
from amino.visualize.energy_dist import plot_energy_dist


def sample_clusters(torsion_angles, n_samples_to_draw=500000, n_bins_per_dim=10):
    bin_edges = [
        np.linspace(-np.pi - 1e-7, np.pi + 1e-7, n_bins_per_dim + 1)
        for _ in range(torsion_angles.shape[1])
    ]

    bin_indices_per_dim = [
        np.digitize(torsion_angles[:, dim], bin_edges[dim], right=False) - 1
        for dim in range(torsion_angles.shape[1])
    ]
    bin_indices_per_dim = np.stack(
        bin_indices_per_dim, axis=1
    )  # shape (num_samples, num_dimensions)

    flat_bin_indices = np.ravel_multi_index(
        bin_indices_per_dim.T, dims=(n_bins_per_dim,) * torsion_angles.shape[1]
    )
    # Step 4: Count how many samples in each flat bin
    num_bins_total = n_bins_per_dim ** torsion_angles.shape[1]
    bin_counts = np.bincount(flat_bin_indices, minlength=num_bins_total)

    bin_probabilities = bin_counts / np.sum(bin_counts)
    bin_probabilities[bin_probabilities == 0] = np.nan
    inv_bin_probabilities = 1.0 / bin_probabilities

    inv_bin_probabilities = np.nan_to_num(inv_bin_probabilities)

    inv_bin_probabilities /= np.sum(inv_bin_probabilities)

    bin_to_samples = defaultdict(list)
    for sample_idx, bin_idx in enumerate(flat_bin_indices):
        bin_to_samples[bin_idx].append(sample_idx)

    sample_indices = []
    for i in range(n_samples_to_draw):
        while True:
            random_bin_idx = np.random.choice(num_bins_total, p=inv_bin_probabilities)
            samples_in_bin = bin_to_samples[random_bin_idx]
            if samples_in_bin:  # bin might be empty, but usually isn't
                sampled_sample_idx = np.random.choice(samples_in_bin)
                sample_indices.append(sampled_sample_idx)
                bin_to_samples[random_bin_idx].remove(sampled_sample_idx)
                break
            else:
                inv_bin_probabilities[random_bin_idx] = 0
                inv_bin_probabilities /= np.sum(inv_bin_probabilities)

    # Step 9: Get the sampled angles
    print(len(sample_indices))
    assert len(sample_indices) == len(set(sample_indices))
    return sample_indices


def grid_energy(amino, n_bins_per_dim, n_samples_to_draw):
    data_path = "dataset/full/data"
    data = SidechainDataset(amino, data_path, fixed_O=True)

    torsion_angles = data.torsion_angles
    start = time.time()
    print(f"Starting clustering")
    idxs = sample_clusters(torsion_angles, n_samples_to_draw, n_bins_per_dim)
    end = time.time()
    print(f"Done clustering in time: {end-start}")

    torch.save(idxs, f"dataset/full/energy/clusters/{amino}_idxs.pt")
    idxs = torch.load(f"dataset/full/energy/clusters/{amino}_idxs.pt")
    print(f"sampled: {len(set(idxs))}, {len(idxs)}")

    fig, axes = plot_angle_distributions(
        torsion_angles,
        torsion_angles[idxs],
        labels=["Full Real dataset", "Subsampled Real dataset"],
        ncols=2,
    )
    fig.suptitle(
        f"{amino} Torsion Angle Distribution Real Dataset", y=1.02, fontsize=14
    )
    fig.savefig(f"results/full/dataset/angle_dist_{amino}.pdf", bbox_inches="tight")

    energies = calculate_energy(data, idxs)
    tensor = torch.tensor(list(energies.values()), dtype=torch.float32).T
    all_energy = torch.ones((len(data), tensor.shape[1])) * torch.inf
    all_energy[idxs] = tensor
    key_to_col = {key: idx for idx, key in enumerate(energies.keys())}
    torch.save((key_to_col, all_energy), f"dataset/full/energy/{amino}.pt")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run grid_energy for given amino acids."
    )

    # Add arguments
    parser.add_argument(
        "--amino_idx", type=int, default=0, help="Index of amino to proces"
    )

    # Parse arguments
    args = parser.parse_args()

    # Extract arguments
    amino_idx = args.amino_idx
    n_bins_per_dim = 10
    n_samples_to_draw = 500000

    # Run the kmeans_energy function for each amino acid
    aminos = ["ARG", "LYS", "MET", "GLU", "GLN"]
    amino = aminos[amino_idx]
    print(amino)
    grid_energy(amino, n_bins_per_dim, n_samples_to_draw)

    data = SidechainDataset(amino, data_path=f"dataset/full/data", fixed_O=True)
    sin_vals = torch.sin(data.torsion_angles)
    cos_vals = torch.cos(data.torsion_angles)
    sin_3vals = torch.sin(3 * data.torsion_angles)
    cos_3vals = torch.cos(3 * data.torsion_angles)

    
    latents = torch.cat((sin_vals, cos_vals, sin_3vals, cos_3vals), dim=-1)
    torch.save(latents, f"dataset/full/torsion_latents_dim2/{amino}.pt")

    latents = torch.cat((sin_vals, cos_vals), dim=-1)
    torch.save(latents, f"dataset/full/torsion_latents_dim1/{amino}.pt")

    energy_path = "dataset/full/energy"
    key_to_col, energy = torch.load(f"{energy_path}/{amino}.pt")
    labels = [None]
    fig = plot_energy_dist(key_to_col, [energy], labels, iqr=True)
    fig.suptitle(f"{amino} Energy Distribution Real Dataset", fontsize=14)
    fig.savefig(
        f"results/full/dataset/energy_dist_{amino}.pdf", bbox_inches="tight"
    )