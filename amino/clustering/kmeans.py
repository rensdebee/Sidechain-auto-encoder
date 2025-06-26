import argparse
import os
import random
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.cluster import MiniBatchKMeans

from amino.data.datasets import SidechainDataset
from amino.energy.struct_to_energy_multi import calculate_energy


def plot_angle_distributions(
    *arrays, labels=None, angle_names=None, figsize=None, ncols=3
):
    if not arrays:
        raise ValueError("At least one input array required")

    num_angles = arrays[0].shape[1]
    for arr in arrays:
        if arr.ndim != 2 or arr.shape[1] != num_angles:
            raise ValueError("All arrays must be 2D with same number of columns")

    nrows = int(np.ceil(num_angles / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (ncols * 5, nrows * 4))
    axes = axes.flatten()

    labels = labels or [f"Dataset {i}" for i in range(len(arrays))]
    angle_names = angle_names or [rf"$\chi_{i}$" for i in range(num_angles)]

    for idx, (ax, title) in enumerate(zip(axes, angle_names)):
        # Plot KDE for each dataset
        for dataset, label in zip(arrays, labels):
            data = dataset[:, idx]
            sns.kdeplot(data, ax=ax, label=label, linewidth=1.2, alpha=0.8)

        ax.set_title(title, pad=12)
        ax.set_xlabel("Torsion angle (Radians)", labelpad=5)
        ax.set_ylabel("Density", labelpad=5)
        ax.grid(True, linestyle="--", alpha=0.6)

        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])

    for j in range(num_angles, len(axes)):
        print(j)
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(arrays),
        frameon=True,
        fontsize=11,
        facecolor="white",
        edgecolor="0.8",
        title_fontsize=12,
        bbox_to_anchor=(0.5, 0.96),
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.93])
    return fig, axes


def plot_unique_distribution(*arrays, titles=None, figsize=(12, 8)):
    num_arrays = len(arrays)
    if titles is None:
        titles = [f"Distribution {i+1}" for i in range(num_arrays)]
    elif len(titles) != num_arrays:
        raise ValueError("Number of titles must match the number of arrays.")

    fig, axes = plt.subplots(nrows=1, ncols=num_arrays, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, (arr, title) in enumerate(zip(arrays, titles)):
        # Get unique values and their counts
        unique_values, counts = np.unique(arr, return_counts=True)

        # Plot the distribution
        axes[i].bar(
            unique_values, counts, color="steelblue", edgecolor="black", alpha=0.8
        )
        axes[i].set_xlabel("Unique Values", fontsize=12)
        axes[i].set_ylabel("Counts", fontsize=12)
        axes[i].set_title(title, fontsize=14, fontweight="bold")
        axes[i].grid(axis="y", linestyle="--", alpha=0.6)
        axes[i].tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    plt.savefig(f"figures/kmeans_dist.png")


def sample_clusters(cluster_list, P):
    # Step 1: Precompute cluster indices
    cluster_indices = defaultdict(list)
    for idx, cluster_idx in enumerate(cluster_list):
        cluster_indices[cluster_idx].append(idx)

    # Step 2: Identify clusters with fewer than P elements
    deficit_clusters = []
    excess_clusters = []
    total_deficit = 0
    for cluster_idx, indices in cluster_indices.items():
        size = len(indices)
        if size < P:
            deficit_clusters.append(cluster_idx)
            total_deficit += P - size
        elif size > P:
            excess_clusters.append(cluster_idx)

    # Step 3: Distribute the deficit proportionally
    weights = []
    additional_samples = defaultdict(int)
    if total_deficit > 0:
        # Calculate the total excess size
        total_excess = sum(
            len(cluster_indices[cluster_idx]) - P for cluster_idx in excess_clusters
        )
        # Distribute the deficit proportionally
        for cluster_idx in excess_clusters:
            excess_size = len(cluster_indices[cluster_idx]) - P
            weights.append(excess_size / total_excess)

        aditional_idxs = np.random.choice(
            excess_clusters, size=total_deficit, p=weights, replace=True
        )
        for idx in aditional_idxs:
            additional_samples[idx] += 1
    # Step 4: Perform the sampling
    sampled_elements = []
    left_over = 0
    for cluster_idx, indices in cluster_indices.items():
        size = len(indices)
        if size > P:
            total_len = len(indices)
            to_sample = P + additional_samples[cluster_idx] + left_over
            if to_sample < total_len:
                sampled_elements.extend(random.sample(indices, to_sample))
                left_over = 0
            else:
                # Sample all elements from this cluster
                sampled_elements.extend(indices)
                left_over = to_sample - total_len
        else:
            # Sample all elements from this cluster
            sampled_elements.extend(indices)

    return sampled_elements


def kmeans_energy(amino, n_clusters, samples_per_cluster):
    data_path = "dataset/full/data"
    data = SidechainDataset(amino, data_path, fixed_O=True)

    p = samples_per_cluster
    torsion = data.torsion_angles
    cores = os.cpu_count()
    print(f"Starting clustering using {cores} cores")
    start = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=256 * cores, n_init="auto"
    )
    clusters = kmeans.fit_predict(torsion)
    end = time.time()
    print(f"Done clustering in time: {end-start}")
    torch.save(clusters, f"dataset/full/energy/clusters/{amino}.pt")

    clusters = torch.load(
        f"dataset/full/energy/clusters/{amino}.pt", weights_only=False
    )
    unique_values, counts = np.unique(clusters, return_counts=True)
    num_clusters = len(unique_values)

    deficit = p * len(counts[counts < p]) - np.sum(counts[counts < p])
    rest = p * len(counts[counts >= p]) + np.sum(counts[counts < p])

    print(f"Num clusters: {num_clusters}")
    print(f"To sample: {p*num_clusters}, { rest + deficit}")

    idxs = sample_clusters(clusters, p)
    torch.save(idxs, f"dataset/full/energy/clusters/{amino}_idxs.pt")

    idxs = torch.load(f"dataset/full/energy/clusters/{amino}_idxs.pt")
    print(f"sampled: {len(set(idxs))}, {len(idxs)}")

    energies = calculate_energy(data, idxs)

    tensor = torch.tensor(list(energies.values()), dtype=torch.float32).T
    torch.save((idxs, tensor), f"dataset/full/energy/clusters/{amino}_tensor.pt")

    all_energy = torch.ones((len(data), tensor.shape[1])) * torch.inf
    all_energy[idxs] = tensor
    key_to_col = {key: idx for idx, key in enumerate(energies.keys())}

    torch.save((key_to_col, all_energy), f"dataset/full/energy/{amino}.pt")

    fig, axes = plot_angle_distributions(
        torsion[idxs], torsion, labels=["Sampled dataset", "Full dataset"], ncols=2
    )
    fig.suptitle(f"{amino} Angle Distribution", y=1.02, fontsize=14)
    fig.savefig(f"results/full/dataset/angle_dist_{amino}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run kmeans_energy for given amino acids."
    )

    parser.add_argument(
        "--amino_idx", type=int, default=0, help="Index of amino to proces"
    )

    args = parser.parse_args()

    amino_idx = args.amino_idx
    samples_per_cluster = 5
    n_clusters = 100000

    aminos = ["ARG", "LYS", "MET", "GLU", "GLN"]
    amino = aminos[amino_idx]
    print(amino)
    kmeans_energy(amino, n_clusters, samples_per_cluster)
