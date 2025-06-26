import torch

from amino.clustering.kmeans import plot_angle_distributions
from amino.data.datasets import SidechainDataset

aminos = ["MET", "ARG", "LYS"]
datasets = ["full", "synth", "synth/high_energy"]
for amino in aminos:
    angles = []
    print(amino)
    idxs = torch.load(
        f"dataset/full/energy/clusters/{amino}_idxs.pt", weights_only=False
    )
    for dataset in datasets:
        data = SidechainDataset(amino, data_path=f"dataset/{dataset}/data")
        print(data.torsion_angles.shape)
        angles.append(data.torsion_angles)
        if dataset == "full":
            angles.append(data.torsion_angles[idxs])
            print(data.torsion_angles[idxs].shape)
    fig, axes = plot_angle_distributions(
        *angles,
        labels=[
            "PDB dataset",
            "Subsampled PDB dataset",
            "Full Synthetic dataset",
            "Filtered Synthetic dataset",
        ],
        ncols=2,
    )
    fig.suptitle(f"{amino} Torsion Angle Distributions", fontsize=14)
    fig.savefig(f"figures/Torsion_dist_{amino}.pdf")
