import os

import pandas as pd
import torch
import torch_kdtree
from sklearn.model_selection import KFold
from tabulate import tabulate

from amino.data.datasets import inf_filter_energy, iqr_filtering_energy

amino_acids = ["ARG", "LYS", "MET"]
forces_to_test = ["PeriodicTorsionForce", "LJForce", "CoulombForce"]
results = []  # Store results in a list for tabulation
og_dist = []
data_domain = "full"
for amino_acid in amino_acids:
    energy_path = f"dataset/{data_domain}/energy/{amino_acid}.pt"
    latent_path = f"dataset/{data_domain}/HAE_latents/{amino_acid}.pt"

    key_to_col, energy_all_forces = torch.load(energy_path, weights_only=True)
    cols = []
    for force in forces_to_test:
        cols.append(key_to_col[force])
    energy_all_forces = energy_all_forces[:, cols]
    energy_all_forces = iqr_filtering_energy(energy_all_forces)

    latents = torch.load(latent_path, weights_only=True)
    energy_mask = inf_filter_energy(energy_all_forces)

    latents = latents[energy_mask]
    energy_all_forces = energy_all_forces[energy_mask]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    mae = []
    std = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(latents, energy_all_forces)):
        train_energy = energy_all_forces[train_idx]
        train_latents = latents[train_idx]
        test_energy = energy_all_forces[test_idx]
        test_latents = latents[test_idx]

        tree = torch_kdtree.build_kd_tree(train_latents, device="cuda")

        distance_tensor, index_tensor = tree.query(test_latents, 5)
        distance_tensor = torch.where(
            distance_tensor == 0, torch.tensor(1e-8), distance_tensor
        ).cpu()
        index_tensor = index_tensor.cpu()
        retrieved_energies = train_energy[index_tensor.int()].squeeze()
        assert torch.isinf(retrieved_energies).sum() == 0
        normalizer = distance_tensor.sum(dim=1, keepdim=True)
        normalized_distance = distance_tensor / normalizer
        weights = (1 / normalized_distance) / (1 / normalized_distance).sum(
            dim=1, keepdim=True
        )
        weights = weights.unsqueeze(-1)
        predicted_energy = (retrieved_energies * weights).sum(dim=1)
        # predicted_energy = retrieved_energies.mean(dim=1)
        diff = torch.abs(predicted_energy - test_energy)

        mae.append(diff.mean(dim=0))
        std.append(diff.std(dim=0))

    mae = torch.stack(mae)
    print(mae.shape)
    std = torch.stack(std)
    type = amino_acid
    for i, force in enumerate(forces_to_test):
        if i > 0:
            type = ""
        og_dist.append(
            [
                type,
                force,
                f"{energy_all_forces[:, i].mean(): .2f} ± {energy_all_forces[:, i].std():.2f}",
            ]
        )
        result = [
            amino_acid,
            force,
            f"{mae.mean(dim=0)[i]:.2f} ± {mae.std(dim=0)[i]:.2f}",
            f"{std.mean(dim=0)[i]:.2f} ± {std.std(dim=0)[i]:.2f}",
        ]
        results.append(result)

headers = ["Residue", "Force Type", "MAE", "STD"]
df = pd.DataFrame(results, columns=headers)
df_value1 = df.copy()
df_value1["Metric"] = "MAE"
df_value1 = df_value1.pivot(
    index=["Force Type", "Metric"], columns="Residue", values="MAE"
)

df_value2 = df.copy()
df_value2["Metric"] = "STD"
df_value2 = df_value2.pivot(
    index=["Force Type", "Metric"], columns="Residue", values="STD"
)

# Combine both tables
pivot_df = pd.concat([df_value1, df_value2]).sort_index().reset_index()
pivot_df["Force Type"] = pivot_df["Force Type"].mask(
    pivot_df["Force Type"].duplicated(), "\u00a0"
)

# Print in a readable format
os.makedirs(f"results/{data_domain}/dataset/", exist_ok=True)
with open(
    f"results/{data_domain}/dataset/interpolation.txt", "w", encoding="utf-8"
) as f:
    print(tabulate(pivot_df, headers="keys", tablefmt="grid", showindex=False), file=f)
    print(
        tabulate(
            og_dist, headers=["Force type", "Residue", "Mean ± std"], tablefmt="grid"
        ),
        file=f,
    )
