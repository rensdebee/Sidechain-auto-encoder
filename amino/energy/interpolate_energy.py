import os

import torch
import torch_kdtree

from amino.data.datasets import inf_filter_energy, iqr_filtering_energy

amino_acids = ["ARG", "LYS", "MET"]
forces = ["PeriodicTorsionForce", "LJForce", "CoulombForce"]
type = "full"
for amino_acid in amino_acids:
    print(amino_acid)
    key_to_col, energy_all_forces = torch.load(
        f"dataset/{type}/energy/{amino_acid}.pt", weights_only=True
    )
    latents = torch.load(
        f"dataset/{type}/HAE_latents/{amino_acid}.pt", weights_only=True
    )
    assert latents.shape[0] == energy_all_forces.shape[0]

    cols = []
    new_key_to_col = {}
    for i, force in enumerate(forces):
        cols.append(key_to_col[force])
        new_key_to_col[force] = i

    energy_all_forces = energy_all_forces[:, cols]

    energy_all_forces = iqr_filtering_energy(energy_all_forces)
    energy_mask = inf_filter_energy(energy_all_forces)

    interpolated_energy = energy_all_forces.clone()
    inf_mask = inf_filter_energy(energy_all_forces)
    non_inf_idxs = torch.nonzero(inf_mask)
    inf_idxs = torch.nonzero(~inf_mask)
    assert len(inf_idxs) + len(non_inf_idxs) == latents.shape[0]
    tree = torch_kdtree.build_kd_tree(latents[non_inf_idxs], device="cuda")

    distance_tensor, index_tensor = tree.query(latents[inf_idxs].cuda(), 5)
    distance_tensor = torch.where(
        distance_tensor == 0, torch.tensor(1e-8), distance_tensor
    ).cpu()
    index_tensor = index_tensor.cpu()
    print(index_tensor.shape)
    retrieved_energies = energy_all_forces[non_inf_idxs][index_tensor.int()].squeeze()
    assert torch.isinf(retrieved_energies).sum() == 0
    print(retrieved_energies.shape)
    normalizer = distance_tensor.sum(dim=1, keepdim=True)
    normalized_distance = distance_tensor / normalizer
    weights = (1 / normalized_distance) / (1 / normalized_distance).sum(
        dim=1, keepdim=True
    )
    weights = weights.unsqueeze(-1)
    print(weights.shape)
    predicted_energy = (retrieved_energies * weights).sum(dim=1)
    assert torch.isinf(predicted_energy).sum() == 0
    interpolated_energy[inf_idxs] = predicted_energy.unsqueeze(-2)

    print(interpolated_energy.shape)
    assert torch.isinf(interpolated_energy).sum() == 0
    os.makedirs(f"dataset/{type}/interpolated_energy", exist_ok=True)
    torch.save(
        (new_key_to_col, interpolated_energy),
        f"dataset/{type}/interpolated_energy/{amino_acid}.pt",
    )
