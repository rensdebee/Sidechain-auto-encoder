import os

import torch
from numpy import amin

from amino.data.datasets import LatentEvalDataset
from amino.visualize.energy_dist import plot_energy_dist

aminos = ["ARG", "LYS", "MET"]
forces = ["PeriodicTorsionForce", "LJForce", "CoulombForce"]
latent_types = ["torsion_latents_dim1", "torsion_latents_dim2"]
pct = 0.05
for amino in aminos:
    pdb = LatentEvalDataset(
        amino,
        f"dataset/full/data",
        f"dataset/full/torsion_latents_dim1",
        f"dataset/full/energy",
        force_types=forces,
        iqr_filter_energy=True,
        normalize_energy=False,
        fixed_O=True,
        inf_filter=True,
        interpolate_energy=False,
    )
    synth = LatentEvalDataset(
        amino,
        f"dataset/synth/data",
        f"dataset/synth/torsion_latents_dim1",
        f"dataset/synth/energy",
        force_types=forces,
        iqr_filter_energy=True,
        normalize_energy=False,
        fixed_O=True,
        inf_filter=False,
        interpolate_energy=False,
    )
    synth_energy = synth.energy.clone()
    pdb_energy = pdb.energy.sum(dim=1).sort().values
    cut_off_idx = int(pdb_energy.shape[0] * pct)
    cut_off_energy = pdb_energy[cut_off_idx]
    keep_mask = synth.energy.sum(dim=1) > cut_off_energy
    num_deleted = len(keep_mask) - keep_mask.sum()
    key_to_col = {}
    for i, force in enumerate(forces):
        key_to_col[force] = i
    energy = synth.energy.clone()[keep_mask]
    structs = synth.sidechain_positions[keep_mask]
    os.makedirs("dataset/synth/high_energy/energy", exist_ok=True)
    os.makedirs("dataset/synth/high_energy/data", exist_ok=True)
    torch.save((key_to_col, energy), f"dataset/synth/high_energy/energy/{amino}.pt")
    torch.save(structs, f"dataset/synth/high_energy/data/{amino}.pt")
    for latent_type in latent_types:
        latent_data = LatentEvalDataset(
            amino,
            f"dataset/synth/data",
            f"dataset/synth/{latent_type}",
            f"dataset/synth/energy",
            force_types=forces,
            iqr_filter_energy=True,
            normalize_energy=False,
            fixed_O=True,
            inf_filter=False,
            interpolate_energy=False,
        ).sidechain_latents[keep_mask]
        os.makedirs(f"dataset/synth/high_energy/{latent_type}/", exist_ok=True)
        torch.save(latent_data, f"dataset/synth/high_energy/{latent_type}/{amino}.pt")

    print(amino)
    print("PDB data:", pdb.energy.shape)
    print("Full synth data:", synth.energy.shape)
    print("Filtered synth data:", synth.energy[keep_mask, :].shape)
    fig = plot_energy_dist(
        key_to_col,
        [pdb.energy, synth.energy, synth.energy[keep_mask, :]],
        ["PDB dataset", "Fill Synthetic dataset", "Filtered Synthetic dataset"],
        iqr=False,
        plot_sum=True,
        kde=False,
    )
    fig.suptitle(f"{amino} Energy Distributions", fontsize=14)
    fig.savefig(f"figures/Combined_energy_dist_{amino}.pdf")
