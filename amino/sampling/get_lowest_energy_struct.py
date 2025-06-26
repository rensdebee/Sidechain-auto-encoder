import torch

from amino.data.datasets import (
    LatentEvalDataset,
    SidechainDataset,
    iqr_filtering_energy,
    normalizing_energy,
)
from amino.utils.utils import create_clean_path, write_pdb

top_k = 5
aminos = ["ARG", "LYS", "MET"]
forces = ["PeriodicTorsionForce", "LJForce", "CoulombForce"]
modes = ["full", "synth/high_energy"]
lrs = [10, 1]
eval_path = "eval/"
models = ["HAE", "torsion", "mapping"]
for amino in aminos:
    for mode in modes:
        print(amino, mode)
        dataset = LatentEvalDataset(
            amino,
            f"dataset/{mode.split('/')[0]}/data",
            f"dataset/{mode.split('/')[0]}/torsion_latents_dim1",
            f"dataset/{mode.split('/')[0]}/energy",
            force_types=forces,
            iqr_filter_energy=True,
            normalize_energy=True,
            fixed_O=True,
            inf_filter=True,
        )
        norm_energy = dataset.energy
        original_energy = dataset.energy * dataset.std + dataset.mean
        val, idxs = torch.topk(
            original_energy.sum(dim=1), k=5, largest=False, sorted=True
        )
        store_path = f"pdbs/low_energy/{amino}/{mode}"
        # create_clean_path(store_path)
        for i, idx in enumerate(idxs):
            struct = dataset.sidechain_positions[idx]
            struct_energy = original_energy[idx]
            f_name = f"data_{mode.split('/')[0]}_"
            # for energy, norm_energy in zip(struct_energy, dataset.energy[idx]):
            #     f_name += f"{energy:.2f}_^{norm_energy:.2f}^_"
            f_name += f"{i}_{struct_energy.sum():.2f}_^{norm_energy[idx].sum():.2f}^"
            write_pdb(amino, dataset.atom_order, struct, f"{store_path}/{f_name}.pdb")

        data_key_to_col, data_energy = torch.load(
            f"dataset/{mode}/energy/{amino}.pt", weights_only=True
        )
        data_energy = iqr_filtering_energy(data_energy)
        cols = []
        for i, name in enumerate(forces):
            col = data_key_to_col[name]
            cols.append(col)
        data_energy = data_energy[:, cols]
        _, data_mean, data_std = normalizing_energy(data_energy)
        # For each LR
        for model in models:
            path = f"{eval_path}/{mode}/{model}/{amino}"
            for lr in lrs:
                try:
                    new_key_to_col, all_energy, pred_energy, clash_metric = torch.load(
                        f"{path}/{lr}/energy_results.pt"
                    )
                    samples = torch.load(f"{path}/{lr}/{amino}.pt")
                    norm_energy = (all_energy - data_mean) / data_std
                    original_energy = all_energy
                    val, idxs = torch.topk(
                        original_energy.sum(dim=1),
                        k=min(top_k, all_energy.shape[0]),
                        largest=False,
                        sorted=True,
                    )
                    for i, idx in enumerate(idxs):
                        struct = samples[idx]
                        f_name = f"sample_{mode.split('/')[0]}_model_{model}_lr_{lr}_"
                        # for energy, norm_energy in zip(
                        #     all_energy[idx], norm_all_energy[idx]
                        # ):
                        #     f_name += f"{energy:.2f}_^{norm_energy:.2f}^_"
                        f_name += f"{i}_{all_energy[idx].sum():.2f}_^{pred_energy[idx].sum():.2f}^"
                        write_pdb(
                            amino,
                            dataset.atom_order,
                            struct,
                            f"{store_path}/{f_name}.pdb",
                        )
                except FileNotFoundError as e:
                    print("File not found", e)
