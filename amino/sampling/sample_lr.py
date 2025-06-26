import torch

from amino.data.datasets import SidechainDataset
from amino.sampling.sampler import (
    sample_low_energy_hae,
    sample_low_energy_mapping,
    sample_low_energy_torsion,
)
from amino.utils.utils import (
    create_clean_path,
    sample_hypersphere,
    sample_torsion_angles,
    write_pdb,
)


def sample_lrs(amino, num_samples, lrs, latent_fn, sample_fn, path, mode):
    datapath = f"dataset/{mode}/data"
    dataset = SidechainDataset(amino, datapath)
    if "mapping" in path:
        latents = []
        for i in range(dataset.num_angles):
            latents.append(
                latent_fn(
                    n=num_samples,
                    dim=2,
                    dataset=dataset,
                )
            )
        latents = torch.stack(latents, dim=1)
    else:
        latents = latent_fn(
            n=num_samples,
            dim=dataset.num_angles,
            dataset=dataset,
        )
    scale = 1
    for lr in lrs:
        for i in range(num_samples):
            create_clean_path(f"{path}/{amino}/{lr}/{i}")
        sampled_struct, pred_energy = sample_fn(latents, amino, lr=lr, mode=mode)

        for i in range(num_samples):
            write_pdb(
                amino,
                dataset.atom_order,
                sampled_struct[-1][i],
                f"{path}/{amino}/{lr}/{i}/{0}_{pred_energy[-1][i,:]}.pdb",
                scale=scale,
            )
        for step, structs in enumerate(sampled_struct):
            for j, struct in enumerate(structs):
                write_pdb(
                    amino,
                    dataset.atom_order,
                    struct,
                    f"{path}/{amino}/{lr}/{j}/{step+1}_{pred_energy[step][j,:]}.pdb",
                    scale=scale,
                )


if __name__ == "__main__":
    # mode = "full"
    # path = f"pdbs/{mode}/HAE_samples_lr"
    # latent_fn = sample_hypersphere
    # sample_fn = sample_low_energy_hae
    # sample_lrs("ARG", 5, [10, 1, 0.1, 0.01], latent_fn, sample_fn, path, mode)

    # e
    aminos = ["ARG", "GLN", "GLU", "LYS", "MET"]
    aminos = ["ARG", "LYS", "MET"]

    num_samples = 5
    modes = ["full", "synth/high_energy"]
    lrs = [10, 1, 0.1, 0.01]
    for mode in modes:
        for amino in aminos:
            print(amino)
            path = f"pdbs/{mode}/HAE_samples_lr"
            latent_fn = sample_hypersphere
            sample_fn = sample_low_energy_hae
            sample_lrs(amino, num_samples, lrs, latent_fn, sample_fn, path, mode)

            path = f"pdbs/{mode}/torsion_samples_lr"
            latent_fn = sample_torsion_angles
            sample_fn = sample_low_energy_torsion
            sample_lrs(amino, num_samples, lrs, latent_fn, sample_fn, path, mode)

            path = f"pdbs/{mode}/mapping_samples_lr"
            latent_fn = sample_hypersphere
            sample_fn = sample_low_energy_mapping
            sample_lrs(amino, num_samples, lrs, latent_fn, sample_fn, path, mode)
