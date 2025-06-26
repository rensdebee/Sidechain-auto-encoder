import os

import torch
from matplotlib import pyplot as plt

from amino.data.clean_dataset import clean_structures
from amino.data.datasets import (
    SidechainDataset,
    iqr_filtering_energy,
    normalizing_energy,
)
from amino.energy.struct_to_energy_multi import calculate_energy
from amino.sampling.sampler import (
    sample_low_energy_hae,
    sample_low_energy_mapping,
    sample_low_energy_torsion,
)
from amino.utils.utils import (
    create_clean_path,
    sample_hypersphere,
    sample_torsion_angles,
)
from amino.visualize.energy_dist import (
    plot_energy_dist,
    plot_energy_TF,
    plot_error_hist,
)


def evaluate_samples(
    amino,
    samples,
    pred_energies,
    forces,
    lrs,
    clash_metrics,
    path,
    type="HAE",
    mode="full",
):
    path = f"{path}/{mode}/{type}/{amino}"
    # For each LR
    for sample, pred_energy, lr, clash_metric in zip(
        samples, pred_energies, lrs, clash_metrics
    ):
        print(f"Learning rate: {lr}")
        try:
            create_clean_path(f"{path}/{lr}")
            torch.save(sample, f"{path}/{lr}/{amino}.pt")
            dataset = SidechainDataset(amino, f"{path}/{lr}", fixed_O=False)
            idxs = list(range(len(dataset)))
            energies = calculate_energy(
                dataset, idxs, max_workers=None, minimize_steps=0
            )
            tensor = torch.tensor(list(energies.values()), dtype=torch.float32).T
            key_to_col = {key: idx for idx, key in enumerate(energies.keys())}
            all_energy = torch.ones((len(dataset), tensor.shape[1])) * torch.inf
            all_energy[idxs] = tensor
            cols = []

            new_key_to_col = {}
            cols = []
            for i, name in enumerate(forces):
                col = key_to_col[name]
                new_key_to_col[name] = i
                cols.append(col)
            all_energy = all_energy[:, cols]

            torch.save(
                (new_key_to_col, all_energy, pred_energy, clash_metric),
                f"{path}/{lr}/energy_results.pt",
            )
        except IndexError:
            print("zero valid structs")


def evaluate_model(amino, latent_fn, sample_fn, lrs, path, forces, type, mode):
    datapath = f"dataset/{mode}/data"
    dataset = SidechainDataset(amino, datapath)

    # evaluate
    samples = []
    pred_energy = []
    clash_metrics = []
    num_samples = 2000
    if "mapping" in type:
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
    for lr in lrs:
        sampled_struct, sample_energy = sample_fn(latents, amino, lr=lr, mode=mode)
        discard_idxs, clash_metric = clean_structures(
            structures=sampled_struct[-1],
            amino_acid=amino,
            atom_order=dataset.atom_order,
            return_metrics=True,
        )
        clash_metrics.append(clash_metric)
        mask = torch.ones(sampled_struct[-1].shape[0], dtype=torch.bool)
        mask[discard_idxs] = False
        samples.append(sampled_struct[-1][mask])
        pred_energy.append(sample_energy[-1][mask])
    print(clash_metrics)
    evaluate_samples(
        amino, samples, pred_energy, forces, lrs, clash_metrics, path, type, mode
    )


def get_metrics(
    amino, lrs, forces, trainings_energy_path, path, type="HAE", mode="full"
):
    real_energy = []
    pred_energies = []
    path = f"{path}/{mode}/{type}/{amino}"
    data_key_to_col, data_energy = torch.load(
        f"{trainings_energy_path}/{amino}.pt", weights_only=True
    )
    data_energy = iqr_filtering_energy(data_energy)
    cols = []
    for i, name in enumerate(forces):
        col = data_key_to_col[name]
        cols.append(col)
    data_energy = data_energy[:, cols]
    _, data_mean, data_std = normalizing_energy(data_energy)
    # For each LR
    for lr in lrs:
        try:
            new_key_to_col, all_energy, pred_energy, clash_metric = torch.load(
                f"{path}/{lr}/energy_results.pt"
            )
            norm_all_energy = (all_energy - data_mean) / data_std
            pred_energies.append(pred_energy)
            (
                num_discarded,
                total_generated,
                num_wrong_bonds,
                num_clashes_bonded,
                num_clashes_unbonded,
            ) = clash_metric
            os.makedirs(f"results/{mode}/{type}", exist_ok=True)
            with open(
                f"results/{mode}/{type}/sampling_metrics_{amino}.txt",
                "a",
                encoding="utf-8",
            ) as f:
                with open(
                    f"results/{mode}/{type}/sampling_dist_{amino}.txt",
                    "a",
                    encoding="utf-8",
                ) as f2:
                    print(f"lr {lr}", file=f)
                    print(f"lr {lr}", file=f2)
                    print(
                        f"num discarded: {num_discarded} [{(num_discarded/total_generated):.3f}]",
                        file=f,
                    )
                    print(
                        f"num wrong bonds: {num_wrong_bonds} [{(num_wrong_bonds/total_generated):.3f}]",
                        file=f,
                    )
                    print(
                        f"num clashes bonded: {num_clashes_bonded} [{(num_clashes_bonded/total_generated):.3f}]",
                        file=f,
                    )
                    print(
                        f"num clashes unbonded: {num_clashes_unbonded} [{(num_clashes_unbonded/total_generated):.3f}]",
                        file=f,
                    )
                    for i, name in enumerate(forces):
                        print(name, file=f)
                        print(name, file=f2)
                        col = new_key_to_col[name]
                        calc = all_energy[:, col]
                        norm_calc = (calc - data_mean[i]) / data_std[i]
                        pred = pred_energy[:, i]
                        diff = torch.abs(pred - calc)
                        print(
                            f"MAE: {torch.mean(diff):.3f} ± {torch.std(diff):.3f}",
                            file=f,
                        )
                        print(
                            f"mean: {torch.mean(calc):.3f} ± {torch.std(calc):.3f}",
                            file=f2,
                        )
                        print(
                            f"norm_mean: {torch.mean(norm_calc):.3f} ± {torch.std(norm_calc):.3f}",
                            file=f2,
                        )
                    print(
                        f"Overall summed norm mean: {torch.mean(norm_all_energy.sum(dim=1)):.3f} ± {torch.std(norm_all_energy.sum(dim=1)):.3f}",
                        file=f2,
                    )
                    print(
                        f"Overall summed mean: {torch.mean(all_energy.sum(dim=1)):.3f} ± {torch.std(all_energy.sum(dim=1)):.3f}",
                        file=f2,
                    )
            real_energy.append(all_energy)
        except FileNotFoundError:
            print("No file")

    fig = plot_error_hist(new_key_to_col, real_energy, pred_energies, lrs)
    fig.suptitle(f"{amino} Absolute Error Distribution", fontsize=14)
    fig.savefig(f"results/{mode}/{type}/error_dist_{amino}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    aminos = ["ARG", "GLN", "GLU", "LYS", "MET"]
    aminos = ["MET", "LYS", "ARG"]
    forces = ["PeriodicTorsionForce", "LJForce", "CoulombForce"]
    modes = ["full", "synth/high_energy"]

    lrs = [10, 1, 0.1, 0.001]

    calc_energy = True
    path = "eval/"

    for mode in modes:
        trainings_energy_path = f"dataset/{mode}/energy"

        for amino in aminos:
            print(amino)

            # HAE model
            if calc_energy:
                latent_fn = sample_hypersphere
                sample_fn = sample_low_energy_hae
                evaluate_model(
                    amino,
                    latent_fn,
                    sample_fn,
                    lrs,
                    path,
                    forces,
                    type="HAE",
                    mode=mode,
                )

                get_metrics(
                    amino,
                    lrs,
                    forces,
                    trainings_energy_path,
                    path,
                    type="HAE",
                    mode=mode,
                )

            # Torsion model
            if calc_energy:
                latent_fn = sample_torsion_angles
                sample_fn = sample_low_energy_torsion
                evaluate_model(
                    amino,
                    latent_fn,
                    sample_fn,
                    lrs,
                    path,
                    forces,
                    type="torsion",
                    mode=mode,
                )

            get_metrics(
                amino,
                lrs,
                forces,
                trainings_energy_path,
                path,
                type="torsion",
                mode=mode,
            )

            # Mapping model
            if calc_energy:
                latent_fn = sample_hypersphere
                sample_fn = sample_low_energy_mapping
                evaluate_model(
                    amino,
                    latent_fn,
                    sample_fn,
                    lrs,
                    path,
                    forces,
                    type="mapping",
                    mode=mode,
                )

            get_metrics(
                amino,
                lrs,
                forces,
                trainings_energy_path,
                path,
                type="mapping",
                mode=mode,
            )
