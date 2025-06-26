import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from amino.data.datasets import LatentDataset, LatentEvalDataset
from amino.models.utils import calculate_3D_squared_distance
from amino.utils.utils import get_model


def run_dataset(dataset, model):
    sd = []
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8192 * 8,
        shuffle=False,
        num_workers=8,
        # persistent_workers=True,
        pin_memory=False if sys.platform.startswith("win") else True,
    )
    if hasattr(model, "decoder"):
        decoder = model.decoder
        energy = model.energy
        energy_dim = model.energy_dim
        ed = [[] for _ in range(energy_dim)]
        norm_ed = [[] for _ in range(energy_dim)]

    else:
        decoder = model.model
        energy = False

    for batch in tqdm(dataloader):
        if energy:
            x, z, e = batch
            e = e.cuda()
        else:
            x, z = batch
        x = x.cuda()
        z = z.cuda()
        output = decoder.decode(z)
        if energy:
            x_hat, e_hat = (
                output[:, :-energy_dim],
                output[:, -energy_dim:],
            )
            for i in range(energy_dim):
                mask = ~torch.isinf(e[:, i])
                if mask.sum() > 0:
                    filtered_e = e[:, i][mask]
                    filtered_e_hat = e_hat[:, i][mask]
                    diff = torch.abs(filtered_e - filtered_e_hat).detach()
                    norm_ed[i].append(diff)
                    if model.mean is not None:
                        filtered_e = filtered_e * model.std[i] + model.mean[i]
                        filtered_e_hat = filtered_e_hat * model.std[i] + model.mean[i]
                        diff = torch.abs(filtered_e - filtered_e_hat).detach()
                        ed[i].append(diff)
        else:
            x_hat = output

        x_hat = x_hat.unflatten(dim=1, sizes=(-1, 3))
        sd.append(calculate_3D_squared_distance(x_hat, x))

    results = {}

    if energy:
        for i in range(energy_dim):
            if model.mean is not None:
                diff = torch.cat(ed[i])
                results[f"{model.config['force_types'][i]}_mae"] = diff.mean()
                results[f"{model.config['force_types'][i]}_stdae"] = diff.std()
            diff = torch.cat(norm_ed[i])
            results[f"{model.config['force_types'][i]}_norm_mae"] = diff.mean()
            results[f"{model.config['force_types'][i]}_norm_stdae"] = diff.std()

    sd = torch.cat(sd)
    results["rmsd"] = torch.sqrt(sd.mean())
    print(results["rmsd"])
    results["mse"] = torch.mean(sd)
    results["std"] = torch.std(sd)

    return results


def evaluate_model(
    checkpoint_path, amino_acid, data_path, latent_path, energy_path=None, mode=""
):
    model, type = get_model(amino_acid, checkpoint_path, return_type=True)
    model = model.cuda()
    model.eval()
    model.freeze()

    os.makedirs(f"results/{mode}/HAE/", exist_ok=True)
    with open(f"results/{mode}/HAE/{type}_metrics.txt", "a", encoding="utf-8") as f:
        print(f"Amino: {amino_acid}", file=f)
        latents = LatentEvalDataset(
            amino_acid,
            data_path,
            latent_path,
            energy_path,
            iqr_filter_energy=(
                model.config["iqr_filter_energy"] if energy_path is not None else False
            ),
            normalize_energy=(
                model.config["normalize_energy"] if energy_path is not None else False
            ),
            interpolate_energy=(
                model.config["interpolate_energy"] if energy_path is not None else False
            ),
            inf_filter=model.config["inf_filter"] if energy_path is not None else False,
            force_types=(
                model.config["force_types"] if energy_path is not None else None
            ),
        )
        result_dataset = run_dataset(latents, model)
        print(
            "Dataset latents:",
            file=f,
        )
        for metric, value in result_dataset.items():
            if "mae" in metric.lower() or "rmsd" in metric.lower():
                print(f"{metric}: {value.cpu().detach().item():.3f}", file=f)

        random_latents = LatentDataset(
            amino_acid,
            data_path,
            latent_path,
            energy_path,
            iqr_filter_energy=(
                model.config["iqr_filter_energy"] if energy_path is not None else False
            ),
            normalize_energy=(
                model.config["normalize_energy"] if energy_path is not None else False
            ),
            interpolate_energy=(
                model.config["interpolate_energy"] if energy_path is not None else False
            ),
            force_types=(
                model.config["force_types"] if energy_path is not None else None
            ),
            inf_filter=model.config["inf_filter"] if energy_path is not None else False,
            increase_factor=1,
            border_pct=0,
        )
        # Initialize a dictionary to accumulate metric values across runs
        metric_values = {}

        for _ in range(2):
            random_latents.resample_data()
            result = run_dataset(random_latents, model)
            # Collect values for each metric
            for metric, value in result.items():
                if metric not in metric_values:
                    metric_values[metric] = []
                metric_values[metric].append(value)

        print("Random latents", file=f)
        # Calculate mean and standard deviation for each metric
        results_summary = {}
        for metric, values in metric_values.items():
            values = torch.tensor(values)
            mean = values.mean().cpu().detach().item()
            std_dev = values.std().cpu().detach().item()
            line = f"{mean:.3f} ± {std_dev:.3f}"
            results_summary[metric] = line
            if "mae" in metric.lower() or "rmsd" in metric.lower():
                print(f"{metric}: {line}", file=f)

        print(result_dataset, file=f)
        print(results_summary, file=f)


if __name__ == "__main__":
    aminos = ["ARG", "GLN", "GLU", "LYS", "MET"]
    aminos = ["ARG", "LYS", "MET"]
    modes = ["synth/high_energy", "full"]
    for amino in aminos:
        for mode in modes:
            with torch.no_grad():
                print(f"HAE {amino}")
                evaluate_model(
                    f"checkpoints/{mode}/HAE/_0",
                    amino,
                    data_path=f"dataset/{mode}/data",
                    latent_path=f"dataset/{mode}/HAE_latents_no_uni",
                    mode=mode,
                )
                print(f"HAE decoder {amino}")
                evaluate_model(
                    f"checkpoints/{mode}/HAEdecoder/_ratio_0.75/",
                    amino,
                    data_path=f"dataset/{mode}/data",
                    latent_path=f"dataset/{mode}/HAE_latents",
                    energy_path=(
                        f"dataset/{mode}/interpolated_energy"
                        if mode == "full"
                        else f"dataset/{mode}/energy"
                    ),
                    mode=mode,
                )
