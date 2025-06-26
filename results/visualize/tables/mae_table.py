import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from amino.data.datasets import iqr_filtering_energy, normalizing_energy

# Data collection
aminos = ["ARG", "LYS", "MET"]
datasets = {"full": "PDB", "synth/high_energy": "Synthetic"}
lrs = ["10", "1", "0.1", "0.001"]
models = ["HAE", "torsion", "mapping"]
model_names = {"HAE": "HAE", "torsion": "Torsion", "mapping": "Hybrid"}
forces = ["PeriodicTorsionForce", "LJForce", "CoulombForce"]
force_names = {
    "PeriodicTorsionForce": "Torsion",
    "LJForce": "LJ",
    "CoulombForce": "Coulomb",
}
show_sum_only = False


def load_data():
    """Load and process data from dist_table.py"""
    # This should be replaced with actual data loading from dist_table.py
    # For demonstration, we'll create mock data
    data = {}
    for amino in aminos:
        data[amino] = {}
        for dataset_key, dataset_name in datasets.items():
            data[amino][dataset_name] = {}
            for lr in lrs:
                data[amino][dataset_name][lr] = {}
                for model in models:
                    data[amino][dataset_name][lr][model] = {}
                    trainings_energy_path = f"dataset/{dataset_key}/energy"
                    data_key_to_col, data_energy = torch.load(
                        f"{trainings_energy_path}/{amino}.pt", weights_only=True
                    )
                    data_energy = iqr_filtering_energy(data_energy)
                    cols = []
                    for i, name in enumerate(forces):
                        col = data_key_to_col[name]
                        cols.append(col)
                    data_energy = data_energy[:, cols]
                    # Compute row sums
                    row_sums = data_energy.sum(dim=1, keepdim=True)  # shape: (2, 1)
                    # Append as a new column
                    data_energy = torch.cat([data_energy, row_sums], dim=1)
                    _, data_mean, data_std = normalizing_energy(data_energy)

                    pred_sum = 0
                    sum = 0
                    new_key_to_col, actual_energy, pred_energy, clash_metric = (
                        torch.load(
                            f"eval/{dataset_key}/{model}/{amino}/{lr}/energy_results.pt"
                        )
                    )
                    for i, force in enumerate(forces):
                        energy = actual_energy[:, new_key_to_col[force]]
                        norm_energy = (energy - data_mean[i]) / data_std[i]
                        energy_pred = pred_energy[:, i]
                        pred_norm_energy = (energy_pred - data_mean[i]) / data_std[i]
                        data[amino][dataset_name][lr][model][force] = (
                            torch.abs(energy_pred - energy).mean().numpy(),
                            torch.abs(pred_norm_energy - norm_energy).mean().numpy(),
                        )
                        sum += energy
                        pred_sum += energy_pred
                    sum_norm = (sum - data_mean[len(forces)]) / data_std[len(forces)]
                    pred_sum_norm = (pred_sum - data_mean[len(forces)]) / data_std[
                        len(forces)
                    ]
                    data[amino][dataset_name][lr][model]["sum"] = (
                        torch.abs(pred_sum - sum).mean().numpy(),
                        torch.abs(pred_sum_norm - sum_norm).mean().numpy(),
                    )
    return data


def generate_latex_table(
    data, aminos, datasets, lrs, models, forces, force_names, show_sum_only=False
):
    # Select forces
    table_forces = (
        ["sum"] if show_sum_only else [f for f in forces if f in force_names] + ["sum"]
    )

    # Compute minima
    min_mean = {}
    min_norm = {}
    for force in table_forces:
        for amino in aminos:
            for ds_name in datasets.values():
                values, norms = [], []
                for model in models:
                    for lr in lrs:
                        entry = (
                            data.get(amino, {})
                            .get(ds_name, {})
                            .get(lr, {})
                            .get(model, {})
                        )
                        mean, norm = entry.get(force, (None, None))
                        if mean is not None:
                            values.append(mean)
                            norms.append(norm)
                if values:
                    min_mean[(force, amino, ds_name)] = min(values)
                    min_norm[(force, amino, ds_name)] = min(norms)

    # Build LaTeX
    n_aminos = len(aminos)
    n_datasets = len(datasets)
    colspec = "|c|c|c|" + "c|" * (n_aminos * n_datasets)
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\hline"]

    # Header
    aminocells = " & ".join(
        [
            f"\multicolumn{{{n_datasets}}}{{c|}}{{\shortstack{{{amino}}}}}"
            for amino in aminos
        ]
    )
    lines.append(
        rf"\multirow{{2}}{{*}}{{\shortstack{{Energy\\Type}}}} & "
        rf"\multirow{{2}}{{*}}{{\shortstack{{Model}}}} & "
        rf"\multirow{{2}}{{*}}{{\shortstack{{Step\\Size}}}} & {aminocells}\\"
    )
    lines.append(rf"\cline{{4-{3 + n_aminos * n_datasets}}}")
    dsrow = " & ".join(["", "", ""] + list(datasets.values()) * n_aminos) + r"\\"
    lines.extend([dsrow, "\\hline"])

    # Rows
    for force in table_forces:
        disp_force = force_names.get(force, force.capitalize())
        total = len(models) * len(lrs)
        lines.append(
            rf"\multirow{{{total}}}{{*}}{{\shortstack{{{disp_force}\\MAE\\(kJ/mol   )}}}}"
        )
        for midx, model in enumerate(models):
            for li, lr in enumerate(lrs):
                cells = []
                for amino in aminos:
                    for ds_name in datasets.values():
                        entry = (
                            data.get(amino, {})
                            .get(ds_name, {})
                            .get(lr, {})
                            .get(model, {})
                        )
                        mean, norm = entry.get(force, (None, None))
                        if mean is None:
                            cells.append("N/A")
                        else:
                            val_fmt = (
                                f"\\textbf{{{mean:.2f}}}"
                                if mean == min_mean.get((force, amino, ds_name))
                                else f"{mean:.2f}"
                            )
                            norm_fmt = (
                                f"\\textbf{{[{norm:.2f}]}}"
                                if norm == min_norm.get((force, amino, ds_name))
                                else f"[{norm:.2f}]"
                            )
                            cells.append(val_fmt + norm_fmt)
                prefix = (
                    rf"& \multirow{{{len(lrs)}}}{{*}}{{{model_names[model]}}} & {lr}"
                    if li == 0
                    else rf"& & {lr}"
                )
                lines.append(prefix + " & " + " & ".join(cells) + r" \\")
            if midx < len(models) - 1:
                lines.append(rf"\cline{{2-{3 + n_aminos * n_datasets}}}")
        lines.append("\\hline\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# Main
if __name__ == "__main__":
    data = load_data()
    print(
        generate_latex_table(
            data, aminos, datasets, lrs, models, forces, force_names, show_sum_only
        )
    )
