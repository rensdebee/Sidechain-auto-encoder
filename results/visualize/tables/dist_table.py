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
use_actual_energy = True


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

                    sum = 0
                    new_key_to_col, actual_energy, pred_energy, clash_metric = (
                        torch.load(
                            f"eval/{dataset_key}/{model}/{amino}/{lr}/energy_results.pt"
                        )
                    )
                    if not use_actual_energy:
                        actual_energy = pred_energy
                    for i, force in enumerate(forces):
                        energy = actual_energy[:, new_key_to_col[force]]
                        norm_energy = (energy - data_mean[i]) / data_std[i]
                        data[amino][dataset_name][lr][model][force] = (
                            energy.mean().numpy(),
                            norm_energy.mean().numpy(),
                        )
                        sum += energy
                    if lr == "1" or lr == "10":
                        print(amino, dataset_name, model)
                        print(sum.mean())
                    sum_norm = (sum - data_mean[-1]) / data_std[-1]
                    data[amino][dataset_name][lr][model]["sum"] = (
                        sum.mean().numpy(),
                        sum_norm.mean().numpy(),
                    )
    return data


import itertools


def generate_latex_table(
    data, aminos, datasets, lrs, models, forces, force_names, show_sum_only=False
):
    """
    Generate a LaTeX table from nested data dictionary, with normalized values in brackets
    and bold highlighting of the lowest mean energy per force, dataset, and step size.

    Args:
        data: dict[amino][dataset_name][lr][model][force] = (mean, norm_mean)
        aminos: list of amino keys (e.g., ["ARG","LYS","MET"])
        datasets: dict mapping dataset_key->dataset_name
        lrs: list of step sizes as strings
        models: list of models (e.g., ["HAE","torsion"])
        forces: list of force keys
        force_names: mapping of force key to display name
        show_sum_only: if True, only include the "sum" force
    Returns:
        A string containing the LaTeX tabular environment.
    """
    # Determine which forces to include
    if show_sum_only:
        table_forces = ["sum"]
    else:
        table_forces = [f for f in forces if f in force_names] + ["sum"]

    # Compute minimal mean across model and lr for each force, amino, and dataset
    min_mean = {}
    for force in table_forces:
        for amino in aminos:
            for ds_key, ds_name in datasets.items():
                values = []
                for model in models:
                    for lr in lrs:
                        mean, _ = data[amino][ds_name][lr][model].get(
                            force, (None, None)
                        )
                        if mean is not None:
                            values.append(mean)
                if values:
                    min_mean[(force, amino, ds_name)] = min(values)

    # Begin LaTeX
    n_aminos = len(aminos)
    n_datasets = len(datasets)
    colspec = "|c|c|c|" + "c|" * (n_aminos * n_datasets)
    lines = [f"\\begin{{tabular}}{{{colspec}}}", r"\hline"]

    # Header rows
    aminocells = " & ".join(
        [
            f"\\multicolumn{{{n_datasets}}}{{c|}}{{\\shortstack{{{amino}}}}}"
            for amino in aminos
        ]
    )
    lines.append(
        rf"\multirow{{2}}{{*}}{{\shortstack{{Energy\\Type}}}} & "
        rf"\multirow{{2}}{{*}}{{\shortstack{{Model}}}} & "
        rf"\multirow{{2}}{{*}}{{\shortstack{{Step\\Size}}}} & {aminocells}\\"
    )
    lines.append(r"\cline{4-" + str(3 + n_aminos * n_datasets) + r"}")
    ds_names = list(datasets.values())
    dsrow = " & ".join(["", "", ""] + ds_names * n_aminos) + r"\\"
    lines.append(dsrow)
    lines.append(r"\hline")

    # Data rows
    for force in table_forces:
        disp_force = force_names.get(force, force.capitalize())
        total_rows = len(models) * len(lrs)
        lines.append(
            rf"\multirow{{{total_rows}}}{{*}}{{\shortstack{{{disp_force}\\(kJ/mol)}}}}"
        )
        for midx, model in enumerate(models):
            for li, lr in enumerate(lrs):
                # Build data cells
                cells = []
                for amino in aminos:
                    for ds_key, ds_name in datasets.items():
                        mean, norm = data[amino][ds_name][lr][model].get(
                            force, (None, None)
                        )
                        if mean is None:
                            cells.append("")
                        else:
                            text = f"{mean:.2f} [{norm:.2f}]"
                            if min_mean.get((force, amino, ds_name)) == mean:
                                cells.append(rf"\textbf{{{text}}}")
                            else:
                                cells.append(text)
                # Row prefix
                if li == 0:
                    prefix = rf"& \multirow{{{len(lrs)}}}{{*}}{{{model_names[model]}}} & {lr}"
                else:
                    prefix = rf"& & {lr}"
                row = prefix + " & " + " & ".join(cells) + r" \\"
                lines.append(row)
                # mid-cline after last lr of a model block
                if li == len(lrs) - 1 and midx < len(models) - 1:
                    lines.append(rf"\cline{{2-{3 + n_aminos * n_datasets}}}")
        lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# Main execution
if __name__ == "__main__":
    # Load and process data
    data = load_data()

    # Generate LaTeX table
    latex_table = generate_latex_table(
        data,
        aminos,
        datasets,
        lrs,
        models,
        forces,
        force_names,
        show_sum_only=show_sum_only,
    )
    print(latex_table)
