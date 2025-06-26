import colorsys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D

# Data collection (unchanged)
aminos = ["ARG", "LYS", "MET"]

forces = ["PeriodicTorsionForce", "LJForce", "CoulombForce"]

force_names = {
    "PeriodicTorsionForce": "Torsion energy",
    "LJForce": "LJ energy",
    "CoulombForce": "Coulomb energy",
    "sum": "Summed energy",
}

lrs = [
    "0.001",
    "10",
    "1",
    "0.1",
]

lrs_markers = {"10": "o", "1": "s", "0.1": "^", "0.001": "D"}

datasets = ["full", "synth/high_energy"]
dataset_names = {"full": "PDB", "synth/high_energy": "Synth"}

models = [
    "HAE",
    "torsion",
    "mapping",
]
model_labels = {"HAE": "HAE", "torsion": "Torsion", "mapping": "Hybrid"}

group_colors = {
    "HAE (PDB)": "#4C72B0",
    "Torsion (PDB)": "#DD292F",
    "Hybrid (PDB)": "#298F44",
    "HAE (Synth)": "#9ABBEF",
    "Torsion (Synth)": "#EC6D72",
    "Hybrid (Synth)": "#6DECB3",
}

data = {}
show_sum = True
sum_only = False

for amino in aminos:
    data[amino] = {}
    for model in models:
        data[amino][model] = {}
        for dataset in datasets:
            data[amino][model][dataset] = {}
            for lr in lrs:
                data[amino][model][dataset][lr] = {}
                data[amino][model][dataset][lr]["sum"] = {}
                sum_pred = 0
                sum_act = 0
                try:
                    new_key_to_col, actual_energy, pred_energy, clash_metric = (
                        torch.load(
                            f"eval/{dataset}/{model}/{amino}/{lr}/energy_results.pt"
                        )
                    )
                    for i, force in enumerate(forces):
                        data[amino][model][dataset][lr][force] = {}
                        sum_pred += pred_energy[:, i]
                        sum_act += actual_energy[:, new_key_to_col[force]]
                        data[amino][model][dataset][lr][force]["act"] = actual_energy[
                            :, new_key_to_col[force]
                        ].numpy()
                        data[amino][model][dataset][lr][force]["pred"] = pred_energy[
                            :, i
                        ].numpy()
                except FileNotFoundError as e:
                    print(e)
                    for i, force in enumerate(forces):
                        empty = np.asarray([0, 0])
                        data[amino][model][dataset][lr][force]["act"] = empty
                        data[amino][model][dataset][lr][force]["pred"] = empty
                        data[amino][model][dataset][lr]["sum"]["act"] = empty
                        data[amino][model][dataset][lr]["sum"]["pred"] = empty
                data[amino][model][dataset][lr]["sum"]["act"] = sum_act.numpy()
                data[amino][model][dataset][lr]["sum"]["pred"] = sum_pred.numpy()

if show_sum:
    forces.append("sum")
if sum_only:
    forces = ["sum"]


def generate_tints(base_color, n_colors, hue_shift_range=0.2):
    # Convert base color to HLS
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Generate subtle hue shifts around base hue
    shifts = np.linspace(-hue_shift_range, hue_shift_range, n_colors)
    hues = [(h + shift) % 1.0 for shift in shifts]
    return [colorsys.hls_to_rgb(hue, l, s) for hue in hues]


for amino in aminos:
    for dataset in datasets:
        num_plots = len(models)
        fig, axes = plt.subplots(
            num_plots,
            len(forces),
            figsize=(6 * len(forces), 5 * num_plots),
            constrained_layout=True,
        )
        axes = np.atleast_2d(axes)
        if axes.shape[0] != num_plots:
            axes = axes.T
        for i, force in enumerate(forces):
            legend_handles = []
            ax = axes[:, i]
            print(len(ax))
            curr_min = np.inf
            curr_max = -np.inf
            for j, model in enumerate(models):
                legend_handles.append([])
                model_label = f"{model_labels[model]} ({dataset_names[dataset]})"
                tint = generate_tints(group_colors[model_label], len(lrs))
                for lr_idx, lr in enumerate(lrs):
                    actual = data[amino][model][dataset][lr][force]["act"]
                    predicted = data[amino][model][dataset][lr][force]["pred"]
                    r_value = np.corrcoef(actual, predicted)[0, 1]
                    mae = np.mean(np.abs(actual - predicted))
                    curr_min = min(curr_min, np.min(predicted))
                    curr_max = max(curr_max, np.max(predicted))
                    sns.scatterplot(
                        x=actual,
                        y=predicted,
                        ax=ax[j],
                        label=f"{lr}\n(MAE: {mae:.2f}, R: {r_value:.3f})",
                        marker=lrs_markers[lr],
                        color=tint[lr_idx],
                        linewidth=0,
                        s=5,
                        alpha=0.6,
                    )
                    legend_handles[-1].append(
                        Line2D(
                            [0],
                            [0],
                            marker=lrs_markers[lr],
                            color="none",
                            markerfacecolor=tint[lr_idx],
                            # markeredgecolor="black",
                            markersize=10,
                            label=f"{lr}\n(MAE: {mae:.2f}, R: {r_value:.3f})",
                        )
                    )
                if j == 0:
                    ax[j].set_title(
                        f"{force_names[force]} Predicted vs Calculated Energy",
                        fontsize=12,
                        fontweight="bold",
                    )
                if j == len(models) - 1:
                    ax[j].set_xlabel("OpenMM calculated Energy (kJ/mol)", fontsize=10)
                if i == 0:
                    ax[j].set_ylabel(
                        f"Predicted energy of {model_labels[model]} model (kJ/mol)",
                        fontsize=10,
                    )

            for j in range(len(models)):
                sns.lineplot(
                    x=[curr_min, curr_max],
                    y=[curr_min, curr_max],
                    color="red",
                    linestyle="--",
                    # label="Ideal Prediction",
                    ax=ax[j],
                )
                ax[j].legend(handles=legend_handles[j])
                ax[j].set_xlim([curr_min - 0.5, curr_max + 0.5])
                ax[j].set_ylim([curr_min - 0.5, curr_max + 0.5])
        fig.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(
            f"figures/act_pred_{amino}_{dataset_names[dataset]}.png",
            dpi=300,
            bbox_inches="tight",
        )
