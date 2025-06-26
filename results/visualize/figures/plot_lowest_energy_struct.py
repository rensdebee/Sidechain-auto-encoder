import numpy as np
import torch
from matplotlib import pyplot as plt
from sympy import false

from amino.data.datasets import (
    LatentEvalDataset,
    SidechainDataset,
    iqr_filtering_energy,
    normalizing_energy,
)
from amino.utils.utils import create_clean_path, write_pdb

aminos = ["ARG", "LYS", "MET"]
forces = ["PeriodicTorsionForce", "LJForce", "CoulombForce"]
modes = ["full", "synth", "synth/high_energy"]
datasets = ["PDB", "Synth", "High Energy Synth"]
dataset_names = ["PDB", "Synth", "Synth"]
lrs = [10, 1]
eval_path = "eval/"
models = ["HAE", "torsion", "mapping"]
model_label = ["HAE", "Torsion", "Hybrid"]
ks = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 200, 300, 500]
data = {}
group_colors = {
    "HAE": "#4C72B0",
    "torsion": "#DD292F",
    "mapping": "#298F44",
}
show_all = False

for amino in aminos:
    data[amino] = {}
    for mode in modes:
        print(amino, mode)
        data[amino][mode] = {}
        dataset = LatentEvalDataset(
            amino,
            f"dataset/{mode}/data",
            f"dataset/{mode}/torsion_latents_dim1",
            f"dataset/{mode}/energy",
            force_types=forces,
            iqr_filter_energy=True,
            normalize_energy=True,
            fixed_O=True,
            inf_filter=True,
        )
        data[amino][mode]["dataset"] = []
        for k_idx, k in enumerate(ks):
            # Normalized top_k
            all_dataset_energy = dataset.energy * dataset.std + dataset.mean
            val, idxs = torch.topk(
                all_dataset_energy.sum(dim=1),
                k=(
                    all_dataset_energy.shape[0]
                    if k_idx == len(ks) - 1 and show_all
                    else k
                ),
                largest=False,
                sorted=True,
            )
            # Unnormalized for plotting
            data[amino][mode]["dataset"].append(
                all_dataset_energy[idxs].sum(dim=1).mean()
            )
        # For each LR
        for model in models:
            data[amino][mode][model] = {}
            path = f"{eval_path}/{mode}/{model}/{amino}"
            for lr in lrs:
                data[amino][mode][model][lr] = []
                for k_idx, k in enumerate(ks):
                    try:
                        new_key_to_col, all_energy, pred_energy, clash_metric = (
                            torch.load(f"{path}/{lr}/energy_results.pt")
                        )
                        # top_k
                        val, idxs = torch.topk(
                            all_energy.sum(dim=1),
                            k=(
                                all_energy.shape[0]
                                if k_idx == len(ks) - 1 and show_all
                                else min(k, all_energy.shape[0])
                            ),
                            largest=False,
                            sorted=True,
                        )
                        # Unnormalized for plotting
                        data[amino][mode][model][lr].append(
                            all_energy[idxs].sum(dim=1).mean()
                        )
                    except FileNotFoundError as e:
                        print("File not found", e)
                        data[amino][mode][model][lr].append(np.nan)

nrows = len(lrs)
ncols = len(aminos)
fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(7 * ncols, 4 * nrows),
    sharey="col",
    # squeeze=False,
    # gridspec_kw={"wspace": 0.15, "hspace": 0.3},
)
for i, amino in enumerate(aminos):
    for j, lr in enumerate(lrs):
        ax = axs[j, i]
        ax.plot(
            ks,
            data[amino]["full"]["dataset"],
            label="PDB training dataset",
            color="#FF5FA2",
            lw=2,
            ms=6,
        )
        ax.plot(
            ks,
            data[amino]["synth/high_energy"]["dataset"],
            label="Synthetic training dataset",
            color="#FFB45F",
            lw=2,
            ms=6,
        )
        ax.plot(
            ks,
            data[amino]["synth"]["dataset"],
            label="Synthetic dataset (Including low-energy Samples)",
            color="#E25FFF",
            lw=2,
            ms=6,
        )

        for p, model in enumerate(models):
            for l, dataset in enumerate(modes):
                marker = "^" if dataset == "synth/high_energy" else "o"
                color = group_colors[f"{model}"]
                if not np.all(np.isnan(data[amino][dataset][model][lr])):
                    ax.plot(
                        ks,
                        data[amino][dataset][model][lr],
                        color=color,
                        ls="--",
                        lw=2,
                        marker=marker,
                        ms=6,
                        label=f"{model_label[p]} ({dataset_names[l]})",
                    )
        if i == 0:
            ax.set_ylabel(f"Average summed energy (kJ/mol)", fontsize=12)
        if j == len(lrs) - 1:
            ax.set_xlabel("Top-k lowest energy samples", fontsize=12)
        ax.set_title(f"{amino}, step size = {lr}", fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels(ks)
        # ax.legend()

handles, labels = plt.gca().get_legend_handles_labels()
plt.figlegend(
    handles,
    labels,
    loc="upper center",
    ncol=len(models) + 1,
    frameon=True,
    fontsize=11,
    facecolor="white",
    edgecolor="0.8",
    title="Model & Dataset",
    title_fontsize=12,
    # bbox_to_anchor=(0.5, 0.01),
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig("figures/Top-K_energy_full.pdf", dpi=300, bbox_inches="tight")
