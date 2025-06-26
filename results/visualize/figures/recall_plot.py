import numpy as np
import torch
from matplotlib import pyplot as plt

from amino.data.datasets import LatentEvalDataset

aminos = ["ARG", "LYS", "MET"]
modes = ["full", "synth/high_energy"]
dataset_names = {"full": "PDB", "synth/high_energy": "Synth"}
eval_path = "eval/"
models = ["HAE", "torsion", "mapping"]
model_label = {"HAE": "HAE", "torsion": "Torsion", "mapping": "Hybrid"}
lrs = [10, 1]
ks = np.arange(1, 2001, 10)
relevance_cutoffs = [10, 50]

amino_colors = {"ARG": "#4C72B0", "LYS": "#DD292F", "MET": "#298F44"}
mode_markers = {"full": ",", "synth/high_energy": ","}
mode_linestyles = {"full": "--", "synth/high_energy": "-."}


def precision_recall_at_k(true_sorted, pred_sorted, ks, relevant_cutoff):
    relevant_set = set(true_sorted[:relevant_cutoff])
    precisions, recalls = [], []
    for k in ks:
        top_k_pred = set(pred_sorted[:k])
        intersection = relevant_set & top_k_pred
        precision = len(intersection) / k
        recall = len(intersection) / relevant_cutoff
        precisions.append(precision * 100)
        recalls.append(recall * 100)
    return precisions, recalls


# Setup
fig, axs = plt.subplots(
    1, len(relevance_cutoffs), figsize=(7 * len(relevance_cutoffs), 5), sharey=True
)
used_labels = set()

# Loop over cutoffs
for cutoff_idx, cutoff in enumerate(relevance_cutoffs):
    ax = axs[cutoff_idx]
    for amino in aminos:
        for mode in modes:
            best_model_info = None
            lowest_energy = float("inf")

            for model in models:
                for lr in lrs:
                    path = f"{eval_path}/{mode}/{model}/{amino}/{lr}/energy_results.pt"
                    try:
                        _, all_energy, pred_energy, _ = torch.load(path)
                        avg_energy = all_energy.sum(dim=1).mean().item()
                        if avg_energy < lowest_energy:
                            best_model_info = (model, pred_energy, all_energy)
                            lowest_energy = avg_energy
                    except FileNotFoundError:
                        continue

            if best_model_info is not None:
                model, pred_energy, all_energy = best_model_info
                actual_sorted = torch.argsort(all_energy.sum(dim=1)).tolist()
                pred_sorted = torch.argsort(pred_energy.sum(dim=1)).tolist()
                _, recall = precision_recall_at_k(
                    actual_sorted, pred_sorted, ks, cutoff
                )

                label = f"{amino} ({dataset_names[mode]})"
                show_label = label not in used_labels
                used_labels.add(label)

                ax.plot(
                    ks,
                    recall,
                    label=label if show_label else None,
                    color=amino_colors[amino],
                    lw=2,
                    ls=mode_linestyles[mode],
                    marker=mode_markers[mode],
                    markersize=5,
                )

    ax.set_title(f"Recall@k\n(Top-{cutoff} calculated)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Top-k predicted", fontsize=12)
    # ax.set_xscale("log")
    # ax.set_xticks(ks)
    # ax.set_xticklabels(ks)
    # ax.set_ylim(, 105)

axs[0].set_ylabel("Recall (%)", fontsize=12)
fig.legend(
    *axs[0].get_legend_handles_labels(),
    loc="upper center",
    ncol=3,
    fontsize=11,
    frameon=True,
    title="Amino Acid + Dataset",
    title_fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig(
    "figures/BestModels_RecallAtK_AminoColors.pdf", dpi=300, bbox_inches="tight"
)
