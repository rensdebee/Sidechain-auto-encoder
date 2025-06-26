import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

# Data collection (unchanged)
aminos = ["ARG", "LYS", "MET"]
forces = ["PeriodicTorsionForce", "LJForce", "CoulombForce"]
lrs = ["10", "1", "0.1", "0.001"]
datasets = ["full", "synth/high_energy"]
models = [
    "HAE",
    "torsion",
    "mapping",
]
group_colors = {
    "HAE (PDB)": "#4C72B0",
    "Torsion (PDB)": "#DD292F",
    "Hybrid (PDB)": "#298F44",
    "HAE (Synth)": "#9ABBEF",
    "Torsion (Synth)": "#EC6D72",
    "Hybrid (Synth)": "#6DECB3",
}
force_names = {
    "PeriodicTorsionForce": "Torsion energy",
    "LJForce": "LJ energy",
    "CoulombForce": "Coulomb energy",
    "sum": "Summed energy",
}
data = {}
show_sum = True
sum_only = True

for amino in aminos:
    data[amino] = {}
    for model in models:
        data[amino][model] = {}
        for dataset in datasets:
            data[amino][model][dataset] = {}
            for lr in lrs:
                data[amino][model][dataset][lr] = {}
                sum_pred = 0
                sum_act = 0
                try:
                    new_key_to_col, actual_energy, pred_energy, clash_metric = (
                        torch.load(
                            f"eval/{dataset}/{model}/{amino}/{lr}/energy_results.pt"
                        )
                    )
                    for i, force in enumerate(forces):
                        sum_pred += pred_energy[:, i]
                        sum_act += actual_energy[:, new_key_to_col[force]]
                        mae = torch.abs(
                            pred_energy[:, i] - actual_energy[:, new_key_to_col[force]]
                        ).numpy()
                        data[amino][model][dataset][lr][force] = mae
                except FileNotFoundError as e:
                    print(e)
                    for i, force in enumerate(forces):
                        mae = np.asarray([0, 0])
                        data[amino][model][dataset][lr][force] = mae
                        data[amino][model][dataset][lr]["sum"] = mae
                sum_mae = torch.abs(sum_pred - sum_act).numpy()
                data[amino][model][dataset][lr]["sum"] = sum_mae

if show_sum:
    forces.append("sum")
if sum_only:
    forces = ["sum"]
# Prepare plot data with new grouping order
plot_data = {}
for amino in aminos:
    plot_data[amino] = {}
    for force in forces:
        plot_data[amino][force] = {"means": [], "stds": []}
        for lr in lrs:
            means_per_lr = []
            stds_per_lr = []
            # New order: HAE-full, torsion-full, HAE-synth, torsion-synth
            for dataset in datasets:
                for model in models:
                    mae_array = data[amino][model][dataset][lr][force]
                    means_per_lr.append(mae_array.mean())
                    stds_per_lr.append(mae_array.std())
            plot_data[amino][force]["means"].append(means_per_lr)
            plot_data[amino][force]["stds"].append(stds_per_lr)

# Plot configuration
plt.figure(figsize=(18, 12))
bar_width = 0.35 / len(models)  # Slightly wider bars
gap_width = 0.05  # Gap between full and synth groups

# New color scheme with distinct group colors
condition_labels = list(group_colors.keys())
condition_colors = [group_colors[label] for label in condition_labels]

# Positions for groups: [HAE-full, torsion-full, (gap), HAE-synth, torsion-synth]
x_base = np.arange(len(lrs))  # Base positions for learning rates
x_positions = []
for i in range(len(models)):
    x_positions.append(x_base - (0.5 + i) * bar_width - gap_width / 2)
x_positions.reverse()
for i in range(len(models)):
    x_positions.append(x_base + (0.5 + i) * bar_width + gap_width / 2)

# Compute per-column (force-wise) Y limits
ylim_by_force = {}

for j, force in enumerate(forces):
    all_mae_values = []
    for i, amino in enumerate(aminos):
        means_data = plot_data[amino][force]["means"]
        stds_data = plot_data[amino][force]["stds"]
        for m, s in zip(means_data, stds_data):
            for mean, std in zip(m, s):
                all_mae_values.append(mean + 1)
                # all_mae_values.append(mean + std)
                # all_mae_values.append(mean - std)
    ymin = 0
    ymax = max(all_mae_values)
    ylim_by_force[force] = (ymin, ymax)

nrows = len(forces)
ncols = len(aminos)
fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(7 * ncols, 5 * nrows),
    # squeeze=False,
    # gridspec_kw={"wspace": 0.15, "hspace": 0.3},
)
axs = np.atleast_2d(axs)
# Create 3x3 grid of subplots
for i, amino in enumerate(aminos):
    for j, force in enumerate(forces):
        ax = axs[j, i]
        ax.set_ylim(ylim_by_force[force])
        means_data = plot_data[amino][force]["means"]  # [lrs][conditions]
        stds_data = plot_data[amino][force]["stds"]

        # Plot bars for each condition
        for k in [x + i for i in range(len(models)) for x in (0, len(models))]:
            ax.bar(
                x_positions[k],
                [m[k] for m in means_data],
                width=bar_width,
                # yerr=[s[k] for s in stds_data],
                color=condition_colors[k],
                label=condition_labels[k],
                capsize=4,
                edgecolor="black",
                linewidth=0.5,
                error_kw={"elinewidth": 1.5, "capthick": 1.5},
            )

        # Configure subplot appearance
        ax.set_xticks(x_base)
        ax.set_xticklabels(lrs)
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")

        # Add group separators
        for x in x_base:
            ax.axvline(
                x + gap_width / 2, color="gray", linestyle=":", alpha=0.7, linewidth=1
            )

        if i == 0:
            ax.set_ylabel(f"{force_names[force]}\nMAE (kJ/mol)", fontsize=12)
        if j == 0:
            ax.set_title(f"{amino}", fontsize=12, fontweight="bold")
        if j == len(forces) - 1:
            ax.set_xlabel("Step Size", fontsize=12)

# Add legend and overall title
handles, labels = plt.gca().get_legend_handles_labels()
plt.figlegend(
    handles,
    labels,
    loc="upper center",
    ncol=len(models),
    frameon=True,
    fontsize=11,
    facecolor="white",
    edgecolor="0.8",
    title="Model & Dataset",
    title_fontsize=12,
    # bbox_to_anchor=(0.5, 0.01),
)
# plt.suptitle(
#     "Mean Absolute Error (MAE) by Amino Acid, Force Type, and Training Configuration",
#     fontsize=14,
#     fontweight="bold",
#     y=0.98,
# )

# Final layout adjustments
plt.tight_layout(rect=[0, 0.0, 1, 0.85])
plt.savefig("figures/mae_comparison_grid_sum_only.pdf", dpi=300, bbox_inches="tight")
