import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from amino.data.datasets import iqr_filtering_energy, normalizing_energy

cmaps = [
    "Greys",
    "Reds",
    "Blues",
    "Greens",
    "Oranges",
    "Purples",
]

colors = [
    "darkgray",
    "tomato",
    "skyblue",
    "limegreen",
    "orange",
    "orchid",
]

force_names = {
    "PeriodicTorsionForce": "Torsion",
    "LJForce": "LJ",
    "CoulombForce": "Coulomb",
}


def plot_energy_dist(key_to_col, energys, labels, iqr=False, plot_sum=False, kde=True):
    # plt.rcParams.update(
    #     {
    #         "font.family": "serif",
    #         "axes.labelsize": 12,
    #         "axes.titlesize": 14,
    #         "legend.fontsize": 10,
    #         "xtick.labelsize": 10,
    #         "ytick.labelsize": 10,
    #     }
    # )

    colors = sns.color_palette("tab10", len(energys))

    valid_keys = [
        key
        for key, col in key_to_col.items()
        if key not in ["CustomTorsionForce", "CMAPTorsionForce", "CMMotionRemover"]
    ]
    if plot_sum:
        valid_keys.append(("SUM"))
    if not valid_keys:
        print("No valid energy types to plot.")
        return

    if len(valid_keys) % 2 == 0:
        num_plots = len(valid_keys)
    else:
        valid_keys.append("OFFAXIS")
        num_plots = len(valid_keys)
    fig, axes = plt.subplots(
        num_plots // 2, 2, figsize=(12, 4 * (num_plots // 2)), constrained_layout=True
    )

    if num_plots == 1:
        axes = [axes]
    axes = axes.flatten()
    new_energys = []
    for energy in energys:
        if iqr:
            energy = iqr_filtering_energy(energy)
        new_energys.append(energy)

    for ax, key in zip(axes, valid_keys):
        if key == "OFFAXIS":
            ax.set_visible(False)
            continue
        if key == "SUM":
            for i, energy in enumerate(new_energys):
                sns.histplot(
                    energy.sum(dim=1)[~torch.isinf(energy[:, col])].cpu().numpy(),
                    stat="density" if len(energys) != 1 else "count",
                    bins=max(
                        10,
                        (
                            energy.shape[0] // 1000
                            if energy.shape[0] > 10000
                            else energy.shape[0] // 10
                        ),
                    ),
                    kde=kde,
                    edgecolor=None,
                    ax=ax,
                    label=labels[i],
                    color=colors[i],
                )
            ax.set_title(
                f"Distribution of Summed energy terms", fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Energy Value (kJ/mol)", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            # if len(energys) != 1:
            #     ax.legend()
            continue
        col = key_to_col[key]
        for i, energy in enumerate(new_energys):
            # energy, _, _ = normalizing_energy(energy)
            sns.histplot(
                energy[:, col][~torch.isinf(energy[:, col])].cpu().numpy(),
                stat="density" if len(energys) != 1 else "count",
                bins=max(
                    10,
                    (
                        energy.shape[0] // 1000
                        if energy.shape[0] > 10000
                        else energy.shape[0] // 10
                    ),
                ),
                kde=kde,
                ax=ax,
                edgecolor=None,
                label=labels[i],
                color=colors[i],
            )
        ax.set_title(
            f"Distribution of {force_names[key]} energy", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Energy Value (kJ/mol)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        # if len(energys) != 1:
        #     ax.legend()
    # Add legend
    handles, labels = fig.gca().get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(energys),
        frameon=True,
        fontsize=11,
        facecolor="white",
        edgecolor="0.8",
        title_fontsize=12,
        bbox_to_anchor=(0.5, 0.96),
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.93])
    return fig


def plot_energy_TF(key_to_col, actual_energies, pred_energies, lrs):
    sns.set_context("paper")
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    valid_keys = [key for key, col in key_to_col.items()]

    num_plots = len(valid_keys)
    fig, axes = plt.subplots(
        num_plots,
        len(lrs),
        figsize=(6 * len(lrs), 5 * num_plots),
        constrained_layout=True,
    )

    if num_plots == 1:
        axes = [axes]

    for j, key in enumerate(valid_keys):
        ax = axes[j]
        col = key_to_col[key]
        curr_min = np.inf
        curr_max = -np.inf
        for i, (actual_energy, pred_energy, lr) in enumerate(
            zip(actual_energies, pred_energies, lrs)
        ):
            actual = (
                actual_energy[:, col][~torch.isinf(actual_energy[:, col])].cpu().numpy()
            )
            predicted = (
                pred_energy[:, col][~torch.isinf(actual_energy[:, col])].cpu().numpy()
            )
            r_value = np.corrcoef(actual, predicted)[0, 1]
            mae = np.mean(np.abs(actual - predicted))
            curr_min = min(curr_min, np.min(actual))
            curr_max = max(curr_max, np.max(actual))
            sns.scatterplot(
                x=actual,
                y=predicted,
                ax=ax[i],
                label=f"Energy Predictions + Density\n(MAE: {mae:.2f}, R: {r_value:.3f})",
                color=colors[i],
                alpha=1,
            )
            # sns.kdeplot(
            #     x=actual,
            #     y=predicted,
            #     cmap=cmaps[i],
            #     fill=True,
            #     cbar=True,
            #     ax=ax[i],
            #     alpha=0.8,
            # )
            # ax[i].hexbin(
            #     actual,
            #     predicted,
            #     gridsize=(10, 10),
            #     mincnt=2,
            #     cmap="Blues",
            # )

            ax[i].set_title(
                f"{key} Actual vs Predicted\n(lr: {lr})",
                fontsize=12,
                fontweight="bold",
            )
            ax[i].set_xlabel("Actual", fontsize=10)
            ax[i].set_ylabel("Predicted", fontsize=10)
            ax[i].legend()

        for i in range(len(lrs)):
            sns.lineplot(
                x=[curr_min, curr_max],
                y=[curr_min, curr_max],
                color="red",
                linestyle="--",
                label="Ideal Prediction",
                ax=ax[i],
            )
            ax[i].set_xlim([curr_min - 0.5, curr_max + 0.5])
            ax[i].set_ylim([curr_min - 0.5, curr_max + 0.5])

    return fig


def plot_error_hist(key_to_col, actual_energies, pred_energies, lrs):
    sns.set_context("paper")
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    valid_keys = [key for key, col in key_to_col.items()]

    num_plots = len(valid_keys)
    fig, axes = plt.subplots(
        num_plots,
        len(lrs),
        figsize=(6 * len(lrs), 5 * num_plots),
        constrained_layout=True,
    )

    if num_plots == 1:
        axes = [axes]

    for j, key in enumerate(valid_keys):
        ax = axes[j]
        col = key_to_col[key]
        for i, (actual_energy, pred_energy, lr, color) in enumerate(
            zip(actual_energies, pred_energies, lrs, colors)
        ):
            actual = (
                actual_energy[:, col][~torch.isinf(actual_energy[:, col])].cpu().numpy()
            )
            predicted = (
                pred_energy[:, col][~torch.isinf(actual_energy[:, col])].cpu().numpy()
            )
            error = np.abs(actual - predicted)
            mean = np.mean(error)
            median = np.median(error)
            sns.histplot(error, ax=ax[i], color=color, bins=100)
            ax[i].axvline(x=mean, c="k", ls="-", lw=2, label=f"Mean: {mean:.3f}")
            ax[i].axvline(x=median, c="k", ls="--", lw=2, label=f"Median: {median:.3f}")

            ax[i].set_title(
                f"{key} Absolute Error Distribution\n(lr: {lr})",
                fontsize=12,
                fontweight="bold",
            )
            ax[i].set_xlabel("Absolute Error", fontsize=10)
            ax[i].set_ylabel("Count", fontsize=10)
            ax[i].legend()

    return fig
