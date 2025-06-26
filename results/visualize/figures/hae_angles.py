import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.ticker import FuncFormatter
from scipy.stats import vonmises_fisher

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

# sns.set_context("paper")
# sns.set_style("whitegrid")


def sample_uniform_sphere(n, dim):
    x = np.random.randn(n, dim)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def angle_distribution(points, num_samples=100000):
    idx1 = np.random.randint(0, len(points), num_samples)
    idx2 = np.random.randint(0, len(points), num_samples)
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    dots = np.sum(points[idx1] * points[idx2], axis=1)
    angles = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(angles)
    return angles


aminos = ["ARG", "LYS", "MET"]
modes = ["full", "synth/high_energy"]
titles = ["PDB Dataset", "Synthetic Dataset"]

colors = [
    "#EA6868",
    "#4DE769",
    "#559CD3",
]

fig, axs = plt.subplots(
    nrows=len(aminos),
    ncols=len(modes),
    figsize=(6 * len(modes), 4 * len(aminos)),
    sharex=True,
    sharey=True,  # <- Share y-axis for consistent scale
    constrained_layout=True,
)

# Convert axs to 2D array if only one row or column
axs = np.atleast_2d(axs)

for i, amino in enumerate(aminos):
    for j, mode in enumerate(modes):
        ax = axs[i, j]

        path_hae = f"dataset/{mode}/HAE_latents/{amino}.pt"
        points_hae = torch.load(path_hae, weights_only=True).cpu().numpy()

        path_no_hae = f"dataset/{mode}/HAE_latents_no_uni/{amino}.pt"
        points_no_hae = torch.load(path_no_hae, weights_only=True).cpu().numpy()

        n_points = points_hae.shape[0]
        dim = points_hae.shape[1]
        random_points = sample_uniform_sphere(n_points, dim)

        ax.hist(
            angle_distribution(points_no_hae),
            bins=200,
            density=True,
            alpha=1,
            label=rf"No uniformity loss, $\kappa$={vonmises_fisher.fit(points_no_hae)[1]:.3f}",
            color=colors[2],
        )

        ax.hist(
            angle_distribution(random_points),
            bins=200,
            density=True,
            alpha=0.8,
            label=rf"Uniformly generated points, $\kappa$={vonmises_fisher.fit(random_points)[1]:.3f}",
            color=colors[0],
        )
        ax.hist(
            angle_distribution(points_hae),
            bins=200,
            density=True,
            alpha=0.8,
            label=rf"HAE uniformity loss, $\kappa$={vonmises_fisher.fit(points_hae)[1]:.3f}",
            color=colors[1],
        )

        # # Per-plot title with amino acid and dimension
        # ax.set_title(f"{amino}, dim={dim}", fontsize=12)

        # ax.grid()
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/np.pi:.1f}π"))

        # Only show x-axis label on last row
        if i == len(aminos) - 1:
            ax.set_xlabel("Angle (radians)")
        else:
            ax.set_xticklabels([])

        # Only show y-axis label and ticks on first column
        if j == 0:
            ax.set_ylabel("Density")

        #  Column titles only on top row
        if i == 0:
            ax.set_title(titles[j], fontsize=14, fontweight="bold")

        ax.annotate(
            f"{amino}, D={dim}",
            xy=(0.02, 0.93),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=11,
            fontweight="normal",
        )

        # Add legend only once
        # if i == 0 and j == 1:
        ax.legend(loc="upper right")

# fig.suptitle("Angle Distributions Between Latent Vectors", fontsize=16, y=1.02)
os.makedirs("figures", exist_ok=True)
plt.savefig(
    "figures/hae_angle_distributions_amino_acids.pdf",
    bbox_inches="tight",
)
# plt.show()
