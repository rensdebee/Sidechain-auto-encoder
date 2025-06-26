import os
import re

import numpy as np

amino_acids = ["ARG", "LYS", "MET"]
datasets = ["PDB", "Synthetic"]
paths = [
    "results/full/torsion_energy",
    "results/synth/high_energy/torsion_energy",
]
periodicity_labels = {"dim1": "1", "dim2": "1+3"}
forces = ["PeriodicTorsionForce", "LJForce", "CoulombForce"]


def find_energy_predictor_files(amino, directory="."):
    pattern = re.compile(rf"energy_predictor_results_{amino}_dim(\d+)_\d+.txt")
    filenames, first_numbers = [], []
    for fname in os.listdir(directory):
        match = pattern.fullmatch(fname)
        if match:
            filenames.append(fname)
            first_numbers.append(int(match.group(1)))
    return filenames, first_numbers


def parse_file(file_path):
    with open(file_path) as f:
        content = f.read()
    data, current = {}, None
    for line in content.splitlines():
        line = line.strip()
        if line.endswith(":"):
            current = line[:-1]
            data[current] = {}
        elif "Mean:" in line and current:
            data[current]["Mean"] = float(line.split(": ")[1])
    return data


def bold_pair(mae, other_mae, norm, other_norm):
    # decide bolding for raw MAE
    if not np.isnan(mae) and not np.isnan(other_mae):
        mae_str = f"\\textbf{{{mae:.3f}}}" if mae <= other_mae else f"{mae:.3f}"
    else:
        mae_str = "N/A" if np.isnan(mae) else f"{mae:.3f}"
    # decide bolding for normalized MAE
    if not np.isnan(norm) and not np.isnan(other_norm):
        norm_str = f"\\textbf{{{norm:.3f}}}" if norm <= other_norm else f"{norm:.3f}"
    else:
        norm_str = "N/A" if np.isnan(norm) else f"{norm:.3f}"
    return f"{mae_str} [{norm_str}]"


# === parse everything ===
results = {aa: {} for aa in amino_acids}
for amino in amino_acids:
    for ds_idx, dataset in enumerate(datasets):
        results[amino][dataset] = {}
        filenames, dims = find_energy_predictor_files(amino, paths[ds_idx])
        if not filenames:
            continue

        # pick smallest and largest dim
        idx_min = min(range(len(dims)), key=lambda i: dims[i])
        idx_max = max(range(len(dims)), key=lambda i: dims[i])
        pick = {"dim1": idx_min, "dim2": idx_max}

        for dim_key, file_i in pick.items():
            fname = filenames[file_i]
            fpath = os.path.join(paths[ds_idx], fname)
            if not os.path.exists(fpath):
                continue
            parsed = parse_file(fpath)
            block = {}
            for force in forces:
                mae = parsed.get(f"val_{force}_mae_epoch", {}).get("Mean", np.nan)
                norm = parsed.get(f"val_{force}_norm_mae_epoch", {}).get("Mean", np.nan)
                block[force] = (mae, norm)
            results[amino][dataset][dim_key] = block

# === build LaTeX ===
latex = [
    r"\begin{tabular}{|c|c|c|ccc|}",
    r"\hline",
    r"\multirow{2}{*}{Amino Acid} & \multirow{2}{*}{Dataset} & \multirow{2}{*}{Periodicity} & \multicolumn{3}{c|}{MAE (kJ/mol) [Normalized MAE] ↓} \\",
    r"\cline{4-6}",
    r" & & & Torsion energy & LJ energy & Coulomb energy \\",
    r"\hline",
]

for amino in amino_acids:
    ds_blocks = []
    for dataset in datasets:
        if dataset not in results[amino] or not results[amino][dataset]:
            continue
        dim_data = results[amino][dataset]
        # prepare periodicity rows
        peri_rows = []
        for dim in ["dim1", "dim2"]:
            other = "dim2" if dim == "dim1" else "dim1"
            row = []
            for force in forces:
                mae, norm = dim_data[dim].get(force, (np.nan, np.nan))
                other_mae, other_norm = dim_data[other].get(force, (np.nan, np.nan))
                row.append(bold_pair(mae, other_mae, norm, other_norm))
            peri_rows.append(row)
        # difference row (unchanged)
        diff = []
        for force in forces:
            m1, n1 = dim_data["dim1"].get(force, (np.nan, np.nan))
            m2, n2 = dim_data["dim2"].get(force, (np.nan, np.nan))
            dm, dn = m2 - m1, n2 - n1
            sign_m = "-" if dm < 0 else "+"
            sign_n = "-" if dn < 0 else "+"
            diff.append(f"{sign_m}{abs(dm):.3f} [{sign_n}{abs(dn):.3f}]")
        ds_blocks.append({"label": dataset, "peri": peri_rows, "diff": diff})

    if not ds_blocks:
        continue

    total = len(ds_blocks) * 3
    # first dataset
    first = ds_blocks[0]
    latex.append(
        rf"\multirow{{{total}}}{{*}}{{{amino}}} & \multirow{{3}}{{*}}{{{first['label']}}} & {periodicity_labels['dim1']} & {' & '.join(first['peri'][0])} \\"
    )
    latex.append(
        rf" & & {periodicity_labels['dim2']} & {' & '.join(first['peri'][1])} \\"
    )
    latex.append(rf" & & \Delta & {' & '.join(first['diff'])} \\")
    latex.append(r"\cline{2-6}")

    # second dataset, if any
    if len(ds_blocks) > 1:
        sec = ds_blocks[1]
        latex.append(
            rf" & \multirow{{3}}{{*}}{{{sec['label']}}} & {periodicity_labels['dim1']} & {' & '.join(sec['peri'][0])} \\"
        )
        latex.append(
            rf" & & {periodicity_labels['dim2']} & {' & '.join(sec['peri'][1])} \\"
        )
        latex.append(rf" & & \Delta & {' & '.join(sec['diff'])} \\")
    latex.append(r"\hline")

latex.append(r"\end{tabular}")

print("\n".join(latex))
