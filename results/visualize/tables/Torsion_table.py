import os
import re
from collections import defaultdict


def parse_file_content(content):
    metrics = {}
    params = {}

    # Extract parameters (amino acid)
    amino_acid_match = re.search(r"'amino_acid': '([A-Z]{3})'", content)
    if amino_acid_match:
        params["amino_acid"] = amino_acid_match.group(1)

    # Extract metrics
    current_metric = None
    for line in content.split("\n"):
        if "val_" in line:
            parts = line.split(":")
            metric_name = parts[0].strip()
            current_metric = metric_name
            metrics[current_metric] = {}
        elif current_metric:
            if "Mean:" in line:
                mean = re.search(r"Mean:\s*([0-9.]+)", line).group(1)
                metrics[current_metric]["mean"] = float(mean)
            elif "Std:" in line:
                std = re.search(r"Std:\s*([0-9.]+)", line).group(1)
                metrics[current_metric]["std"] = float(std)

    return params, metrics


def collect_data(folder_path, dataset_type, data):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(".txt") or not "torsion" in root:
                continue
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                content = f.read()
                params, metrics = parse_file_content(content)
                if "amino_acid" not in params:
                    continue
                aa = params["amino_acid"]

                # Determine file type
                if "val_rmsd_epoch" in metrics:  # Decoder file
                    data[aa][dataset_type]["rmsd"] = metrics["val_rmsd_epoch"]["mean"]
                else:  # Energy file
                    pattern = re.compile(
                        rf"energy_predictor_results_{aa}_dim(\d+)_\d+.txt"
                    )
                    match = pattern.fullmatch(file)
                    if match:
                        dim = int(match.group(1))
                        if dim <= 10:
                            continue
                    else:
                        continue
                    energy_metrics = {}
                    for force in ["PeriodicTorsionForce", "LJForce", "CoulombForce"]:
                        mae_key = f"val_{force}_mae_epoch"
                        norm_key = f"val_{force}_norm_mae_epoch"
                        if mae_key in metrics and norm_key in metrics:
                            energy_metrics[force] = (
                                "%.3f" % metrics[mae_key]["mean"],
                                "%.3f" % metrics[norm_key]["mean"],
                            )
                    data[aa][dataset_type]["energy"] = energy_metrics


def generate_latex_table(data):
    latex = """\\begin{tabular}{|c|c|c|ccc|}
    \\hline
    \\multirow{2}{*}{Amino Acid} & \\multirow{2}{*}{\\shortstack{Training\\\\Dataset}} & \\multirow{2}{*}{\\shortstack{Reconstruction\\\\RMSD (\\r{A}) ↓}} & \\multicolumn{3}{c|}{MAE (kJ/mol) [Normalized MAE] ↓} \\\\
    \\cline{4-6}
     & &  & Torsion energy & LJ energy & Coulomb energy\\\\
    \\hline
"""

    def format_force(real_pair, synth_pair):
        real_val, real_norm = real_pair
        synth_val, synth_norm = synth_pair

        EPSILON = 1e-6  # threshold for treating values as equal

        # --- Raw MAE comparison ---
        if real_val != "N/A" and synth_val != "N/A":
            r_val = float(real_val)
            s_val = float(synth_val)

            if abs(r_val - s_val) < EPSILON:
                real_raw = f"\\textbf{{{real_val}}}"
                synth_raw = f"\\textbf{{{synth_val}}}"
            elif r_val <= s_val:
                real_raw = f"\\textbf{{{real_val}}}"
                synth_raw = synth_val
            else:
                real_raw = real_val
                synth_raw = f"\\textbf{{{synth_val}}}"
        else:
            real_raw, synth_raw = real_val, synth_val

        # --- Normalized MAE comparison ---
        if real_norm != "N/A" and synth_norm != "N/A":
            r_norm = float(real_norm)
            s_norm = float(synth_norm)

            if abs(r_norm - s_norm) < EPSILON:
                real_n = f"\\textbf{{{real_norm}}}"
                synth_n = f"\\textbf{{{synth_norm}}}"
            elif r_norm <= s_norm:
                real_n = f"\\textbf{{{real_norm}}}"
                synth_n = synth_norm
            else:
                real_n = real_norm
                synth_n = f"\\textbf{{{synth_norm}}}"
        else:
            real_n, synth_n = real_norm, synth_norm

        return f"${real_raw}\\,[{real_n}]$", f"${synth_raw}\\,[{synth_n}]$"

    for aa in ["ARG", "LYS", "MET"]:  # Maintain example order
        if aa not in data:
            continue
        datasets = data[aa]

        real_rmsd = (
            f"{datasets['real']['rmsd']:.3f}"
            if "real" in datasets and "rmsd" in datasets["real"]
            else "N/A"
        )
        synth_rmsd = (
            f"{datasets['synthetic']['rmsd']:.3f}"
            if "synthetic" in datasets and "rmsd" in datasets["synthetic"]
            else "N/A"
        )
        EPSILON = 1e-6  # Put this at the top-level so both RMSD and energy can use it

    # --- RMSD comparison ---
    if real_rmsd != "N/A" and synth_rmsd != "N/A":
        r_rmsd = float(real_rmsd)
        s_rmsd = float(synth_rmsd)

        if abs(r_rmsd - s_rmsd) < EPSILON:
            real_rmsd = f"\\textbf{{{real_rmsd}}}"
            synth_rmsd = f"\\textbf{{{synth_rmsd}}}"
        elif r_rmsd <= s_rmsd:
            real_rmsd = f"\\textbf{{{real_rmsd}}}"
        else:
            synth_rmsd = f"\\textbf{{{synth_rmsd}}}"

        real_energy = datasets["real"].get("energy", {})
        synth_energy = datasets["synthetic"].get("energy", {})

        # Format each force type
        torsion_real = real_energy.get("PeriodicTorsionForce", ("N/A", "N/A"))
        torsion_synth = synth_energy.get("PeriodicTorsionForce", ("N/A", "N/A"))
        lj_real = real_energy.get("LJForce", ("N/A", "N/A"))
        lj_synth = synth_energy.get("LJForce", ("N/A", "N/A"))
        coulomb_real = real_energy.get("CoulombForce", ("N/A", "N/A"))
        coulomb_synth = synth_energy.get("CoulombForce", ("N/A", "N/A"))

        t_real, t_synth = format_force(torsion_real, torsion_synth)
        lj_real_fmt, lj_synth_fmt = format_force(lj_real, lj_synth)
        c_real, c_synth = format_force(coulomb_real, coulomb_synth)

        # Add rows
        latex += f"    \\multirow{{2}}{{*}}{{{aa}}} & PDB      & ${real_rmsd}$ & {t_real} & {lj_real_fmt} & {c_real} \\\\\n"
        latex += f"                         & Synthetic & ${synth_rmsd}$ & {t_synth} & {lj_synth_fmt} & {c_synth} \\\\\n"
        latex += "    \\hline\n"

    latex += "\\end{tabular}"
    return latex


# Main execution
data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

# Update paths to your dataset folders
collect_data("results/full", "real", data)
collect_data("results/synth/high_energy", "synthetic", data)

# Generate and print LaTeX table
print(generate_latex_table(data))
