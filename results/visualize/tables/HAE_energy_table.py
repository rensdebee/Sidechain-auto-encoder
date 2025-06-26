import os
import re


def parse_energy_metrics(decoder_path):
    with open(decoder_path, "r") as f:
        lines = f.readlines()

    data = {}
    aa = None
    mode = None

    for line in lines:
        line = line.strip()
        if line.startswith("Amino:"):
            aa = line.split(":")[1].strip()
            data[aa] = {"Dataset": {}, "Random": {}}
        elif line.startswith("Dataset latents"):
            mode = "Dataset"
        elif line.startswith("Random latents"):
            mode = "Random"
        elif any(k in line for k in ["TorsionForce", "LJForce", "CoulombForce"]):
            match = re.match(r"([\w_]+):\s*([\d.]+)", line)
            if match:
                key, val = match.groups()
                data[aa][mode][key] = float(val)
    return data


def make_latex_table(data_real, data_synth, mode_label):
    rows = []

    amino_acids = sorted(data_real.keys())
    for aa in amino_acids:
        real = data_real[aa][mode_label]
        synth = data_synth[aa][mode_label]

        def format_pair(real_val, real_norm, synth_val, synth_norm):
            EPSILON = 1e-6  # Threshold to treat values as "similar"

            # Raw MAE
            if abs(real_val - synth_val) < EPSILON:
                real_raw = f"\\textbf{{{real_val:.3f}}}"
                synth_raw = f"\\textbf{{{synth_val:.3f}}}"
            elif real_val <= synth_val:
                real_raw = f"\\textbf{{{real_val:.3f}}}"
                synth_raw = f"{synth_val:.3f}"
            else:
                real_raw = f"{real_val:.3f}"
                synth_raw = f"\\textbf{{{synth_val:.3f}}}"

            # Normalized MAE
            if abs(real_norm - synth_norm) < EPSILON:
                real_n = f"\\textbf{{{real_norm:.3f}}}"
                synth_n = f"\\textbf{{{synth_norm:.3f}}}"
            elif real_norm <= synth_norm:
                real_n = f"\\textbf{{{real_norm:.3f}}}"
                synth_n = f"{synth_norm:.3f}"
            else:
                real_n = f"{real_norm:.3f}"
                synth_n = f"\\textbf{{{synth_norm:.3f}}}"

            # Combine into LaTeX string (brackets not bolded)
            real_str = f"${real_raw}\\,[{real_n}]$"
            synth_str = f"${synth_raw}\\,[{synth_n}]$"
            return real_str, synth_str

        t_real, t_synth = format_pair(
            real["PeriodicTorsionForce_mae"],
            real["PeriodicTorsionForce_norm_mae"],
            synth["PeriodicTorsionForce_mae"],
            synth["PeriodicTorsionForce_norm_mae"],
        )
        lj_real, lj_synth = format_pair(
            real["LJForce_mae"],
            real["LJForce_norm_mae"],
            synth["LJForce_mae"],
            synth["LJForce_norm_mae"],
        )
        c_real, c_synth = format_pair(
            real["CoulombForce_mae"],
            real["CoulombForce_norm_mae"],
            synth["CoulombForce_mae"],
            synth["CoulombForce_norm_mae"],
        )

        rows.append(
            f"\\multirow{{2}}{{*}}{{{aa}}} & PDB      & {t_real} & {lj_real}  & {c_real} \\\\"
        )
        rows.append(
            f"                          & Synthetic & {t_synth} & {lj_synth}  & {c_synth} \\\\"
        )
        rows.append("\\hline")

    # Table header
    table = []
    table.append("\\begin{tabular}{|c|c|ccc|}")
    table.append("\\hline")
    table.append(
        "\\multirow{2}{*}{Amino Acid} & \\multirow{2}{*}{\\shortstack{Training\\\\Dataset}} & \\multicolumn{3}{c|}{Energy MAE (kJ/mol) [Normalized MAE] ↓} \\\\"
    )
    table.append("\\cline{3-5}")
    table.append(
        "                          &                          & Torsion energy & LJ energy & Coulomb energy \\\\"
    )
    table.append("\\hline")
    table.extend(rows)
    table.append("\\end{tabular}")

    return "\n".join(table)


# === Set your file paths ===
real_path = "results/full/HAE/decoder_metrics.txt"
synth_path = "results/synth/high_energy/HAE/decoder_metrics.txt"  # Replace if needed

real_data = parse_energy_metrics(real_path)
synth_data = parse_energy_metrics(synth_path)

# === Generate tables ===
print("%% Dataset Latents Table")
print(make_latex_table(real_data, synth_data, "Dataset"))

print("\n%% Random Latents Table")
print(make_latex_table(real_data, synth_data, "Random"))
