import os
import re


def parse_rmsds(folder):
    ae_path = os.path.join(folder, "ae_metrics.txt")
    decoder_path = os.path.join(folder, "decoder_metrics.txt")

    def extract_rmsds(filepath):
        with open(filepath, "r") as f:
            lines = f.readlines()

        data = {}
        aa = None
        mode = None
        for line in lines:
            line = line.strip()
            if line.startswith("Amino:"):
                aa = line.split(":")[1].strip()
                data[aa] = {"Dataset": None, "Random": None}
            elif "Dataset latents" in line:
                mode = "Dataset"
            elif "Random latents" in line:
                mode = "Random"
            elif line.startswith("rmsd:") and aa:
                val = float(re.findall(r"[\d.]+", line)[0])
                data[aa][mode] = val
        return data

    ae_data = extract_rmsds(ae_path)
    dec_data = extract_rmsds(decoder_path)

    combined = {}
    for aa in ae_data.keys():
        combined[aa] = {"real": {"first": ae_data[aa], "second": dec_data[aa]}}
    return combined


# === INPUT YOUR FOLDERS HERE ===
real_folder = "results/full/HAE"
synthetic_folder = "results/synth/high_energy/HAE"  # Replace if needed

real_data = parse_rmsds(real_folder)
synthetic_data = parse_rmsds(synthetic_folder)

# Merge real and synthetic
amino_acids = sorted(real_data.keys())

# Collect all values for bolding logic
table_rows = []

for aa in amino_acids:
    vals = {
        "real_first_dataset": real_data[aa]["real"]["first"]["Dataset"],
        "real_second_dataset": real_data[aa]["real"]["second"]["Dataset"],
        "synth_first_dataset": synthetic_data[aa]["real"]["first"]["Dataset"],
        "synth_second_dataset": synthetic_data[aa]["real"]["second"]["Dataset"],
        "real_first_random": real_data[aa]["real"]["first"]["Random"],
        "real_second_random": real_data[aa]["real"]["second"]["Random"],
        "synth_first_random": synthetic_data[aa]["real"]["first"]["Random"],
        "synth_second_random": synthetic_data[aa]["real"]["second"]["Random"],
    }

    # Find bold targets
    dataset_values = [
        vals["real_first_dataset"],
        vals["real_second_dataset"],
        vals["synth_first_dataset"],
        vals["synth_second_dataset"],
    ]
    random_values = [
        vals["real_first_random"],
        vals["real_second_random"],
        vals["synth_first_random"],
        vals["synth_second_random"],
    ]

    min_dataset = min(dataset_values)
    min_random = min(random_values)

    def fmt(v, is_dataset):
        bold = (v == min_dataset) if is_dataset else (v == min_random)
        return f"$\\textbf{{{v:.3f}}}$" if bold else f"${v:.3f}$"

    real_row = f"""\\multirow{{2}}{{*}}{{{aa}}} & PDB      & {fmt(vals["real_first_dataset"], True)} & {fmt(vals["real_second_dataset"], True)} & {fmt(vals["real_first_random"], False)} & {fmt(vals["real_second_random"], False)} \\\\"""
    synth_row = f"""                     & Synthetic & {fmt(vals["synth_first_dataset"], True)} & {fmt(vals["synth_second_dataset"], True)} & {fmt(vals["synth_first_random"], False)} & {fmt(vals["synth_second_random"], False)} \\\\"""

    table_rows.append(real_row)
    table_rows.append(synth_row)
    table_rows.append("\\hline")

# Print LaTeX table
print("\\begin{tabular}{|c|c|cc|cc|}")
print("\\hline")
print(
    "\\multirow{2}{*}{Amino Acid} & \\multirow{2}{*}{\\shortstack{Training\\\\Dataset}} & \\multicolumn{2}{c|}{Dataset latents RMSD (\\r{A}) ↓} & \\multicolumn{2}{c|}{Random latents RMSD (\\r{A}) ↓} \\\\"
)
print("\\cline{3-6}")
print(" & & First decoder & Second decoder & First decoder & Second decoder\\\\")
print("\\hline")

for row in table_rows:
    print(row)

print("\\end{tabular}")
