import os
import re

# List of model folders and their display names in the table
defined_models = ["HAE", "torsion", "mapping"]
display_names = {"HAE": "HAE", "torsion": "Torsion", "mapping": "Hybrid"}

# Amino acids and step sizes to include
amino_acids = ["ARG", "LYS", "MET"]
step_sizes = ["10", "1", "0.1", "0.001"]

# Base directories where results are stored per model
default_real_base = "results/full"
default_synth_base = "results/synth/high_energy"


# Parse the metrics file to extract clash percentage (num discarded *100) per step size
def parse_clash_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    # Split by learning rate blocks
    blocks = re.split(r"lr\s+", content)[1:]
    clashes = {}
    for block in blocks:
        lines = block.strip().splitlines()
        # Extract the numeric lr value from the block header
        header = lines[0].strip()
        match = re.match(r"([0-9.]+)", header)
        if not match:
            continue
        lr = match.group(1)
        # find discarded percent
        pct = None
        for line in lines:
            if line.startswith("num discarded:"):
                m = re.search(r"\[([0-9.]+)\]", line)
                if m:
                    pct = float(m.group(1)) * 100
                break
        if pct is not None:
            clashes[lr] = pct
    print(clashes)
    return clashes


# Generate LaTeX table for clash metrics only
def make_clash_latex(
    models, display_names, real_base, synth_base, amino_acids, step_sizes
):
    # Build column spec: |c|c| + one group per amino acid
    group_cols = len(step_sizes)
    col_spec = "|c|c|" + "".join(
        ["".join(["c" for _ in step_sizes]) + "|" for _ in amino_acids]
    )

    # Build header lines
    lines = []
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    # First header row: Dataset, Model, and amino acid spans
    first_cols = ["\\multirow{2}{*}{Dataset}", "\\multirow{2}{*}{Model}"]
    aa_spans = [f"\\multicolumn{{{group_cols}}}{{c|}}{{{aa}}}" for aa in amino_acids]
    lines.append("&".join(first_cols + aa_spans) + "\\\\")

    # Cline under amino acid labels (columns 3 to end)
    total_cols = 2 + group_cols * len(amino_acids)
    lines.append(f"\\cline{{3-{total_cols}}}")

    # Second header row: step size labels under each amino acid
    step_labels = [s for _ in amino_acids for s in step_sizes]
    lines.append("&&" + "&".join(step_labels) + "\\\\")
    lines.append("\\hline")

    # Table body for real and synthetic datasets
    for dataset_label, base_dir in [("PDB", real_base), ("Synth", synth_base)]:
        n_models = len(models)
        for idx, model in enumerate(models):
            disp = display_names.get(model, model)
            # Prefix dataset multirow only for the first model
            prefix = (
                f"\\multirow{{{n_models}}}{{*}}{{{dataset_label}}}&"
                if idx == 0
                else "&"
            )
            row = [prefix + disp]
            # Fill metrics for each amino acid and step size
            for aa in amino_acids:
                filepath = os.path.join(base_dir, model, f"sampling_metrics_{aa}.txt")
                if os.path.isfile(filepath):
                    clashes = parse_clash_file(filepath)
                else:
                    clashes = {}
                for lr in step_sizes:
                    val = clashes.get(lr)
                    row.append(f"{val:.2f}" if val is not None else "-")
            lines.append("&".join(row) + "\\\\")
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    table = make_clash_latex(
        defined_models,
        display_names,
        default_real_base,
        default_synth_base,
        amino_acids,
        step_sizes,
    )
    print(table)
