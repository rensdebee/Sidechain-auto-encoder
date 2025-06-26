import os
import re
from collections import defaultdict


def parse_file_content(content):
    metrics, params = {}, {}
    aa_match = re.search(r"'amino_acid': '([A-Z]{3})'", content)
    if aa_match:
        params["amino_acid"] = aa_match.group(1)
    current = None
    for line in content.splitlines():
        if "val_" in line and ":" in line:
            name = line.split(":")[0].strip()
            metrics[name] = {}
            current = name
        elif current:
            if "Mean:" in line:
                metrics[current]["mean"] = float(
                    re.search(r"Mean:\s*([0-9.]+)", line).group(1)
                )
            elif "Std:" in line:
                metrics[current]["std"] = float(
                    re.search(r"Std:\s*([0-9.]+)", line).group(1)
                )
    return params, metrics


def collect_data(path, dtype, store):
    for root, _, files in os.walk(path):
        for fn in files:
            if not fn.endswith(".txt"):
                continue
            params, metrics = parse_file_content(open(os.path.join(root, fn)).read())
            if "amino_acid" not in params:
                continue
            aa = params["amino_acid"]
            d = store[aa][dtype]
            if "val_rmsd_epoch" in metrics:
                d["rmsd"] = metrics["val_rmsd_epoch"]["mean"]
            if "val_mean_angular_error_deg" in metrics:
                d["angular"] = metrics["val_mean_angular_error_deg"]["mean"]
            em = {}
            for force in ["PeriodicTorsionForce", "LJForce", "CoulombForce"]:
                k1, k2 = f"val_{force}_mae_epoch", f"val_{force}_norm_mae_epoch"
                if k1 in metrics and k2 in metrics:
                    em[force] = (metrics[k1]["mean"], metrics[k2]["mean"])
            if em:
                d["energy"] = em


EPSILON = 1e-6  # Add this near the top of the file


def generate_latex(data):
    forces = ["PeriodicTorsionForce", "LJForce", "CoulombForce"]
    lbls = {
        "PeriodicTorsionForce": "Torsion energy",
        "LJForce": "LJ energy",
        "CoulombForce": "Coulomb energy",
    }
    header = r"""
\begin{tabular}{|c|c|c|c|ccc|}
\hline
\multirow{2}{*}{Amino Acid} & \multirow{2}{*}{\shortstack{Training\\Dataset}} & \multirow{2}{*}{\shortstack{Reconstruction\\RMSD (\r{A}) ↓}} & \multirow{2}{*}{\shortstack{Mean Angular\\Error (deg) ↓}} & \multicolumn{3}{c|}{MAE (kJ/mol) [Normalized MAE] ↓} \\
\cline{5-7}
 & & &  & Torsion energy & LJ energy & Coulomb energy\\
\hline
"""
    out = [header]
    for aa in sorted(data):
        real, synth = data[aa]["real"], data[aa]["synthetic"]
        # RMSD / Angular formatting
        rr, sr = real.get("rmsd"), synth.get("rmsd")
        ra, sa = real.get("angular"), synth.get("angular")

        def format_pair(x, y):
            if x is None or y is None:
                return "N/A", "N/A"
            if abs(x - y) < EPSILON:
                return f"\\textbf{{{x:.3f}}}", f"\\textbf{{{y:.3f}}}"
            elif x <= y:
                return f"\\textbf{{{x:.3f}}}", f"{y:.3f}"
            else:
                return f"{x:.3f}", f"\\textbf{{{y:.3f}}}"

        rrs, srs = format_pair(rr, sr)
        ras, sas = format_pair(ra, sa)

        # energy bold flags
        low = {}
        for f in forces:
            rv, rn = real.get("energy", {}).get(f, (None, None))
            sv, sn = synth.get("energy", {}).get(f, (None, None))
            if rv is None or rn is None or sv is None or sn is None:
                low[f] = None
            else:
                close_v = abs(rv - sv) < EPSILON
                close_n = abs(rn - sn) < EPSILON
                low[f] = {
                    "real_bold_val": close_v or rv <= sv,
                    "synth_bold_val": close_v or sv < rv,
                    "real_bold_norm": close_n or rn <= sn,
                    "synth_bold_norm": close_n or sn < rn,
                }

        def cell(f, is_real):
            tup = real["energy"][f] if is_real else synth["energy"][f]
            v, n = tup
            flags = low[f]
            if flags is None:
                return "N/A"
            if is_real:
                vs = f"\\textbf{{{v:.3f}}}" if flags["real_bold_val"] else f"{v:.3f}"
                ns = f"\\textbf{{{n:.3f}}}" if flags["real_bold_norm"] else f"{n:.3f}"
            else:
                vs = f"\\textbf{{{v:.3f}}}" if flags["synth_bold_val"] else f"{v:.3f}"
                ns = f"\\textbf{{{n:.3f}}}" if flags["synth_bold_norm"] else f"{n:.3f}"
            return f"{vs}\\,[{ns}]"

        # rows
        out.append(
            f"    \multirow{{2}}{{*}}{{{aa}}} & PDB       & $ {rrs} $ & $ {ras} $"
            + "".join([f" & $ {cell(f,True)} $" for f in forces])
            + " \\\\ \n"
        )
        out.append(
            f"                         & Synthetic & $ {srs} $ & $ {sas} $"
            + "".join([f" & $ {cell(f,False)} $" for f in forces])
            + " \\\\ \n    \hline\n"
        )
    out.append("\\end{tabular}")
    return "".join(out)


# Run
store = defaultdict(lambda: {"real": {}, "synthetic": {}})
collect_data("results/full/mapping_encoder", "real", store)
collect_data("results/synth/high_energy/mapping_encoder", "synthetic", store)
print(generate_latex(store))
