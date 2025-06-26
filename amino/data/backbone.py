import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from tqdm import tqdm

three_to_one = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def extract_batch(batch_paths):
    grouped = defaultdict(list)  # K → [tensor]
    sequences = defaultdict(list)  # K → [sequence strings]

    for pdb_path in batch_paths:
        try:
            coords = []
            seq = []
            current_res_id = None
            current_atoms = {}
            current_res_name = None

            with open(pdb_path, "r") as f:
                for line in f:
                    if not line.startswith("ATOM"):
                        continue
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    res_id = line[22:26].strip() + line[26].strip()
                    chain_id = line[21].strip()
                    if chain_id != "P":
                        continue
                    if atom_name not in ["N", "CA", "C", "O"]:
                        continue

                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    if res_id != current_res_id:
                        if current_atoms and all(
                            k in current_atoms for k in ["N", "CA", "C", "O"]
                        ):
                            coords.extend(
                                [
                                    current_atoms["N"],
                                    current_atoms["CA"],
                                    current_atoms["C"],
                                    current_atoms["O"],
                                ]
                            )
                            seq.append(three_to_one.get(current_res_name, "X"))
                        current_atoms = {}
                        current_res_id = res_id
                        current_res_name = res_name

                    current_atoms[atom_name] = [x, y, z]

            if current_atoms and all(k in current_atoms for k in ["N", "CA", "C", "O"]):
                coords.extend(
                    [
                        current_atoms["N"],
                        current_atoms["CA"],
                        current_atoms["C"],
                        current_atoms["O"],
                    ]
                )
                seq.append(three_to_one.get(current_res_name, "X"))

            if coords:
                tensor = torch.tensor(coords, dtype=torch.float32)
                K = tensor.shape[0] // 4
                grouped[K].append(tensor)
                sequences[K].append("".join(seq))
        except:
            continue

    return grouped, sequences


def process_all_pdbs_batched(folder_path, output_dir, max_workers=8, batch_size=100):
    from pathlib import Path

    pdb_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".pdb")
    ]
    batches = [
        pdb_files[i : i + batch_size] for i in range(0, len(pdb_files), batch_size)
    ]
    all_grouped = defaultdict(list)
    all_seqs = defaultdict(list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_batch, batch) for batch in batches]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing batches"
        ):
            result = future.result()
            grouped, seqs = result
            for K in grouped:
                all_grouped[K].extend(grouped[K])
                all_seqs[K].extend(seqs[K])

    os.makedirs(output_dir, exist_ok=True)

    for K in all_grouped:
        torch.save(
            torch.stack(all_grouped[K]), os.path.join(output_dir, f"peptides_len{K}.pt")
        )

        seq_path = Path(output_dir) / f"peptides_len{K}.txt"
        with open(seq_path, "w") as f:
            f.write("\n".join(all_seqs[K]))

        print(
            f"Saved: {len(all_grouped[K])} peptides_len{K}.pt and peptides_len{K}.txt"
        )


if __name__ == "__main__":
    process_all_pdbs_batched("dataset/pdb", "dataset/backbone_tensors", max_workers=8)
