import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO

import numpy as np
import torch
from openmm import Vec3
from openmm.app import Element, PDBFile, Topology
from pymol import cmd
from tqdm import tqdm

from amino.data.datasets import SidechainDataset
from amino.utils.utils import create_clean_path


def compute_distance(pos1, pos2):
    dx = pos1.x - pos2.x
    dy = pos1.y - pos2.y
    dz = pos1.z - pos2.z
    return math.sqrt(dx**2 + dy**2 + dz**2)


def clean_structures(
    structures, amino_acid, atom_order, offset=0, return_metrics=False
):
    num_bonds = Counter()
    discarded_idxs = []
    num_clashes_bonded = 0
    num_clashes_unbonded = 0
    num_wrong_bonds = 0
    for idx, structure in enumerate(tqdm(structures)):
        structure = structure.numpy() if hasattr(structure, "numpy") else structure

        # Rebuild topology using worker globals
        topology = Topology()
        chain = topology.addChain()
        residue = topology.addResidue(amino_acid, chain)

        _ = [
            topology.addAtom(name, elem, residue)
            for name, elem in zip(
                atom_order,
                [Element.getBySymbol(name[0]) for name in atom_order],
            )
        ]

        positions = [Vec3(*pos) for pos in structure]
        topology.createStandardBonds()
        topology.createDisulfideBonds(positions)

        pdb_string = StringIO()
        PDBFile.writeFile(topology, positions, pdb_string)
        pdb_string.seek(0)  # Reset buffer position
        cmd.delete("all")
        cmd.read_pdbstr(pdb_string.getvalue(), "molecule")
        nb = len(cmd.get_model("molecule").bond)
        num_bonds.update([nb])

        # Compute bonded pairs
        bonded_pairs = set()
        neighbors = defaultdict(list)
        for bond in topology.bonds():
            a1, a2 = bond.atom1, bond.atom2
            bonded_pairs.add(frozenset({a1.index, a2.index}))
            neighbors[a1].append(a2)
            neighbors[a2].append(a1)

        # Calculate clashes
        num_atoms = len(positions)
        add_bonded = False
        add_unbonded = False
        discard = False
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                distance = compute_distance(positions[i], positions[j])
                if frozenset({i, j}) not in bonded_pairs and distance < 2.0:
                    discard = True
                    if not add_unbonded:
                        num_clashes_unbonded += 1
                        add_unbonded = True
                if frozenset({i, j}) in bonded_pairs and distance > 3.0:
                    discard = True
                    if not add_bonded:
                        num_clashes_bonded += 1
                        add_bonded = True
        if nb != topology.getNumBonds():
            discard = True
            num_wrong_bonds += 1
        if discard:
            discarded_idxs.append(idx + offset)
    if return_metrics:
        return discarded_idxs, [
            len(discarded_idxs),
            len(structures),
            num_wrong_bonds,
            num_clashes_bonded,
            num_clashes_unbonded,
        ]
    return discarded_idxs


def clean_dataset(structures, dataset, workers=8):
    discarded_idxs = []

    def chunker(seq, num_chunks):
        chunk_size = math.ceil(len(seq) / num_chunks)
        return ((seq[i : i + chunk_size], i) for i in range(0, len(seq), chunk_size))

    chunks = list(chunker(structures, workers))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all chunks at once
        futures = {
            executor.submit(
                clean_structures, chunk, dataset.amino_acid, dataset.atom_order, offset
            ): chunk
            for chunk, offset in chunks
        }

        for future in as_completed(futures):
            try:
                chunk_discarded_idxs = future.result()
                discarded_idxs.extend(chunk_discarded_idxs)
            except Exception as e:
                print(f"Error in chunk: {e}")

    return discarded_idxs


if __name__ == "__main__":
    aminos = ["ARG", "GLU", "GLN", "LYS", "MET"]

    output_lines = []
    create_clean_path("zzz")
    for amino in aminos:
        dataset = SidechainDataset(
            amino,
            data_path="./dataset/dirty",
            fixed_O=True,
        )
        to_remove = clean_dataset(dataset.sidechain_positions, dataset)
        mask = torch.ones(len(dataset), dtype=torch.bool)
        mask[to_remove] = False
        cleaned = dataset.sidechain_positions[mask]
        torch.save(cleaned, f"dataset/clean_10/{amino}.pt")
        print(len(to_remove))
