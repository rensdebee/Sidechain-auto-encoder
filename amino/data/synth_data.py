import itertools
from collections import defaultdict
from io import StringIO

import numpy as np
import torch
from openmm import Vec3
from openmm.app import Element, PDBFile, Topology
from pymol import cmd
from tqdm import tqdm

from amino.clustering.kmeans import plot_angle_distributions
from amino.data.clean_dataset import compute_distance
from amino.data.datasets import SidechainDataset
from amino.utils.utils import build_structure, create_clean_path, write_pdb


def check_structure(structure, amino_acid, atom_order):
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
    if nb != topology.getNumBonds():
        return False

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
    discard = False
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = compute_distance(positions[i], positions[j])
            if frozenset({i, j}) not in bonded_pairs and distance < 2.0:
                return False
            if frozenset({i, j}) in bonded_pairs and distance > 3.0:
                return False

    return True


def create_synth_data(amino_acid, num_pertubed_samples=500000):
    real_data = SidechainDataset(amino_acid)
    atom_order = real_data.atom_order
    torsion_orders = real_data.torsion_atom_order
    num_torsion = real_data.num_angles
    real_samples = real_data.sidechain_positions

    pertubed_samples = torch.empty(
        0, real_samples.shape[1], real_samples.shape[2], dtype=real_samples.dtype
    )
    real_idxs = np.random.choice(
        range(real_samples.shape[0]), size=num_pertubed_samples, replace=False
    )
    for idx in tqdm(real_idxs):
        to_pertube = real_samples[idx]
        while True:
            random_angles = np.random.uniform(-np.pi, np.pi, size=(1, num_torsion))
            pertubed = build_structure(
                amino_acid,
                to_pertube,
                atom_order,
                torsion_orders,
                angles=torch.tensor(random_angles),
            )
            if check_structure(pertubed, amino_acid, real_data.atom_order):
                break
        pertubed_samples = torch.cat((pertubed_samples, pertubed.unsqueeze(0)))

    print(pertubed_samples.shape)
    create_clean_path("dataset/synth/data")
    torch.save(pertubed_samples, f"dataset/synth/data/{amino_acid}.pt")

    synth_data = SidechainDataset(amino_acid, data_path=f"dataset/synth/data")
    fig, axes = plot_angle_distributions(
        synth_data.torsion_angles,
        labels=[
            "Synthetic dataset",
        ],
        ncols=2,
    )
    fig.suptitle(
        f"{amino_acid} Torsion Angle Distributions Synthetic Dataset",
        y=1.02,
        fontsize=14,
    )
    fig.savefig(
        f"results/synth/dataset/angle_dist_{amino_acid}.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    aminos = ["ARG", "GLN", "GLU", "LYS", "MET"]
    aminos = ["ARG", "LYS", "MET"]
    for amino in aminos:
        print(amino)
        num_pertubed_samples = 500000
        create_synth_data(
            amino,
            num_pertubed_samples=num_pertubed_samples,
        )
    for amino in aminos:
        data = SidechainDataset(amino, data_path=f"dataset/synth/data", fixed_O=True)
        sin_vals = torch.sin(data.torsion_angles)
        cos_vals = torch.cos(data.torsion_angles)
        sin_3vals = torch.sin(3 * data.torsion_angles)
        cos_3vals = torch.cos(3 * data.torsion_angles)

        latents = torch.cat((sin_vals, cos_vals, sin_3vals, cos_3vals), dim=-1)
        torch.save(latents, f"dataset/synth/torsion_latents_dim2/{amino}.pt")

        latents = torch.cat((sin_vals, cos_vals), dim=-1)
        torch.save(latents, f"dataset/synth/torsion_latents_dim1/{amino}.pt")

        struct = data.sidechain_positions[-500:]
        print(struct.shape)
        for i, struc in enumerate(struct):
            write_pdb(
                amino,
                data.atom_order,
                struc,
                out=f"pdbs/synth_data/{amino}/{i}.pdb",
                scale=1,
            )
