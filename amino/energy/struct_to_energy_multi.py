import argparse
import io
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

import numpy as np
import torch
from openmm import Vec3
from openmm.app import Element, ForceField, PDBFile, Topology
from tqdm import tqdm

from amino.data.datasets import SidechainDataset
from amino.energy.energy import pdb_to_energy
from amino.visualize.energy_dist import plot_energy_dist

# Worker-process global variables
_worker_globals = {
    "forcefield": ForceField("charmm36.xml"),
    "amino_acid": None,
    "atom_names": None,
    "platform": "CPU",
}


def init_worker(atom_names, amino_acid):
    """Initialize worker process with shared read-only resources"""
    if atom_names is not None:
        _worker_globals["elements"] = [
            Element.getBySymbol(name[0]) for name in atom_names
        ]
        _worker_globals["atom_names"] = atom_names
    if amino_acid is not None:
        _worker_globals["amino_acid"] = amino_acid


def build_topology(coord):
    coord = coord.numpy() if hasattr(coord, "numpy") else coord
    # Rebuild topology using worker globals
    topology = Topology()
    chain = topology.addChain()

    for i, (name, elem) in enumerate(
        zip(_worker_globals["atom_names"], _worker_globals["elements"])
    ):
        if (_worker_globals["amino_acid"] == "GLY" and i % 4 == 0) or i == 0:
            residue = topology.addResidue(_worker_globals["amino_acid"], chain)
        topology.addAtom(name, elem, residue)

    positions = [Vec3(*pos) for pos in coord]
    return topology, positions


def process_single_arg(arg):
    """Process one coordinate set with preserved index"""
    idx, coord, minimize_steps = arg
    file = False
    if type(coord) == str:
        file = True
    try:
        if not file:
            topology, positions = build_topology(coord)
            # PDB processing
            with io.StringIO() as pdb_io:
                PDBFile.writeFile(topology, positions, pdb_io)
                pdb_io.seek(0)
                energy = pdb_to_energy(
                    pdbfile=pdb_io,
                    forcefield=_worker_globals["forcefield"],
                    platform=_worker_globals["platform"],
                    minimize_steps=minimize_steps,
                    # write=[f"pdbs/backbones/", f"{idx}"],
                )
        else:
            energy = pdb_to_energy(
                pdbfile=coord,
                forcefield=_worker_globals["forcefield"],
                platform=_worker_globals["platform"],
                minimize_steps=minimize_steps,
                # write=[f"pdbs/weird_energy/{idx}", f"{ minimize_steps}"],
            )
        return idx, energy

    except Exception as e:
        print(f"Error processing index {idx}: {str(e)}")
        return idx, float("inf")


def process_chunks(args, queue, worker_id):
    chunk_results = []
    for arg in args:
        result = process_single_arg(arg)
        chunk_results.append(result)
        queue.put(("progress", worker_id))
    queue.put(("done", worker_id))
    return chunk_results


def calculate_energy(
    dataset, idxs=None, max_workers=None, minimize_steps=0, files=False
):
    """
    Main function with index-preserving parallel processing, returns energy given by order in idxs
    meaning if idxs is unsorted resutling energy is also unsorted

    """

    if not files:
        if idxs is None:
            idxs = list(range(len(dataset)))
        atom_names = dataset.atom_order
        args = [
            (i, dataset.sidechain_positions[idx].numpy(), minimize_steps)
            for i, idx in enumerate(idxs)
        ]
    else:
        assert idxs is None, "If working with files idxs should be none"
        args = [(i, f, minimize_steps) for i, f in enumerate(dataset)]

    num_workers = max_workers if max_workers is not None else os.cpu_count()
    num_chunks = min(num_workers, len(args))
    chunks = np.array_split(np.asarray(args, dtype="object"), num_chunks)
    chunks = [chunk.tolist() for chunk in chunks]  # Convert numpy chunks to lists
    print(f"Using: {num_workers} processes")
    with Manager() as manager:
        queue = manager.Queue()
        with tqdm(total=len(args), desc="Total progress") as main_pbar:
            # Start progress listener thread
            def listen_progress():
                active_workers = len(chunks)
                while active_workers > 0:
                    msg = queue.get()
                    if msg[0] == "progress":
                        main_pbar.update(1)
                    elif msg[0] == "done":
                        active_workers -= 1

            listener = threading.Thread(target=listen_progress)
            listener.start()
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=init_worker,
                initargs=(
                    (atom_names, dataset.amino_acid) if not files else (None, None)
                ),
            ) as executor:
                futures = [
                    executor.submit(process_chunks, chunk, queue, worker_id)
                    for worker_id, chunk in enumerate(chunks)
                ]
                results = []
                for future in as_completed(futures):
                    results.extend(future.result())

            # Cleanup
            listener.join()

    # Reorder results by original index
    results.sort(key=lambda x: x[0])
    energy = [energy for _, energy in results]
    energy_dict = {}
    for key in energy[0]:
        energy_dict[key] = []
        for d in energy:
            if type(d) is not dict:
                energy_dict[key].append(torch.inf)
            else:
                energy_dict[key].append(d[key])
    return energy_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run kmeans_energy for given amino acids."
    )

    # Add arguments
    parser.add_argument(
        "--amino_idx", type=int, default=-1, help="Index of amino to proces"
    )

    # Parse arguments
    args = parser.parse_args()

    # Extract arguments
    amino_idx = args.amino_idx

    # Run the kmeans_energy function for each amino acid
    aminos = ["ARG", "LYS", "MET", "GLU", "GLN", "GLY"]
    amino = aminos[amino_idx]
    print(amino)

    data_path = "dataset/synth/data"
    dataset = SidechainDataset(amino, data_path, fixed_O=True)
    energies = calculate_energy(dataset, max_workers=None)
    tensor = torch.tensor(list(energies.values()), dtype=torch.float32).T
    key_to_col = {key: idx for idx, key in enumerate(energies.keys())}
    torch.save((key_to_col, tensor), f"dataset/synth/energy/{amino}.pt")

    energy_path = "dataset/synth/energy"
    key_to_col, energy = torch.load(f"{energy_path}/{amino}.pt")
    labels = [None]
    fig = plot_energy_dist(
        key_to_col,
        [energy],
        labels,
        iqr=True,
    )
    fig.suptitle(f"{amino} Energy Distribution Synthetic Dataset", fontsize=14)
    fig.savefig(f"results/synth/dataset/energy_dist_{amino}.pdf", bbox_inches="tight")
