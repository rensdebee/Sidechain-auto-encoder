import glob
import os
import random
import shutil
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader

from amino.data.chi_atoms import chi_atoms
from amino.models.AutoEncoder import AutoencoderLightning
from amino.models.Decoder import DecoderLightning
from amino.models.EnergyPredictor import EnergyPredictionModel
from amino.models.MappingNetwork import MappingNetwork
from amino.models.utils import calculate_3D_squared_distance


def calculate_dihedral_angles(atom_positions):
    # Validate input shape
    if atom_positions.shape[1:] != (4, 3):
        raise ValueError("atom_positions must have shape (num_residues, 4, 3)")

    # Extract bond vectors
    b1 = atom_positions[:, 1] - atom_positions[:, 0]
    b2 = atom_positions[:, 2] - atom_positions[:, 1]
    b3 = atom_positions[:, 3] - atom_positions[:, 2]

    # Calculate normals to the planes
    n1 = torch.cross(b1, b2, dim=1)
    n2 = torch.cross(b2, b3, dim=1)

    # Normalize the normals (avoid division by zero)
    n1_norm = torch.norm(n1, dim=1, keepdim=True).clamp(min=1e-20)
    n2_norm = torch.norm(n2, dim=1, keepdim=True).clamp(min=1e-20)
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    # Normalize b2
    b2_norm = torch.norm(b2, dim=1, keepdim=True).clamp(min=1e-20)
    b2 = b2 / b2_norm

    # Compute the cosine and sine of the dihedral angles
    cos_phi = torch.sum(n1 * n2, dim=1)
    sin_phi = torch.sum(torch.cross(n1, n2, dim=1) * b2, dim=1)

    # Compute dihedral angles
    dihedral_angles = torch.atan2(sin_phi, cos_phi)

    return dihedral_angles


def extract_atom_order(pdb_file):
    residue_atom_order = defaultdict(list)
    with open(pdb_file, "r") as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                line = line.split()
                # Extracting sequence number, residue name, and atom name
                amino_acid_name = line[3].strip()
                atom_name = line[2].strip()
                residue_atom_order[amino_acid_name].append(atom_name)
    return residue_atom_order


def get_phi_psi(backbones):
    """
    backbones: (batch_size, n_atoms, 3) where n_atoms = num_residues * 4
    Returns:
        phi: (batch_size, num_residues) with NaN at [0]
        psi: (batch_size, num_residues) with NaN at [-1]
    """
    B, n_atoms, _ = backbones.shape
    assert n_atoms % 4 == 0, "Each residue must have exactly 4 atoms (N, CA, C, O)"

    num_residues = n_atoms // 4
    atom_idx = lambda res, offset: res * 4 + offset

    # Construct index tensors
    res_ids = torch.arange(num_residues, device=backbones.device)

    # Phi atoms: C(i-1), N(i), CA(i), C(i)
    phi_mask = res_ids >= 1
    phi_res_ids = res_ids[phi_mask]

    C_prev_idx = atom_idx(phi_res_ids - 1, 2)
    N_idx = atom_idx(phi_res_ids, 0)
    CA_idx = atom_idx(phi_res_ids, 1)
    C_idx = atom_idx(phi_res_ids, 2)

    # Psi atoms: N(i), CA(i), C(i), N(i+1)
    psi_mask = res_ids < num_residues - 1
    psi_res_ids = res_ids[psi_mask]

    N_idx_psi = atom_idx(psi_res_ids, 0)
    CA_idx_psi = atom_idx(psi_res_ids, 1)
    C_idx_psi = atom_idx(psi_res_ids, 2)
    N_next_idx = atom_idx(psi_res_ids + 1, 0)

    # Gather atoms for phi
    phi_atoms = torch.stack(
        [
            backbones[:, C_prev_idx],  # C(i-1)
            backbones[:, N_idx],  # N(i)
            backbones[:, CA_idx],  # CA(i)
            backbones[:, C_idx],  # C(i)
        ],
        dim=2,
    )  # shape: (B, len(phi_res_ids), 4, 3)
    phi_atoms = phi_atoms.reshape(((B * len(phi_res_ids)), 4, 3))
    # Gather atoms for psi
    psi_atoms = torch.stack(
        [
            backbones[:, N_idx_psi],  # N(i)
            backbones[:, CA_idx_psi],  # CA(i)
            backbones[:, C_idx_psi],  # C(i)
            backbones[:, N_next_idx],  # N(i+1)
        ],
        dim=2,
    )  # shape: (B, len(psi_res_ids), 4, 3)
    psi_atoms = psi_atoms.reshape(((B * len(psi_res_ids)), 4, 3))

    # Compute angles
    phi = torch.full((B, num_residues), float("nan"), device=backbones.device)
    psi = torch.full((B, num_residues), float("nan"), device=backbones.device)

    phi[:, phi_mask] = calculate_dihedral_angles(phi_atoms).reshape(
        (B, len(phi_res_ids))
    )
    psi[:, psi_mask] = calculate_dihedral_angles(psi_atoms).reshape(
        (B, len(psi_res_ids))
    )
    return phi, psi


def extract_atom_coords(pdb_file):
    coords = []
    with open(pdb_file, "r") as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                line = line.split()
                # Extracting sequence number, residue name, and atom name
                coords.append(list(map(float, line[6:9])))

    return torch.tensor(coords)


def read_pdb(pdb_file):
    coords = []
    aminos = []
    atoms = []
    with open(pdb_file, "r") as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                line = line.split()
                aminos.append(line[3].strip())
                atoms.append(line[2].strip())
                coords.append(list(map(float, line[6:9])))
    return aminos, atoms, torch.tensor(coords)


def write_pdb(aminos, atoms, coords, out, scale=1):
    backbone = False
    if isinstance(aminos, str):
        if aminos == "GLY":
            backbone = True
        aminos = [aminos] * len(atoms)
    chain = "A"
    amino_idx = 1
    if backbone:
        amino_idx = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, (amino, atom, coord) in enumerate(zip(aminos, atoms, coords)):
            prev_amino = aminos[i - 1] if i > 0 else None
            if (prev_amino and prev_amino != amino) or (backbone and (i % 4) == 0):
                amino_idx += 1
                f.write("\n")
            x, y, z = coord * scale

            # Create ATOM record
            line = (
                f"ATOM  {i+1:5d} {atom:4s} "
                f"{amino:3s} {chain:1s}"
                f"{amino_idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {atom[0]:>2s}\n"
            )
            f.write(line)
        f.write("\n")


def torsion_angles(amino_acid, data_matrix, atom_order):
    for i in range(1, 6):
        if amino_acid in chi_atoms[f"chi{i}"]:
            plane = chi_atoms[f"chi{i}"][amino_acid][:4]
            N = np.where(np.isin(atom_order, plane))[0]
            angles = calculate_dihedral_angles(data_matrix[:, N, :]).unsqueeze(1)

            if i == 1:
                torsion_angles = angles
            else:
                torsion_angles = torch.hstack([torsion_angles, angles])
    return torsion_angles


def torsion_atom_order(amino_acid, atom_order):
    order = []
    for i in range(1, 6):
        if amino_acid in chi_atoms[f"chi{i}"]:
            plane = chi_atoms[f"chi{i}"][amino_acid][:4]
            N = np.where(np.isin(atom_order, plane))[0]
            order.append((plane, N))
    return order


def sample_hypersphere(n, dim=3, dataset=None):
    points = torch.randn(n, dim)

    norms = torch.norm(points, p=2, dim=1, keepdim=True)
    points = points / norms

    return points


def sample_torsion_angles(n, dim=3, dataset=None, kdes=None):
    if dataset is not None:
        num = dataset.torsion_angles.shape[0]
        idx = random.sample(range(num), n)
        angles = dataset.torsion_angles[idx]
        return angles
    elif kdes is not None:
        angles = []
        for i, kde in enumerate(kdes):
            # Sample from KDE
            samples = kde.resample(size=n)[0]
            samples_wrapped = (samples + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
            angles.append(samples_wrapped)
        angles = np.array(angles)
        return torch.tensor(angles.T, dtype=torch.float)
    else:
        angles = torch.empty(n, dim).uniform_(-torch.pi, torch.pi)
        return angles


def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def apply_transform(coords, R, t):
    """Apply rotation and translation to coordinates"""
    return (R.mm(coords.T)).T + t


def structures_to_pdb(structures, atom_order, amino_acid, filename, scale=1):
    num_structures, num_atoms, _ = structures.shape
    chain = "A"
    with open(filename, "w", encoding="utf-8") as f:
        for structure_idx in range(num_structures):
            for atom_idx in range(num_atoms):
                x, y, z = structures[structure_idx, atom_idx] * scale
                element = atom_order[atom_idx]
                atom_name = element

                # Create ATOM record
                line = (
                    f"ATOM  {num_atoms*structure_idx + atom_idx+1:5d} {atom_name:4s} "
                    f"{amino_acid:3s} {chain:1s}"
                    f"{structure_idx+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00          {element[0]:>2s}\n"
                )
                f.write(line)
            f.write("\n")


def transform_pdb(pdb, R, t, out, scale=1):
    amino, atom, coords = read_pdb(pdb)
    transformed_coords = apply_transform(coords, R, t)
    write_pdb(amino, atom, transformed_coords, out, scale)
    return amino, atom, transformed_coords


def compute_torsion_rotations(
    coords, atom_sequence, torsion_atoms, angles=torch.linspace(0, 360, 360)
):
    """
    Rotate a torsion angle through 360 degrees and generate coordinates for each step.
    """
    # Find indices of torsion atoms
    try:
        i0 = atom_sequence.index(torsion_atoms[0])
        i1 = atom_sequence.index(torsion_atoms[1])
        i2 = atom_sequence.index(torsion_atoms[2])
        i3 = atom_sequence.index(torsion_atoms[3])
    except ValueError as e:
        raise ValueError("Torsion atoms not found in atom sequence") from e
    # Get axis points (B and C coordinates)
    B = coords[i1]
    C = coords[i2]
    axis_vector = C - B

    # Get indices of atoms to rotate (from fourth torsion atom onwards)
    rotate_index = i2 + 1
    if rotate_index != i3 and "H" not in atom_sequence[rotate_index]:
        rotate_index = i3
    rotating_indices = list(range(rotate_index, len(atom_sequence)))
    if "OXT" in atom_sequence:
        oxt_idx = atom_sequence.index("OXT")
        rotating_indices.remove(oxt_idx)
    if "O" in atom_sequence:
        o_idx = atom_sequence.index("O")
        if o_idx in rotating_indices:
            rotating_indices.remove(o_idx)
    if not rotating_indices:
        raise ValueError("No atoms to rotate - check torsion atom ordering")

    all_coords = []

    for angle in angles:
        # Convert to radians
        theta = torch.deg2rad(angle)

        # Rotate the relevant atoms
        rotated = rotate_points(points=coords[rotating_indices], B=B, C=C, theta=theta)

        # Create new coordinate tensor
        new_coords = coords.clone()
        new_coords[rotating_indices] = rotated
        all_coords.append(new_coords)

    return torch.stack(all_coords)


def rotate_points(points, B, C, theta):
    """
    Rotate points around the B-C axis by theta radians using Rodrigues' formula
    """
    axis_vector = C - B
    axis_norm = torch.norm(axis_vector)

    # Handle zero-length axis (shouldn't happen with valid input)
    if axis_norm < 1e-8:
        return points.clone()

    k = axis_vector / axis_norm
    translated = points - B

    # Compute rotation components
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    dot = torch.sum(translated * k, dim=1)
    cross = torch.cross(k.expand_as(translated), translated, dim=-1)

    # Apply Rodrigues' rotation formula
    rotated = (
        translated * cos_theta
        + cross * sin_theta
        + k * (1 - cos_theta) * dot.unsqueeze(1)
    )

    return rotated + B


def get_model(amino_acid, checkpoint_path="checkpoints", dim="*", return_type=False):
    if amino_acid is None:
        ckpt_path = checkpoint_path + f"/*[0-9].ckpt"
    else:
        ckpt_path = checkpoint_path + f"/*/{amino_acid}_dim_{dim}/**/" + "*[0-9].ckpt"
    low_rmsd = 1000
    # print(ckpt_path)
    # print(glob.glob(ckpt_path, recursive=True))
    for file in glob.glob(ckpt_path, recursive=True):
        rmsd = float(file.split("=")[-1][:-5])
        if rmsd < low_rmsd:
            low_rmsd = rmsd
            ckpt_file = file
    print(f"Using model with metric {low_rmsd} at path:\n{ckpt_file}")
    if "decoder" in ckpt_file:
        model = DecoderLightning.load_from_checkpoint(checkpoint_path=ckpt_file)
        type = "decoder"
    elif "energy" in ckpt_file and "torsion" in ckpt_file:
        model = EnergyPredictionModel.load_from_checkpoint(checkpoint_path=ckpt_file)
        type = "energy"
    elif "mapping" in ckpt_file:
        model = MappingNetwork.load_from_checkpoint(checkpoint_path=ckpt_file)
        type = "mapping"
    else:
        model = AutoencoderLightning.load_from_checkpoint(checkpoint_path=ckpt_file)
        type = "ae"
    if return_type:
        return model, type
    return model


def create_clean_path(path):
    os.makedirs(path, exist_ok=True)
    shutil.copy("pdbs/movie.py", path + "/")
    files = glob.glob(f"{path}/*.pdb")
    for f in files:
        if "template" not in f:
            os.remove(f)


def generate_latents(amino_acid, checkpoint_path, data_set, dim="*"):
    data = data_set
    data_loader = DataLoader(data, batch_size=8096, shuffle=False)

    ae, type = get_model(amino_acid, checkpoint_path, dim=dim, return_type=True)
    ae.eval()
    if type == "mapping":
        latent_dim = ae.HAE_dim
    else:
        latent_dim = ae.model.latent_dim
    latents = torch.empty((0, latent_dim))
    sd = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch["sidechain_position"].cuda()
            if type == "mapping":
                z, _ = ae(x)
            else:
                x_hat, z, (q_z, p_z), (mean, var) = ae(x.flatten(start_dim=1))
                if mean is not None:
                    z = mean
                x_hat = x_hat.unflatten(1, (-1, 3))
                sd.append(calculate_3D_squared_distance(x_hat, x))
            latents = torch.cat((latents, z.detach().cpu()))
    if type != "mapping":
        sd = torch.cat(sd)
        print(f"Trainings RMSD: {torch.sqrt(sd.mean())}")
    return latents


def build_structure(amino_acid, coords, atom_sequence, torsion_atom_sequence, angles):
    """
    Rotates side chains based on input torsion angles to build the 3D structure of an amino acid.

    Args:
        amino_acid (str): Name of the amino acid.
        coords (torch.Tensor): Initial atom coordinates of shape (N_atoms, 3).
        atom_sequence (List[str]): List of atom names corresponding to the coords.
        torsion_atom_sequence (List[Tuple[str, List[int]]]): List of torsion atom indices for each chi angle.
        angles (torch.Tensor): New torsion angles in radians of shape (n_angles,).

    Returns:
        torch.Tensor: Updated coordinates after torsion rotations (N_atoms, 3).
    """
    angles = list(angles.squeeze())
    original_angles = list(
        torsion_angles(amino_acid, coords.unsqueeze(0), atom_sequence).squeeze()
    )  # Calculate original torsion angles of structure

    for i, angle in enumerate(angles):
        # Get atom indices defining the torsion angle
        i0, i1, i2, i3 = torsion_atom_sequence[i][1]
        B = coords[i1]  # Axis start
        C = coords[i2]  # Axis end

        # Define which atoms to rotate
        rotate_index = i2 + 1
        if rotate_index != i3 and "H" not in atom_sequence[rotate_index]:
            rotate_index = i3
        rotating_indices = list(range(rotate_index, len(atom_sequence)))

        # Exclude terminal atoms
        if "OXT" in atom_sequence:
            oxt_idx = atom_sequence.index("OXT")
            rotating_indices.remove(oxt_idx)
        if "O" in atom_sequence:
            o_idx = atom_sequence.index("O")
            if o_idx in rotating_indices:
                rotating_indices.remove(o_idx)

        if not rotating_indices:
            raise ValueError("No atoms to rotate - check torsion atom ordering")

        assert abs(angle) <= torch.pi, "Angles should be in radian between -pi and pi"

        # Convert to relative rotation angle
        theta = angle - original_angles[i]

        # Rotate the selected atoms
        coords[rotating_indices] = rotate_points(
            points=coords[rotating_indices], B=B, C=C, theta=theta
        )

    return coords


def torsion_kdes(torsion_tensor):
    kdes = []
    torsion_tensor = torsion_tensor.numpy()
    num_torsion_angles = torsion_tensor.shape[1]
    for i in range(num_torsion_angles):
        data = torsion_tensor[:, i]

        # Fit KDE
        kde = gaussian_kde(data)
        kdes.append(kde)
    return kdes
