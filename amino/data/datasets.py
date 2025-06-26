import os

import torch
from torch.utils.data import Dataset

if torch.cuda.is_available():
    from torch_kdtree import build_kd_tree

from amino.utils.utils import (
    extract_atom_order,
    get_phi_psi,
    sample_hypersphere,
    torsion_angles,
    torsion_atom_order,
)


class SidechainDataset(Dataset):
    """
    Dataset class to train an sidechain autoencoder each sample represents
    an conformation of the amino acid sidechain with corresponding torsion angles
    and information about the atom structure
    """

    def __init__(
        self,
        amino_acid="ARG",
        data_path="./dataset/clean",
        fixed_O=False,
        energy_path=None,
        force_types=None,
        iqr_filter_energy=False,
        normalize_energy=False,
        inf_filter=False,
    ):
        self.amino_acid = amino_acid.upper()
        # Load data tensor from path or if data_tensor is passed as arg
        data_file = os.path.join(data_path, f"{self.amino_acid}.pt")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found!")
        self.sidechain_positions = torch.load(data_file, weights_only=True).cpu()

        # Set O atom to median O atom of all examples
        if fixed_O:
            o_ref = torch.median(self.sidechain_positions[:, 4, :], dim=0).values
            self.sidechain_positions[:, 4, :] = o_ref

        # Get order of atoms
        atom_order_data = extract_atom_order(os.path.join("dataset", "template.pdb"))
        if self.amino_acid not in atom_order_data:
            raise KeyError(f"Amino acid {self.amino_acid} not found in template.")
        self.atom_order = atom_order_data[self.amino_acid]
        self.num_atoms = len(self.atom_order)
        assert (
            self.num_atoms == self.sidechain_positions.shape[1]
        ), f"Number of atoms in dataset {self.sidechain_positions.shape[1]} doesn't match template {self.num_atoms}"

        # Calculate all torsion angles
        if self.amino_acid != "GLY":
            self.torsion_atom_order = torsion_atom_order(
                self.amino_acid, self.atom_order
            )
            self.torsion_angles = torsion_angles(
                self.amino_acid, self.sidechain_positions, self.atom_order
            )
            self.num_angles = self.torsion_angles.shape[1]
            assert self.num_angles == len(
                self.torsion_atom_order
            ), f"Number of side-chain torsion angles in dataset {len(self.torsion_atom_order)} doesn't match template {self.num_angles}"
        else:
            # Backbone
            phi, psi = get_phi_psi(self.sidechain_positions)
            phi_trimmed = phi[:, 1:]  # shape (B, R-1)
            psi_trimmed = psi[:, :-1]  # shape (B, R-1)

            # Interleave φ and ψ along last dimension
            interleaved = torch.stack((phi_trimmed, psi_trimmed), dim=2)  # (B, R-1, 2)
            self.torsion_angles = interleaved.reshape(phi.shape[0], -1)  # (B, (R-1)*2)
            self.num_angles = self.torsion_angles.shape[1]
            self.torsion_atom_order = [torch.nan]
        assert (
            self.torsion_angles.shape[0] == self.sidechain_positions.shape[0]
        ), f"Number of samples in dataset {self.sidechain_positions.shape} doesn't match torsion angles {self.torsion_angles.shape}"

        self.mean, self.std = None, None
        self.energy = None
        if energy_path is not None:
            data_file = os.path.join(energy_path, f"{self.amino_acid}.pt")
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Data file {data_file} not found!")
            key_to_col, energy = torch.load(data_file, weights_only=True)
            if force_types is not None:
                cols = []
                for force in force_types:
                    cols.append(key_to_col[force])
                energy = energy[:, cols]
            else:
                force_types = list(key_to_col.keys())

            if iqr_filter_energy:
                energy = iqr_filtering_energy(energy)

            if inf_filter:
                mask = inf_filter_energy(energy)
                energy = energy[mask]
                self.torsion_angles = self.torsion_angles[mask]
                self.sidechain_positions = self.sidechain_positions[mask]

            if normalize_energy:
                energy, self.mean, self.std = normalizing_energy(energy)

            self.energy = energy
            self.force_types = force_types
            self.energy_dim = self.energy.shape[1]
            assert (
                self.energy.shape[0] == self.sidechain_positions.shape[0]
            ), f"Number of samples in dataset {self.sidechain_positions.shape} doesn't match number of energy {self.energy.shape}"

    def __len__(self):
        return len(self.sidechain_positions)

    def __getitem__(self, idx):
        item = {
            "sidechain_position": self.sidechain_positions[idx],
            "amino_acid": self.amino_acid,
            "torsion_angles": self.torsion_angles[idx],
            "atom_order": self.atom_order,
            "torsion_atom_order": self.torsion_atom_order,
        }
        if self.energy is not None:
            item["energy"] = self.energy[idx]
        return item


class LatentDataset(Dataset):
    """
    Latent dataset class each sample represents a latent and correspond
    3D x,y,z sidechain conformation. The class also samples random latents
    with their closet real latent conformation
    """

    def __init__(
        self,
        amino_acid="ARG",
        data_path="dataset/clean",
        latent_path="dataset/latents",
        energy_path=None,
        force_types=None,
        iqr_filter_energy=False,
        interpolate_energy=False,
        normalize_energy=False,
        inf_filter=False,
        fixed_O=True,
        increase_factor=0.5,
        border_pct=0,
    ):
        self.amino_acid = amino_acid.upper()
        self.border_pct = border_pct
        self.increase_factor = increase_factor
        self.init = True
        self._torch_kdtree = None
        self.force_types = force_types
        self.num_reloads = 0
        data_file = os.path.join(data_path, f"{self.amino_acid}.pt")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found!")
        self.sidechain_positions = torch.load(data_file, weights_only=True).cpu()

        if fixed_O:
            o_ref = torch.median(self.sidechain_positions[:, 4, :], dim=0).values
            self.sidechain_positions[:, 4, :] = o_ref

        atom_order_data = extract_atom_order(os.path.join("dataset", "template.pdb"))
        if self.amino_acid not in atom_order_data:
            raise KeyError(f"Amino acid {self.amino_acid} not found in template.")
        self.atom_order = atom_order_data[self.amino_acid]
        self.num_atoms = len(self.atom_order)

        assert (
            self.num_atoms == self.sidechain_positions.shape[1]
        ), f"Number of atoms in dataset {self.sidechain_positions.shape[1]} doesn't match template {self.num_atoms}"

        latent_file = os.path.join(latent_path, f"{self.amino_acid}.pt")
        if not os.path.exists(latent_file):
            raise FileNotFoundError(f"Latent file {latent_file} not found!")
        self.sidechain_latents = torch.load(latent_file, weights_only=True).cpu()
        self.latent_dim = self.sidechain_latents.shape[1]

        self.mean, self.std = None, None
        self.energy = None
        if energy_path is not None:
            data_file = os.path.join(energy_path, f"{self.amino_acid}.pt")
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Data file {data_file} not found!")
            key_to_col, energy = torch.load(data_file, weights_only=True)
            if force_types is not None:
                cols = []
                for force in force_types:
                    cols.append(key_to_col[force])
                energy = energy[:, cols]
            else:
                force_types = list(key_to_col.keys())

            if iqr_filter_energy:
                energy = iqr_filtering_energy(energy)

            if interpolate_energy:
                energy = interpolating_energy(energy, self.sidechain_latents)

            if inf_filter:
                mask = inf_filter_energy(energy)
                energy = energy[mask]
                self.sidechain_latents = self.sidechain_latents[mask]
                self.sidechain_positions = self.sidechain_positions[mask]

            if normalize_energy:
                energy, self.mean, self.std = normalizing_energy(energy)

            self.energy = energy
            self.force_types = force_types
            self.energy_dim = self.energy.shape[1]

            # Calculate all torsion angles
            self.torsion_atom_order = torsion_atom_order(
                self.amino_acid, self.atom_order
            )
            self.torsion_angles = torsion_angles(
                self.amino_acid, self.sidechain_positions, self.atom_order
            )
            self.num_angles = self.torsion_angles.shape[1]
            assert self.num_angles == len(
                self.torsion_atom_order
            ), f"Number of side-chain torsion angles in dataset {len(self.torsion_atom_order)} doesn't match template {self.num_angles}"

            assert (
                self.energy.shape[0]
                == self.sidechain_latents.shape[0]
                == self.sidechain_positions.shape[0]
            ), f"Number of energy {self.energy.shape[0]} doesn't match with number of latents {self.sidechain_latents.shape[0]} or sidechain coords {self.sidechain_positions.shape[0]} after filtering"
        else:
            self.energy = None
            self.energy_dim = 0
        self.num_real_samples = self.sidechain_positions.shape[0]

    # Fixes windows multi-procces pickling issue
    @property
    def torch_kdtree(self):
        # Lazy initialization: build the KDTree only once per process
        if self._torch_kdtree is None:
            print("Building Tree")
            self._torch_kdtree = build_kd_tree(self.sidechain_latents, device="cuda")
        return self._torch_kdtree

    def __getstate__(self):
        # Exclude the non-picklable KDTree from being pickled
        state = self.__dict__.copy()
        state["_torch_kdtree"] = None
        return state

    def resample_data(self):
        # Sample non_border_points:
        non_border_pct = 1 - self.border_pct
        num_non_border_points = int(
            self.num_real_samples * self.increase_factor * non_border_pct
        )
        if self.init:
            print(f"Samping {num_non_border_points} random points")

        non_border_samples = sample_hypersphere(
            num_non_border_points, self.latent_dim
        ).cuda()
        distances, non_border_idxs = self.torch_kdtree.query(non_border_samples, 1)

        # Sample border_points:
        num_border_points = int(
            self.num_real_samples * self.increase_factor * self.border_pct
        )
        if self.init:
            print(f"Samping {num_border_points} border points")
            self.init = False

        samples_per_iter = int(num_border_points * 0.5)
        keep_pct = 0.03
        border_samples = torch.empty((0, self.latent_dim), device=torch.device("cuda"))
        border_idxs = torch.empty((0, 1), device=torch.device("cuda"))
        while True and num_border_points > 0:
            # Generate sample points in 3D space
            samples = sample_hypersphere(samples_per_iter, self.latent_dim).cuda()

            # Find nearest and second nearest neighbor for each sample
            distances, all_idxs = self.torch_kdtree.query(samples, 2)

            # Compute error metric
            error = (distances[:, 0] - distances[:, 1]) ** 2
            _, top_k_idxs = torch.topk(
                error, max(1, int(samples_per_iter * keep_pct)), largest=False
            )

            # Get border points
            border_samples = torch.vstack((border_samples, samples[top_k_idxs]))
            border_idxs = torch.cat((border_idxs, all_idxs[top_k_idxs, :1]))

            _, counts = torch.unique(border_idxs, return_counts=True)
            if counts.sum() >= num_border_points:
                break

        self.sample_latents = torch.cat((non_border_samples, border_samples)).cpu()
        self.nearest_real_sample_idxs = (
            torch.cat(
                (
                    non_border_idxs,
                    border_idxs,
                )
            )
            .int()
            .squeeze()
            .cpu()
        )

    def __len__(self):
        return len(self.nearest_real_sample_idxs)

    def __getitem__(self, idx):
        if self.energy is None:
            return (
                self.sidechain_positions[self.nearest_real_sample_idxs[idx]],
                self.sample_latents[idx],
            )
        return (
            self.sidechain_positions[self.nearest_real_sample_idxs[idx]],
            self.sample_latents[idx],
            self.energy[self.nearest_real_sample_idxs[idx]],
        )


class LatentEvalDataset(Dataset):

    def __init__(
        self,
        amino_acid="ARG",
        data_path="dataset/clean",
        latent_path="dataset/latents",
        energy_path=None,
        force_types=None,
        iqr_filter_energy=False,
        interpolate_energy=False,
        normalize_energy=False,
        inf_filter=False,
        fixed_O=True,
    ):
        self.amino_acid = amino_acid.upper()
        self.force_types = force_types
        self.energy_normalized = False
        data_file = os.path.join(data_path, f"{self.amino_acid}.pt")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found!")
        self.sidechain_positions = torch.load(data_file, weights_only=True).cpu()

        if fixed_O:
            o_ref = torch.median(self.sidechain_positions[:, 4, :], dim=0).values
            self.sidechain_positions[:, 4, :] = o_ref

        atom_order_data = extract_atom_order(os.path.join("dataset", "template.pdb"))
        if self.amino_acid not in atom_order_data:
            raise KeyError(f"Amino acid {self.amino_acid} not found in template.")
        self.atom_order = atom_order_data[self.amino_acid]
        self.num_atoms = len(self.atom_order)

        assert (
            self.num_atoms == self.sidechain_positions.shape[1]
        ), f"Number of atoms in dataset {self.sidechain_positions.shape[1]} doesn't match template {self.num_atoms}"

        latent_file = os.path.join(latent_path, f"{self.amino_acid}.pt")
        if not os.path.exists(latent_file):
            raise FileNotFoundError(f"Latent file {latent_file} not found!")
        self.sidechain_latents = torch.load(latent_file, weights_only=True).cpu()
        self.latent_dim = self.sidechain_latents.shape[1]

        self.mean, self.std = None, None
        self.energy = None
        if energy_path is not None:
            data_file = os.path.join(energy_path, f"{self.amino_acid}.pt")
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Data file {data_file} not found!")
            key_to_col, energy = torch.load(data_file, weights_only=True)
            if force_types is not None:
                cols = []
                for force in force_types:
                    cols.append(key_to_col[force])
                energy = energy[:, cols]
            else:
                force_types = list(key_to_col.keys())

            if iqr_filter_energy:
                energy = iqr_filtering_energy(energy)

            if interpolate_energy:
                energy = interpolating_energy(energy, self.sidechain_latents)

            if inf_filter:
                mask = inf_filter_energy(energy)
                energy = energy[mask]
                self.sidechain_latents = self.sidechain_latents[mask]
                self.sidechain_positions = self.sidechain_positions[mask]

            if normalize_energy:
                assert not self.energy_normalized, "Energy already normalized"
                self.energy_normalized = True
                energy, self.mean, self.std = normalizing_energy(energy)

            self.energy = energy
            self.force_types = force_types
            self.energy_dim = self.energy.shape[1]

            # Calculate all torsion angles
            self.torsion_atom_order = torsion_atom_order(
                self.amino_acid, self.atom_order
            )
            self.torsion_angles = torsion_angles(
                self.amino_acid, self.sidechain_positions, self.atom_order
            )
            self.num_angles = self.torsion_angles.shape[1]
            assert self.num_angles == len(
                self.torsion_atom_order
            ), f"Number of side-chain torsion angles in dataset {len(self.torsion_atom_order)} doesn't match template {self.num_angles}"

            assert (
                self.energy.shape[0]
                == self.sidechain_latents.shape[0]
                == self.sidechain_positions.shape[0]
            ), f"Number of energy {self.energy.shape[0]} doesn't match with number of latents {self.sidechain_latents.shape[0]} or sidechain coords {self.sidechain_positions.shape[0]} after filtering"
        else:
            self.energy = None

    def normalize_energy(self, mean=None, std=None):
        assert not self.energy_normalized, "Energy already normalized"
        self.energy_normalized = True
        self.energy, self.mean, self.std = normalizing_energy(
            energy=self.energy, mean=mean, std=std
        )
        return self.mean, self.std

    def __len__(self):
        return self.sidechain_positions.shape[0]

    def __getitem__(self, idx):
        if self.energy == None:
            return self.sidechain_positions[idx], self.sidechain_latents[idx]
        return (
            self.sidechain_positions[idx],
            self.sidechain_latents[idx],
            self.energy[idx],
        )


def nanstd(o, dim, keepdim=False):

    result = torch.sqrt(
        torch.nanmean(
            torch.pow(torch.abs(o - torch.nanmean(o, dim=dim).unsqueeze(dim)), 2),
            dim=dim,
        )
    )

    if keepdim:
        result = result.unsqueeze(dim)

    return result


def normalizing_energy(energy, mean=None, std=None):
    # Mask for finite values
    mask = torch.isfinite(energy)
    valid_energy = energy.clone()
    valid_energy[~mask] = float("nan")
    if mean is None and std is None:
        mean = torch.nanmean(valid_energy, dim=0)
        std = nanstd(valid_energy, dim=0)
    normalized_energy = torch.where(mask, (energy - mean) / std, energy)
    return normalized_energy, mean, std


def interpolating_energy(energy, latents):
    interpolated_energy = energy.clone()

    num_energy = energy.shape[1]
    for i in range(num_energy):
        non_energy_mask = torch.isinf(energy[:, i])
        energy_mask = ~non_energy_mask

        tree = build_kd_tree(latents[energy_mask], device="cuda")
        distance_tensor, index_tensor = tree.query(latents[non_energy_mask], 12)
        distance_tensor = distance_tensor.cpu()
        index_tensor = index_tensor.cpu()
        retrieved_energies = energy[energy_mask, i][index_tensor.int()]
        normalizer = distance_tensor.sum(dim=1, keepdim=True)
        normalized_distance = distance_tensor / normalizer
        weights = normalized_distance.flip(dims=[1])
        predicted_energy = (retrieved_energies * weights).sum(dim=1)

        interpolated_energy[non_energy_mask, i] = predicted_energy

    return interpolated_energy


def iqr_filtering_energy(energy):
    # Mask for finite values
    mask = torch.isfinite(energy)
    valid_energy = energy.clone()
    valid_energy[~mask] = float("nan")  # Convert inf to NaN for easier filtering

    q1 = torch.nanquantile(valid_energy, 0.25, dim=0)
    q3 = torch.nanquantile(valid_energy, 0.75, dim=0)
    iqr = q3 - q1

    upper_bound = q3 + 4 * iqr
    iqr_mask = valid_energy <= upper_bound
    iqr_outlier_mask = ~iqr_mask
    energy[iqr_outlier_mask] = torch.inf
    row_has_inf = torch.isinf(energy).any(dim=1)
    energy[row_has_inf] = torch.inf

    return energy


def inf_filter_energy(energy):
    energy = torch.where(
        torch.any(torch.isinf(energy), dim=1, keepdim=True), torch.inf, energy
    )
    mask = ~torch.isinf(energy[:, 0])
    return mask
