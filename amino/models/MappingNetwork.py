from math import cos, pi

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from amino.models.utils import (
    distance_matrix_loss,
    hypersphere_combined_loss,
    normalize_latent,
    recon_regu,
)


class MappingNetwork(pl.LightningModule):
    def __init__(self, config):
        super(MappingNetwork, self).__init__()
        self.config = config
        self.input_dim = config["input_dim"]
        self.num_angles = config["num_angles"]
        self.HAE_dim = config["HAE_dim"]
        self.mul_HAE = config["mul_HAE"]
        if self.mul_HAE:
            self.HAE_dim = 2 * self.num_angles
        self.torsion_dim = config["torsion_dim"]
        self.no_decoder = not config["decoder"]
        self.val_sd = []
        self.train_sd = []
        self.val_ad = []
        self.train_ad = []
        self.train_loss = 0
        self.val_loss = 0
        self.lr = config["lr"]
        self.hidden_dims = config["hidden_dims"]
        self.warmup_epochs = int(config["epochs"] * 0.05)
        self.energy = config["energy"]
        if self.no_decoder:
            self.energy = False
        self.energy_dim = 1
        if self.energy:
            self.energy_dim = config["energy_dim"]
            self.mean = config["data_mean"]
            self.std = config["data_std"]
            self.val_diff = [[] for _ in range(self.energy_dim)]
            self.train_diff = [[] for _ in range(self.energy_dim)]
            self.val_diff_norm = [[] for _ in range(self.energy_dim)]
            self.train_diff_norm = [[] for _ in range(self.energy_dim)]
        self.save_hyperparameters()

        # Encoder
        encoder_layers = [
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),
            nn.ReLU(),
        ]
        for i in range(1, len(self.hidden_dims)):
            encoder_layers += [
                nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                nn.BatchNorm1d(self.hidden_dims[i]),
                nn.ReLU(),
            ]
        encoder_layers.append(nn.Linear(self.hidden_dims[-1], self.HAE_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        if self.mul_HAE:
            self.mapping_network = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                    )
                    for _ in range(self.num_angles)
                ]
            )
        else:
            self.mapping_network = nn.Sequential(
                nn.Linear(self.HAE_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, self.torsion_dim),
            )

        if self.energy:
            self.energy_network = nn.Sequential(
                nn.Linear(self.num_angles * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.energy_dim),
            )
        # Decoder
        if not self.no_decoder:
            decoder_layers = [
                nn.Linear(self.num_angles * 2, self.hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.BatchNorm1d(self.hidden_dims[-1]),
            ]
            for i in range(-1, -1 * len(self.hidden_dims), -1):
                decoder_layers += [
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i - 1]),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.BatchNorm1d(self.hidden_dims[i - 1]),
                ]
            decoder_layers += [nn.Linear(self.hidden_dims[0], self.input_dim)]
            self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x.flatten(1))
        if self.mul_HAE:
            z = z.view(x.shape[0], self.num_angles, 2)
        pred_sin_cos, pred_x, pred_e = self.reconstruct(z)
        return z, pred_sin_cos, pred_x, pred_e

    def reconstruct(self, z):
        z = normalize_latent(
            z
        )  # [batch_size, HAE_dim] or # [batch_size, num angles, 2]
        if self.mul_HAE:
            rotated = []
            for k in range(self.num_angles):
                v_k = z[:, k]  # (batch_size, 2)
                theta_k = self.mapping_network[k](v_k).flatten()  # (batch_size,)

                cos_theta = torch.cos(theta_k)
                sin_theta = torch.sin(theta_k)
                # Build rotation matrix per batch
                rot_matrix = torch.stack(
                    [
                        torch.stack([cos_theta, -sin_theta], dim=1),
                        torch.stack([sin_theta, cos_theta], dim=1),
                    ],
                    dim=2,
                )  # shape: (batch_size, 2, 2)
                # Apply rotation
                v_k_rotated = torch.bmm(rot_matrix, v_k.unsqueeze(-1)).squeeze(
                    -1
                )  # (batch_size, 2)
                rotated.append(v_k_rotated)
                pred_sin_cos = torch.stack(rotated, dim=1)  # (batch_size, K, 2)
        else:
            angles = self.mapping_network(z)
            if self.torsion_dim == self.num_angles:  # [batch_size, num_angles]
                pred_sin_cos = self.angles_to_unit_vectors(
                    angles
                )  # [batch_size, num_angles, 2]
            else:  # [batch_size, num_angles* 2]
                pred_sin_cos = self.normalize_angles(
                    angles.view(angles.size(0), self.num_angles, 2)
                )  # [batch_size, num_angles, 2]
        pred_x = None
        if not self.no_decoder:
            pred_x = self.decoder(pred_sin_cos.flatten(1))
            pred_x = pred_x.unflatten(dim=1, sizes=(-1, 3))
        pred_e = None
        if self.energy:
            pred_e = self.energy_network(pred_sin_cos.flatten(1))
        return pred_sin_cos, pred_x, pred_e

    def training_step(self, batch, batch_idx):
        x = batch["sidechain_position"]
        torsion_angles = batch["torsion_angles"]
        sin_cos = self.angles_to_unit_vectors(torsion_angles)
        z, pred_sin_cos, pred_x, pred_e = self(x)

        # Calc loss
        if not self.no_decoder:
            recon_loss = (
                torch.nn.functional.mse_loss(pred_x, x)
                + distance_matrix_loss(pred_x, x)
            ) / 2
            self.train_sd.append(self.calculate_3D_squared_distance(pred_x, x))
            self.log(
                "recon_loss",
                recon_loss,
                batch_size=z.shape[0],
            )

        torsion_angle_loss = torch.nn.functional.mse_loss(pred_sin_cos, sin_cos)
        self.log(
            "torsion_angle_loss",
            torsion_angle_loss,
            batch_size=z.shape[0],
        )
        # rmsd_matrix = self.compute_rmsd_matrix(x)
        hae_loss = hypersphere_combined_loss(z, rmsd_matrix=None)
        self.log(
            "train_HAE_loss_step",
            hae_loss,
            batch_size=z.shape[0],
        )
        if not self.no_decoder:
            recon_factor = 0.35 if self.energy else 0.45
            loss = (
                0.5 * torsion_angle_loss + recon_factor * recon_loss + 0.05 * hae_loss
            )
        else:
            loss = 0.9 * torsion_angle_loss + 0.1 * hae_loss
        if self.energy:
            e = batch["energy"]
            loss += 0.1 * self.energy_loss_fn(pred_e, e)
        self.train_ad.append(
            self.calculate_angle_difference(pred_sin_cos, torsion_angles)
        )
        self.log(
            "train_loss_step",
            loss,
            batch_size=z.shape[0],
        )
        return loss

    def on_train_epoch_end(self):
        if not self.no_decoder:
            self.log(
                "train_rmsd_epoch",
                torch.sqrt(torch.mean(torch.cat(self.train_sd))),
            )
        self.train_sd = []

        angular_diff_deg = torch.cat(self.train_ad)
        # Log mean over all angles (scalar)
        self.log("train_mean_angular_error_deg", angular_diff_deg.mean())

        # Log mean per angle index (e.g., angle_0, angle_1, ...)
        per_angle_mean = angular_diff_deg.mean(dim=0)  # [N]
        for i, val in enumerate(per_angle_mean):
            self.log(f"train_angle_{i}_mean_error_deg", val)
        self.train_ad = []

        if self.energy:
            for i in range(self.energy_dim):
                energy_type = self.config["force_types"][i]
                mean = torch.mean(torch.cat(self.train_diff[i]))
                self.log(f"train_{energy_type}_mae_epoch", mean)
                mean = torch.mean(torch.cat(self.train_diff_norm[i]))
                self.log(f"train_{energy_type}_norm_mae_epoch", mean)
            self.train_diff = [[] for _ in range(self.energy_dim)]
            self.train_diff_norm = [[] for _ in range(self.energy_dim)]

    def validation_step(self, batch, batch_idx):
        x = batch["sidechain_position"]
        torsion_angles = batch["torsion_angles"]
        z, pred_sin_cos, pred_x, pred_e = self(x)

        # Calc metrics
        if self.energy:
            e = batch["energy"]
            self.eval_energy(pred_e, e)
        if not self.no_decoder:
            self.val_sd.append(self.calculate_3D_squared_distance(pred_x, x))
        self.val_ad.append(
            self.calculate_angle_difference(pred_sin_cos, torsion_angles)
        )

    def on_validation_epoch_end(self):
        if not self.no_decoder:
            self.log(
                "val_rmsd_epoch",
                torch.sqrt(torch.mean(torch.cat(self.val_sd))),
            )
        self.val_sd = []

        angular_diff_deg = torch.cat(self.val_ad)
        # Log mean over all angles (scalar)
        self.log("val_mean_angular_error_deg", angular_diff_deg.mean())

        # Log mean per angle index (e.g., angle_0, angle_1, ...)
        per_angle_mean = angular_diff_deg.mean(dim=0)
        for i, val in enumerate(per_angle_mean):
            self.log(f"val_angle_{i}_mean_error_deg", val)
        self.val_ad = []

        if self.energy:
            for i in range(self.energy_dim):
                energy_type = self.config["force_types"][i]
                mean = torch.mean(torch.cat(self.val_diff[i]))
                self.log(f"val_{energy_type}_mae_epoch", mean)
                mean = torch.mean(torch.cat(self.val_diff_norm[i]))
                self.log(f"val_{energy_type}_norm_mae_epoch", mean)
            self.val_diff = [[] for _ in range(self.energy_dim)]
            self.val_diff_norm = [[] for _ in range(self.energy_dim)]

    def energy_loss_fn(self, pred_e, e):
        energy_loss = 0
        for i in range(self.energy_dim):
            mask = ~torch.isinf(e[:, i])
            if mask.sum() > 0:
                energy_type = self.config["force_types"][i]

                filtered_e = e[:, i][mask]
                filtered_e_hat = pred_e[:, i][mask]

                ith_loss = nn.functional.mse_loss(
                    filtered_e_hat, filtered_e, reduction="mean"
                )
                energy_loss += ith_loss
                self.log(
                    f"train_{energy_type}_loss_step",
                    ith_loss,
                    batch_size=pred_e.shape[0],
                )

                diff = torch.abs(filtered_e - filtered_e_hat).detach()
                self.train_diff_norm[i].append(diff)
                if self.mean is not None:
                    filtered_e = filtered_e * self.std[i] + self.mean[i]
                    filtered_e_hat = filtered_e_hat * self.std[i] + self.mean[i]
                    diff = torch.abs(filtered_e - filtered_e_hat).detach()
                    self.train_diff[i].append(diff)

        return energy_loss

    def eval_energy(self, pred_e, e):
        for i in range(self.energy_dim):
            mask = ~torch.isinf(e[:, i])
            if mask.sum() > 0:
                filtered_e = e[:, i][mask]
                filtered_e_hat = pred_e[:, i][mask]

                diff = torch.abs(filtered_e - filtered_e_hat).detach()
                self.val_diff_norm[i].append(diff)
                if self.mean is not None:
                    filtered_e = filtered_e * self.std[i] + self.mean[i]
                    filtered_e_hat = filtered_e_hat * self.std[i] + self.mean[i]
                    diff = torch.abs(filtered_e - filtered_e_hat).detach()
                    self.val_diff[i].append(diff)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        decay_scale = 0.5

        # Define the warm-up scheduler
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warm-up from 0 to initial learning rate
                return float(epoch + 1) / float(self.warmup_epochs)
            else:
                # Cosine decay
                progress = (epoch - self.warmup_epochs) / max(
                    1, self.trainer.max_epochs - self.warmup_epochs
                )
                return (1.0 - decay_scale) + decay_scale * 0.5 * (
                    1.0 + cos(pi * progress)
                )

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "epoch",  # Update LR at the end of each epoch
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def normalize_angles(self, vectors):  # [batch_size, num_angles, 2]
        return vectors / (vectors.norm(dim=-1, keepdim=True) + 1e-8)

    def angles_to_unit_vectors(self, angle_batch):  # [batch_size, num_angles]
        return torch.stack(
            (torch.sin(angle_batch), torch.cos(angle_batch)), dim=-1
        )  # [batch_size, num_angles, 2]

    def calculate_3D_squared_distance(self, pred, target):
        diff = pred - target
        sd = torch.sum(diff * diff, dim=-1)
        return sd.flatten().detach()

    def calculate_angle_difference(self, pred_sin_cos, target_angles):
        pred_angles = torch.rad2deg(
            torch.atan2(pred_sin_cos[..., 0], pred_sin_cos[..., 1]).detach()
        )  # [batch_size, num_angles]
        target_angles = torch.rad2deg(target_angles)
        angular_diff_rad = (pred_angles - target_angles + 180) % (360) - 180
        angular_diff_deg = torch.abs(angular_diff_rad)
        return angular_diff_deg

    def compute_rmsd_matrix(self, coords_batch):
        # Compute pairwise squared differences using broadcasting
        diffs = (
            coords_batch[:, None, :, :] - coords_batch[None, :, :, :]
        )  # [B, B, N, 3]

        # Sum squared coordinates, average over atoms, then sqrt
        sq_dists = torch.sum(diffs**2, dim=-1)  # [B, B, N]
        rmsd_matrix = torch.sqrt(torch.mean(sq_dists, dim=-1))  # [B, B]
        return rmsd_matrix

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Customize which elements of the batch are moved to the GPU
        batch["sidechain_position"] = batch["sidechain_position"].to(
            device
        )  # Move only 'sidechain_position' to GPU
        batch["torsion_angles"] = (
            batch["torsion_angles"].to(device).float()
        )  # Move only 'torsion_angles' to GPU
        if self.energy:
            batch["energy"] = batch["energy"].to(device)
        return batch
