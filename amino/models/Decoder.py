import sys
from math import cos, pi

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from amino.models.utils import calculate_3D_squared_distance, distance_matrix_loss


class Decoder(nn.Module):

    def __init__(
        self,
        output_dim,
        latent_dim,
        hidden_dims=[1024],
    ):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        assert len(hidden_dims) > 0, "At least one hidden dimension must be provided"
        assert latent_dim > 0, "Latent dimension must be bigger then 0"

        # Decoder
        decoder_layers = [
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
        ]
        for i in range(-1, -1 * len(hidden_dims), -1):
            decoder_layers += [
                nn.Linear(hidden_dims[i], hidden_dims[i - 1]),
                nn.BatchNorm1d(hidden_dims[i - 1]),
                nn.ReLU(),
            ]
        decoder_layers += [nn.Linear(hidden_dims[0], output_dim)]
        self.decoder = nn.Sequential(*decoder_layers)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, z):
        # Decode
        x_hat = self.decode(z)

        return x_hat


class DecoderLightning(pl.LightningModule):
    def __init__(self, config, dataset=None):
        super(DecoderLightning, self).__init__()
        self.config = config
        self.decoder = Decoder(
            output_dim=config["output_dim"],
            hidden_dims=config["hidden_dims"],
            latent_dim=config["latent_dim"],
        )
        self.learning_rate = config["lr"]
        self.warmup_epochs = 10
        self.energy = config["energy"]
        if self.energy:
            self.energy_dim = config["energy_dim"]
            self.mean = config["data_mean"]
            self.std = config["data_std"]
            self.val_diff = [[] for _ in range(self.energy_dim)]
            self.train_diff = [[] for _ in range(self.energy_dim)]
        self.save_hyperparameters(ignore="dataset")
        self.dataset = dataset
        self.train_sd = []
        self.val_sd = []

    def forward(self, x):
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        if self.energy:
            x, z, e = batch
            output = self(z)
            x_hat, e_hat = (
                output[:, : -self.energy_dim],
                output[:, -self.energy_dim :],
            )
            energy_loss = 0
            for i in range(self.energy_dim):
                mask = ~torch.isinf(e[:, i])
                if mask.sum() > 0:
                    energy_type = self.config["force_types"][i]

                    filtered_e = e[:, i][mask]
                    filtered_e_hat = e_hat[:, i][mask]

                    ith_loss = nn.functional.mse_loss(
                        filtered_e_hat, filtered_e, reduction="mean"
                    )
                    if self.mean is not None:
                        filtered_e = filtered_e * self.std[i] + self.mean[i]
                        filtered_e_hat = filtered_e_hat * self.std[i] + self.mean[i]
                    energy_loss += ith_loss
                    diff = torch.abs(filtered_e - filtered_e_hat).detach()
                    self.train_diff[i].append(diff)
                    self.log(
                        f"train_{energy_type}_loss_step",
                        ith_loss,
                        batch_size=x.shape[0],
                    )
        else:
            x, z = batch
            x_hat = self(z)

        x_hat = x_hat.unflatten(dim=1, sizes=(-1, 3))

        recon_loss = (
            nn.functional.mse_loss(x_hat, x, reduction="mean")
            + distance_matrix_loss(x_hat, x)
        ) / 2
        loss = recon_loss
        if self.energy:
            loss = recon_loss
            loss = 0.5 * loss + 0.5 * energy_loss
        self.log(
            "train_recon_loss_step",
            recon_loss,
            batch_size=x.shape[0],
        )
        self.log(
            "train_loss_step",
            loss,
            batch_size=x.shape[0],
        )
        sd = calculate_3D_squared_distance(x_hat, x)
        self.log(
            "train_rmsd_step",
            torch.sqrt(torch.mean(sd)),
            batch_size=x.shape[0],
        )
        self.log(
            "train_msd_step",
            torch.mean(sd),
            batch_size=x.shape[0],
        )
        self.train_sd.append(sd)
        return loss

    def on_train_epoch_end(self):
        sd = torch.cat(self.train_sd)
        self.log(
            "train_rmsd_epoch",
            torch.sqrt(torch.mean(sd)),
        )
        self.log(
            "train_msd_epoch",
            torch.mean(sd),
        )
        self.train_sd.clear()
        if self.energy:
            for i in range(self.energy_dim):
                diff = torch.cat(self.train_diff[i])
                mean = torch.mean(diff)
                std = torch.std(diff)
                energy_type = self.config["force_types"][i]
                self.log(f"train_{energy_type}_mae_epoch", mean)
                self.log(f"train_{energy_type}_std_epoch", std)
            self.train_diff = [[] for _ in range(self.energy_dim)]

    def validation_step(self, batch, batch_idx):
        if self.energy:
            x, z, e = batch
            output = self(z)
            x_hat, e_hat = output[:, : -self.energy_dim], output[:, -self.energy_dim :]
            for i in range(self.energy_dim):
                mask = ~torch.isinf(e[:, i])
                if mask.sum() > 0:
                    filtered_e = e[:, i][mask]
                    filtered_e_hat = e_hat[:, i][mask]
                    if self.mean is not None:
                        filtered_e = filtered_e * self.std[i] + self.mean[i]
                        filtered_e_hat = filtered_e_hat * self.std[i] + self.mean[i]
                    diff = torch.abs(filtered_e - filtered_e_hat).detach()
                    self.val_diff[i].append(diff)
        else:
            x, z = batch
            x_hat = self(z)
        x_hat = x_hat.unflatten(dim=1, sizes=(-1, 3))

        sd = calculate_3D_squared_distance(x_hat, x)
        self.val_sd.append(sd)

    def on_validation_epoch_end(self):
        sd = torch.cat(self.val_sd)
        self.log(
            "val_rmsd_epoch",
            torch.sqrt(torch.mean(sd)),
        )
        self.log(
            "val_msd_epoch",
            torch.mean(sd),
        )
        self.val_sd.clear()
        if self.energy:
            for i in range(self.energy_dim):
                diff = torch.cat(self.val_diff[i])
                mean = torch.mean(diff)
                std = torch.std(diff)
                energy_type = self.config["force_types"][i]
                self.log(f"val_{energy_type}_mae_epoch", mean)
                self.log(f"val_{energy_type}_std_epoch", std)
            self.val_diff = [[] for _ in range(self.energy_dim)]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
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

    def train_dataloader(self):
        with torch.no_grad():
            self.dataset.resample_data()
        loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            persistent_workers=True if self.config["num_workers"] > 0 else False,
            pin_memory=False if sys.platform.startswith("win") else True,
        )
        return loader


if __name__ == "__main__":
    decoder = Decoder(768, 4, [2048, 1024, 512, 128])
    print(decoder)
