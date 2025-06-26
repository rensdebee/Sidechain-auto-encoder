import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EnergyPredictionModel(pl.LightningModule):
    def __init__(self, config, dataset=None):
        super(EnergyPredictionModel, self).__init__()
        self.input_dim = config["latent_dim"]
        self.output_dim = config["output_dim"]
        self.energy_types = config["force"]
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.output_dim)  # Predict scalar energy value
        self.mean = config["data_mean"]
        self.std = config["data_std"]
        self.val_diff = [[] for _ in range(self.output_dim)]
        self.train_diff = [[] for _ in range(self.output_dim)]
        self.norm_val_diff = [[] for _ in range(self.output_dim)]
        self.train_loss = 0
        self.val_loss = 0
        self.save_hyperparameters(ignore="dataset")
        self.dataset = dataset
        self.config = config

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def training_step(self, batch, batch_idx):
        _, latents, e = batch
        e_hat = self(latents)
        loss = 0
        for i, energy_type in enumerate(self.energy_types):
            mask = ~torch.isinf(e[:, i])
            if mask.sum() > 0:
                filtered_e = e[:, i][mask]

                filtered_e_hat = e_hat[:, i][mask]
                ith_loss = nn.functional.mse_loss(
                    filtered_e_hat, filtered_e, reduction="mean"
                )
                filtered_e = filtered_e * self.std[i] + self.mean[i]
                filtered_e_hat = filtered_e_hat * self.std[i] + self.mean[i]
                loss += ith_loss
                diff = torch.abs(filtered_e - filtered_e_hat).detach()
                self.train_diff[i].append(diff)
                self.log(
                    f"train_{energy_type}_loss_step",
                    ith_loss,
                    batch_size=e_hat.shape[0],
                )
        self.train_loss += loss
        return loss

    def on_train_epoch_end(self):
        for i, energy_type in enumerate(self.energy_types):
            diff = torch.cat(self.train_diff[i])
            mean = torch.mean(diff)
            std = torch.std(diff)
            self.log(f"train_{energy_type}_mae_epoch", mean)
            self.log(f"train_{energy_type}_std_epoch", std)
        self.train_diff = [[] for _ in range(self.output_dim)]
        self.log(
            f"train_loss_epoch",
            self.train_loss,
        )
        self.train_loss = 0

    def validation_step(self, batch, batch_idx):
        _, latents, e = batch
        e_hat = self(latents)
        loss = 0
        for i, energy_type in enumerate(self.energy_types):
            mask = ~torch.isinf(e[:, i])
            if mask.sum() > 0:
                filtered_e = e[:, i][mask]
                filtered_e_hat = e_hat[:, i][mask]
                ith_loss = nn.functional.mse_loss(
                    filtered_e_hat, filtered_e, reduction="mean"
                )
                loss += ith_loss

                diff = torch.abs(filtered_e - filtered_e_hat).detach()
                self.norm_val_diff[i].append(diff)
                if self.mean is not None:
                    filtered_e = filtered_e * self.std[i] + self.mean[i]
                    filtered_e_hat = filtered_e_hat * self.std[i] + self.mean[i]
                    diff = torch.abs(filtered_e - filtered_e_hat).detach()
                self.val_diff[i].append(diff)

        self.val_loss += loss

    def on_validation_epoch_end(self):
        for i, energy_type in enumerate(self.energy_types):
            diff = torch.cat(self.val_diff[i])
            mean = torch.mean(diff)
            std = torch.std(diff)
            self.log(f"val_{energy_type}_mae_epoch", mean)
            self.log(f"val_{energy_type}_std_epoch", std)
            diff = torch.cat(self.norm_val_diff[i])
            mean = torch.mean(diff)
            std = torch.std(diff)
            self.log(f"val_{energy_type}_norm_mae_epoch", mean)
            self.log(f"val_{energy_type}_norm_std_epoch", std)
        self.val_diff = [[] for _ in range(self.output_dim)]
        self.norm_val_diff = [[] for _ in range(self.output_dim)]
        self.log(
            f"val_loss_epoch",
            self.val_loss,
        )
        self.val_loss = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

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
