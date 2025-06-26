from math import cos, pi

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from amino.distributions.hyperspherical_uniform import (
    HypersphericalUniform as SVAEHypersphericalUniform,
)
from amino.distributions.power_spherical import HypersphericalUniform, PowerSpherical
from amino.distributions.von_mises_fisher import VonMisesFisher
from amino.models.utils import (
    calculate_3D_squared_distance,
    distance_matrix_loss,
    hypersphere_combined_loss,
    normalize_latent,
)


class AutoEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dims=[1024],
        latent_dim=10,
        HAE=False,
        VAE=False,
        SVAE=False,
        PSVAE=False,
        torsion=False,
    ):
        super(AutoEncoder, self).__init__()
        self.hae = HAE
        self.vae = VAE
        self.svae = SVAE
        self.psvae = PSVAE
        self.torsion = torsion
        self.latent_dim = latent_dim

        assert len(hidden_dims) > 0, "At least one hidden dimension must be provided"
        assert latent_dim > 0, "Latent dimension must be bigger then 0"
        assert (
            int(HAE) + int(VAE) != 2
        ), "Cannot be HAE and VAE at the same time, chose one!"
        # Encoder
        encoder_layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
        ]
        for i in range(1, len(hidden_dims)):
            encoder_layers += [
                nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                nn.BatchNorm1d(hidden_dims[i]),
                nn.ReLU(),
            ]
        self.encoder = nn.Sequential(*encoder_layers)

        # latent mean and variance for VAE
        if self.vae:
            self.mean_layer = nn.Linear(hidden_dims[-1], latent_dim)
            self.var_layer = nn.Linear(hidden_dims[-1], latent_dim)
        if self.svae or self.psvae:
            self.mean_layer = nn.Linear(hidden_dims[-1], latent_dim)
            self.var_layer = nn.Linear(hidden_dims[-1], 1)
        # Latent layer for hae or ae
        else:
            self.latent_layer = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = [
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(hidden_dims[-1]),
        ]
        for i in range(-1, -1 * len(hidden_dims), -1):
            decoder_layers += [
                nn.Linear(hidden_dims[i], hidden_dims[i - 1]),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.BatchNorm1d(hidden_dims[i - 1]),
            ]
        decoder_layers += [nn.Linear(hidden_dims[0], input_dim)]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        x = self.encoder(x)
        # (S)VAE
        if self.vae or self.svae or self.psvae:
            mean, var = self.mean_layer(x), self.var_layer(x)
            var = nn.functional.softplus(var)
            if self.svae or self.psvae:
                mean = normalize_latent(mean)
                var = var + 1
            return mean, var
        # (H)AE
        else:
            z = self.latent_layer(x)
            # HAE
            if self.hae:
                z = normalize_latent(z)
            return z

    def reparameterize(self, mean, var):
        if self.svae:
            q_z = VonMisesFisher(mean, var)
            p_z = SVAEHypersphericalUniform(self.latent_dim - 1, device=mean.device)
        elif self.vae:
            q_z = torch.distributions.normal.Normal(mean, var)
            p_z = torch.distributions.normal.Normal(
                torch.zeros_like(mean), torch.ones_like(var)
            )
        elif self.psvae:
            q_z = PowerSpherical(mean, var.squeeze())
            p_z = HypersphericalUniform(self.latent_dim - 1, device=mean.device)
        else:
            raise NotImplementedError
        return q_z, p_z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        # Encode
        # (S)VAE
        if self.vae or self.svae or self.psvae:
            mean, var = self.encode(x)
            q_z, p_z = self.reparameterize(mean, var)
            z = q_z.rsample()
        # (H)AE
        else:
            z = self.encode(x)
            if self.torsion:
                z = nn.functional.tanh(z)
            q_z, p_z = None, None
            mean, var = None, None
        # Decode
        x_hat = self.decode(z)

        return x_hat, z, (q_z, p_z), (mean, var)


class AutoencoderLightning(pl.LightningModule):
    def __init__(self, config):
        super(AutoencoderLightning, self).__init__()
        self.config = config
        self.model = AutoEncoder(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            latent_dim=config["latent_dim"],
            VAE=config["VAE"],
            HAE=config["HAE"],
            SVAE=config["SVAE"],
            PSVAE=config["PSVAE"] if "PSVAE" in config else False,
            torsion=config["torsion"],
        )
        self.torsion = config["torsion"]
        self.learning_rate = config["lr"]
        self.train_sd = []
        self.warmup_epochs = 10
        self.reg_ratio = config["reg_ratio"]
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["sidechain_position"].flatten(1)
        x_hat, z, (q_z, p_z), (_, _) = self(x)
        x_hat = x_hat.unflatten(dim=1, sizes=(-1, 3))
        x = x.unflatten(dim=1, sizes=(-1, 3))
        if self.torsion:
            q_z = batch["torsion_angles"]
        loss = self.loss_fn(x_hat, x, q_z, p_z, z)
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
        self.log(
            "train_rmsd_epoch",
            torch.sqrt(torch.mean(torch.cat(self.train_sd))),
        )
        self.log(
            "train_msd_epoch",
            torch.mean(torch.cat(self.train_sd)),
        )
        self.train_sd = []

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Customize which elements of the batch are moved to the GPU
        batch["sidechain_position"] = batch["sidechain_position"].to(
            device
        )  # Move only 'sidechain_position' to GPU
        batch["torsion_angles"] = (
            batch["torsion_angles"].to(device).float()
        )  # Move only 'torsion_angles' to GPU
        return batch

    def loss_fn(self, x_hat, x, q_z, p_z, z):
        reconstruction_loss = (
            nn.functional.mse_loss(x_hat, x, reduction="mean")
            + distance_matrix_loss(x_hat, x)
        ) / 2
        self.log(
            "train_recon_loss_step",
            reconstruction_loss,
            batch_size=x.shape[0],
        )
        loss = reconstruction_loss
        if self.model.vae:
            kld = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
            self.log(
                "train_kld_loss_step",
                kld,
                batch_size=x.shape[0],
            )
            loss = self.reg_ratio * kld + (1 - self.reg_ratio) * reconstruction_loss
        elif self.model.svae or self.model.psvae:
            kld = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
            self.log(
                "train_kld_loss_step",
                kld,
                batch_size=x.shape[0],
            )
            loss = self.reg_ratio * kld + (1 - self.reg_ratio) * reconstruction_loss
        elif self.model.hae:
            rmsd_matrix = None
            # rmsd_matrix = compute_rmsd_matrix(x)
            hae_loss = hypersphere_combined_loss(z, rmsd_matrix=rmsd_matrix)
            loss = (
                self.reg_ratio * hae_loss + (1 - self.reg_ratio) * reconstruction_loss
            )
        elif self.torsion and q_z.shape[1] * 2 == z.shape[1]:
            torsion_angles = q_z
            sin_vals = torch.sin(torsion_angles)  # Shape: (batch, num_angles)
            cos_vals = torch.cos(torsion_angles)  # Shape: (batch, num_angles)

            # Stack to shape: (batch, 2*num_angles)
            torsion_angles = torch.cat((sin_vals, cos_vals), dim=-1)
            torsion_loss = nn.functional.mse_loss(z, torsion_angles, reduction="mean")
            loss = (
                self.reg_ratio * torsion_loss
                + (1 - self.reg_ratio) * reconstruction_loss
            )
            self.log(
                "train_torsion_loss_step",
                torsion_loss,
                batch_size=x.shape[0],
            )
        return loss

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


if __name__ == "__main__":
    ae = AutoEncoder(768, [2048, 1024, 512, 128], HAE=False, VAE=True)
    print(ae)
