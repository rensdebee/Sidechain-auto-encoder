import argparse
import ast
import sys
from datetime import datetime

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from amino.data.datasets import SidechainDataset
from amino.models.AutoEncoder import AutoencoderLightning


def main(config):
    unique_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    config["unique_id"] = unique_id
    if config["VAE"]:
        type = "VAE"
    elif config["HAE"]:
        type = "HAE"
    elif config["SVAE"]:
        type = "SVAE"
    elif config["PSVAE"]:
        type = "PSVAE"
    else:
        type = "AE"
    post_fix = f"_{config['reg_ratio']}"

    dataset = SidechainDataset(config["amino_acid"], config["data_path"], fixed_O=True)

    config["input_dim"] = dataset.num_atoms * 3
    if config["latent_dim"] == 0:
        if config["torsion"]:
            config["latent_dim"] = dataset.num_angles * 2
        else:
            config["latent_dim"] = max(len(dataset.torsion_atom_order), 3)

    name = f"{config['amino_acid']}_dim_{config['latent_dim']}"
    print(config)
    if "synth/high_energy" in config["data_path"]:
        mode = "synth/high_energy"
    elif "synth" in config["data_path"]:
        mode = "synth"
    elif "full" in config["data_path"]:
        mode = "full"
    else:
        mode = ""
    model = AutoencoderLightning(config)

    # Create DataLoaders for train and test datasets
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        persistent_workers=True,
        pin_memory=False if sys.platform.startswith("win") else True,
    )

    wandb_logger = WandbLogger(
        name=name + "_" + post_fix + f"_{unique_id}", project=f"{type}_phase_1"
    )
    wandb_logger.experiment.config.update(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_rmsd_epoch",
        mode="min",
        save_last=True,
        save_on_train_epoch_end=True,
        save_top_k=1,
        dirpath=f"checkpoints/{mode}/{type}/{post_fix}/{unique_id}/{name}",
        filename="best_{epoch}-{train_rmsd_epoch:.3f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=config["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    torch.set_float32_matmul_precision("high")
    torch.compile(model, mode="reduce-overhead")
    trainer.fit(model, train_loader)

    # Finish W&B run
    wandb.finish()


def load_config(args):
    """
    Load configuration from a YAML file and override with command-line arguments.
    """
    # Load defaults from command-line arguments
    config = {
        key: value
        for key, value in vars(args).items()
        if key != "config_file" and value is not None
    }

    # Load from YAML file if provided
    if args.config_file:
        print("Reading yaml")
        with open(args.config_file, "r") as file:
            yaml_config = yaml.safe_load(file) or {}
            config = {
                **config,
                **yaml_config,
            }  # YAML takes precedence, CLI overrides YAML

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autoencoder training script with YAML and CLI config support."
    )

    # Command-line arguments with defaults from the original code
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--amino_acid", type=str, default="ARG", help="Amino acid to process."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4096, help="Batch size for training."
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--VAE",
        action="store_true",
        default=False,
        help="Enable Variational Autoencoder mode.",
    )
    parser.add_argument(
        "--SVAE",
        action="store_true",
        default=False,
        help="Enable Variational Autoencoder mode.",
    )
    parser.add_argument(
        "--PSVAE",
        action="store_true",
        default=False,
        help="Enable Variational Autoencoder mode.",
    )
    parser.add_argument(
        "--HAE",
        action="store_true",
        default=False,
        help="Enable Hierarchical Autoencoder mode.",
    )
    parser.add_argument(
        "--torsion",
        action="store_true",
        default=False,
        help="Enable Hierarchical Autoencoder mode.",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/clean",
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="[1024, 256, 128]",
        help="Hidden dimensions for the autoencoder.",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=0, help="Latent dimension size."
    )
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument("--reg_ratio", type=float, default=0, help="HAE loss ratio.")

    # Parse arguments
    args = parser.parse_args()

    # Load configuration
    config = load_config(args)
    config["hidden_dims"] = ast.literal_eval(config["hidden_dims"])
    # Run main with the loaded configuration
    main(config)
