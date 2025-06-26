import argparse
import os
import sys
from datetime import datetime

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from amino.data.datasets import LatentDataset, LatentEvalDataset
from amino.models.EnergyPredictor import EnergyPredictionModel


# Cross-validation loop
def run_cross_validation(
    config,
):
    unique_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    config["unique_id"] = unique_id
    amino_acid = config["amino_acid"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    type = "HAE_energy"
    if "mapping" in config["latent_path"]:
        type = "mapping_energy"

    eval_dataset = LatentEvalDataset(
        amino_acid=config["amino_acid"],
        data_path=config["data_path"],
        latent_path=config["latent_path"],
        energy_path=config["energy_path"],
        force_types=config["force"],
        iqr_filter_energy=config["iqr_filter_energy"],
        interpolate_energy=config["interpolate_energy"],
        normalize_energy=config["normalize_energy"],
        inf_filter=config["inf_filter"],
        fixed_O=True,
    )

    dataset = LatentDataset(
        amino_acid=config["amino_acid"],
        data_path=config["data_path"],
        latent_path=config["latent_path"],
        energy_path=config["energy_path"],
        force_types=config["force"],
        increase_factor=config["data_increase"],
        border_pct=config["border_ratio"],
        iqr_filter_energy=config["iqr_filter_energy"],
        interpolate_energy=config["interpolate_energy"],
        normalize_energy=config["normalize_energy"],
        inf_filter=config["inf_filter"],
        fixed_O=True,
    )
    config["latent_dim"] = dataset.latent_dim
    config["output_dim"] = dataset.energy_dim
    if "synth/high_energy" in config["data_path"]:
        mode = "synth/high_energy"
    elif "synth" in config["data_path"]:
        mode = "synth"
    elif "full" in config["data_path"]:
        mode = "full"
    else:
        mode = ""

    wandb_logger = WandbLogger(
        name=f"{amino_acid}_{config['unique_id']}",
        project=f"Energy predictor {type}",
    )
    wandb_logger.experiment.config.update(config)

    train_loader = None
    val_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        pin_memory=False if sys.platform.startswith("win") else True,
    )
    config["latent_dim"] = dataset.latent_dim
    config["data_mean"] = dataset.mean
    config["data_std"] = dataset.std
    # Model initialization
    model = EnergyPredictionModel(config, dataset)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_epoch",
        mode="min",
        save_top_k=1,
        dirpath=f"checkpoints/{mode}/{type}/{unique_id}/{amino_acid}_dim_{config['latent_dim']}",
        filename="best_{epoch}-{val_loss_epoch:.3f}",
    )

    # Trainer setup
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        reload_dataloaders_every_n_epochs=1,
    )

    torch.set_float32_matmul_precision("high")
    torch.compile(model, mode="reduce-overhead")
    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Load best model and validate
    best_model = EnergyPredictionModel.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path
    )
    val_trainer = pl.Trainer(logger=False, accelerator=trainer.accelerator, devices=1)
    val_metrics = val_trainer.validate(best_model, val_loader)[0]

    dataset.increase_factor = 2
    dataset.border_pct = 0
    dataset.resample_data()
    loader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        pin_memory=False if sys.platform.startswith("win") else True,
    )
    random_latent_metrics = val_trainer.validate(best_model, loader)[0]

    # Write to text file
    path = f"results/{mode}/{type}"
    os.makedirs(path, exist_ok=True)
    output_file = f"{path}/energy_predictor_results_{amino_acid}_dim{config['latent_dim']}_{unique_id}.txt"
    with open(output_file, "w") as f:
        f.write("HAE energy Metric Summary:\n")
        f.write("Dataset metrics:\n")
        for key, value in val_metrics.items():
            f.write(f"{key}: {value:.3f}:\n")

        f.write("Random Latents metrics:\n")
        for key, value in random_latent_metrics.items():
            f.write(f"{key}: {value:.3f}:\n")
        f.write(str(config))
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
        "--batch_size", type=int, default=512, help="Batch size for training."
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--latent_path",
        type=str,
        default="./dataset/HAE_latents",
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/clean",
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--energy_path",
        type=str,
        default="./dataset/energy",
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--force",
        type=list,
        default=["PeriodicTorsionForce"],
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--iqr_filter_energy",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--interpolate_energy",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--normalize_energy",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--inf_filter",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--border_ratio", type=float, default=0, help="Latent dimension size."
    )
    parser.add_argument(
        "--data_increase", type=float, default=3, help="Latent dimension size."
    )
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args)
    # Run main with the loaded configuration
    run_cross_validation(config)
