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
from amino.data.datasets import LatentDataset, LatentEvalDataset
from amino.models.Decoder import DecoderLightning


def main(config):
    unique_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    config["unique_id"] = unique_id
    eval_dataset = LatentEvalDataset(
        amino_acid=config["amino_acid"],
        data_path=config["data_path"],
        latent_path=config["latent_path"],
        energy_path=config["energy_path"],
        force_types=config["force_types"],
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
        force_types=config["force_types"],
        increase_factor=config["data_increase"],
        border_pct=config["border_ratio"],
        iqr_filter_energy=config["iqr_filter_energy"],
        interpolate_energy=config["interpolate_energy"],
        normalize_energy=config["normalize_energy"],
        inf_filter=config["inf_filter"],
        fixed_O=True,
    )
    train_loader = None
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        pin_memory=False if sys.platform.startswith("win") else True,
    )

    config["latent_dim"] = dataset.latent_dim
    config["data_mean"] = dataset.mean
    config["data_std"] = dataset.std

    type = config["name"]
    type += "decoder"
    config["energy"] = False
    config["energy_dim"] = None
    if dataset.energy is not None:
        config["energy"] = True
        config["energy_dim"] = dataset.energy_dim
    else:
        type += "_no_energy"
    post_fix = f"_ratio_{config['border_ratio']}"
    if "synth/high_energy" in config["data_path"]:
        mode = "synth/high_energy"
    elif "synth" in config["data_path"]:
        mode = "synth"
    elif "full" in config["data_path"]:
        mode = "full"
    else:
        mode = ""

    name = f"{config['amino_acid']}_dim_{config['latent_dim']}"

    config["output_dim"] = dataset.num_atoms * 3
    if dataset.energy is not None:
        config["output_dim"] += len(config["force_types"])
    print(config)
    model = DecoderLightning(config, dataset)

    wandb_logger = WandbLogger(
        name=name + "_" + post_fix + f"_{unique_id}", project=f"{type}"
    )
    wandb_logger.experiment.config.update(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_rmsd_epoch",
        mode="min",
        save_on_train_epoch_end=True,
        save_top_k=1,
        dirpath=f"./checkpoints/{mode}/{type}/{post_fix}/{unique_id}/{name}",
        filename=("best_{epoch}-{val_rmsd_epoch:.3f}"),
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=config["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback, lr_monitor],
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=5,
    )

    torch.set_float32_matmul_precision("high")
    torch.compile(model, mode="reduce-overhead")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=eval_loader,
    )

    dataset = LatentEvalDataset(
        amino_acid=config["amino_acid"],
        data_path=config["data_path"],
        latent_path=config["latent_path"],
        energy_path=config["energy_path"],
        force_types=config["force_types"],
    )

    # Create DataLoaders for train and test datasets
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        pin_memory=False if sys.platform.startswith("win") else True,
    )
    best_model = DecoderLightning.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path
    )
    val_trainer = pl.Trainer(logger=False, accelerator=trainer.accelerator, devices=1)
    results = val_trainer.validate(best_model, dataloaders=dataloader)
    print("Evaluation Results:", results)

    # Finish W&B run
    wandb.finish()


def load_config(args):
    """
    Load configuration from a YAML file and override with command-line arguments.
    """
    # Load defaults from command-line arguments
    config = {key: value for key, value in vars(args).items() if key != "config_file"}
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
        "--name",
        type=str,
        default="",
        help="Path to the data directory.",
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
        "--latent_path",
        type=str,
        default="./dataset/clean",
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--energy_path",
        type=str,
        default=None,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--force_types",
        type=list,
        default=["PeriodicTorsionForce"],
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="[1024, 256, 128]",
        help="Hidden dimensions for the autoencoder.",
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

    # Parse arguments
    args = parser.parse_args()

    # Load configuration
    config = load_config(args)
    config["hidden_dims"] = ast.literal_eval(config["hidden_dims"])
    # Run main with the loaded configuration
    main(config)
