import argparse
import ast
import os
import sys
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

import wandb
from amino.data.datasets import LatentEvalDataset
from amino.models.Decoder import DecoderLightning


def run_cross_validation(
    config,
):
    unique_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    config["unique_id"] = unique_id
    amino_acid = config["amino_acid"]
    config["torsion"] = True
    config["energy"] = False
    num_folds = config["num_folds"]
    batch_size = config["batch_size"]
    type = "torsion_decoder"
    if "mapping" in config["latent_path"]:
        type = "mapping_decoder"
    dataset = LatentEvalDataset(
        amino_acid=config["amino_acid"],
        data_path=config["data_path"],
        latent_path=config["latent_path"],
        fixed_O=True,
    )
    config["latent_dim"] = dataset.latent_dim

    name = f"{config['amino_acid']}_dim_{config['latent_dim']}"

    config["output_dim"] = dataset.num_atoms * 3
    print(config)

    if "synth/high_energy" in config["data_path"]:
        mode = "synth/high_energy"
    elif "synth" in config["data_path"]:
        mode = "synth"
    elif "full" in config["data_path"]:
        mode = "full"
    else:
        mode = ""

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Training fold {fold+1}/{num_folds}")
        wandb_logger = WandbLogger(
            name=name + f"_fold:{fold+1}",
            project=f"{type}",
            group=f"{amino_acid}_crossval_{unique_id}",
        )
        wandb_logger.experiment.config.update(config)

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        model = DecoderLightning(config, None)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config["num_workers"],
            persistent_workers=True if config["num_workers"] > 0 else False,
            pin_memory=False if sys.platform.startswith("win") else True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config["num_workers"],
            persistent_workers=True if config["num_workers"] > 0 else False,
            pin_memory=False if sys.platform.startswith("win") else True,
        )

        # Model initialization
        model = DecoderLightning(config, dataset)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_rmsd_epoch",
            mode="min",
            save_on_train_epoch_end=True,
            save_top_k=1,
            dirpath=f"./checkpoints/{mode}/{type}/{unique_id}/{name}//fold{fold+1}",
            filename=("best_{epoch}-{val_rmsd_epoch:.3f}"),
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        trainer = pl.Trainer(
            logger=wandb_logger,
            log_every_n_steps=10,
            max_epochs=config["epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else None,
            callbacks=[checkpoint_callback, lr_monitor],
            reload_dataloaders_every_n_epochs=0,
            check_val_every_n_epoch=1,
        )

        torch.set_float32_matmul_precision("high")
        torch.compile(model, mode="reduce-overhead")
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        best_model = DecoderLightning.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path
        )

        val_trainer = pl.Trainer(
            logger=False, accelerator=trainer.accelerator, devices=1
        )
        val_metrics = val_trainer.validate(best_model, val_loader)[0]
        fold_results.append(val_metrics)

        wandb.finish()

    all_metrics = {}
    for fold_result in fold_results:
        for key, value in fold_result.items():
            all_metrics.setdefault(key, []).append(value)

    results = {}
    for metric, values in all_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        results[metric] = {"mean": mean, "std": std}

    # Write to text file
    path = f"results/{mode}/{type}"
    os.makedirs(path, exist_ok=True)
    output_file = f"{path}/decoder_results_{amino_acid}_{unique_id}.txt"
    with open(output_file, "w") as f:
        f.write("Cross-Validation Metric Summary:\n")
        for metric, stats in results.items():
            f.write(f"{metric}:\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std:  {stats['std']:.4f}\n")
            f.write(f"  Values: {all_metrics[metric]}\n")
        f.write(str(config))


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
        "--num_folds", type=int, default=5, help="Number of training epochs."
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
        "--hidden_dims",
        type=str,
        default="[1024, 256, 128]",
        help="Hidden dimensions for the autoencoder.",
    )

    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")

    # Parse arguments
    args = parser.parse_args()

    # Load configuration
    config = load_config(args)
    config["hidden_dims"] = ast.literal_eval(config["hidden_dims"])
    # Run main with the loaded configuration
    run_cross_validation(config)
