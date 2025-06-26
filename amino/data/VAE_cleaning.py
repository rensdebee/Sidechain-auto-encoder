import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from amino.data.datasets import SidechainDataset
from amino.utils.utils import create_clean_path, get_model, write_pdb

if __name__ == "__main__":
    amino_acids = ["ARG", "GLN", "GLU", "LYS", "MET"]

    for amino in amino_acids:
        dataset = SidechainDataset(amino, data_path="dataset/clean_10", fixed_O=True)

        # Load model
        vae = get_model(amino, "checkpoints/VAE")
        vae.eval()
        vae.freeze()

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=8192,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
            pin_memory=False if sys.platform.startswith("win") else True,
        )

        total_loss = torch.empty((0,))
        with torch.no_grad():
            for batch in tqdm(data_loader):
                x = batch["sidechain_position"].flatten(1).cuda()
                x_hat, mean, logvar, z = vae(x)
                loss = F.mse_loss(x_hat, x, reduction="none").sum(dim=1).detach().cpu()
                total_loss = torch.cat((total_loss, loss))
        print(len(dataset), total_loss.shape)
        val, idxs = torch.topk(total_loss, k=int(len(dataset) * 0.01), sorted=False)
        mask = torch.ones((len(dataset)), dtype=torch.bool)
        mask[idxs] = False
        print(idxs.shape)
        sidechain = dataset.sidechain_positions[idxs]
        print(dataset.sidechain_positions[mask].shape)
        torch.save(dataset.sidechain_positions[mask], f"dataset/clean/{amino}.pt")
        print(sidechain.shape)
        create_clean_path(f"pdbs/weird_loss/{amino}")
        for i, struct in enumerate(sidechain):
            write_pdb(
                dataset.amino_acid,
                dataset.atom_order,
                struct,
                f"pdbs/weird_loss/{amino}/{i}.pdb",
            )
