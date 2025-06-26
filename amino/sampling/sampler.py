import math
from functools import partial

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from amino.utils.utils import get_model


def lr_lambda(current_step, constant_steps, total_steps):
    if current_step < constant_steps:
        # Constant phase
        return 1.0
    else:
        # Cosine decay phase
        decay_steps = total_steps - constant_steps
        progress = (current_step - constant_steps) / decay_steps
        return 0.5 * (1 + math.cos(math.pi * progress))


def sample_low_energy_hae(latent, amino, iter=500, lr=0.01, mode="full"):
    model = get_model(
        amino, f"checkpoints/{mode}/HAEdecoder/_ratio_0.75", dim=latent.shape[1]
    )
    model.eval()
    model.freeze()
    train_mean, train_std = model.mean.cuda(), model.std.cuda()
    energy_dim = model.energy_dim

    latent = latent.cuda()
    latent.requires_grad = True
    opt = torch.optim.AdamW([latent], lr)
    scheduler = LambdaLR(
        opt,
        lr_lambda=partial(lr_lambda, constant_steps=int(iter * 0.5), total_steps=iter),
    )

    steps = []
    e_steps = []
    for j in range(iter):
        # print(latent.norm())
        opt.zero_grad()
        output = model(latent)
        x_hat, e_hat = (
            output[:, :-energy_dim],
            output[:, -energy_dim:],
        )
        # print(e_hat)
        e_hat = e_hat * train_std + train_mean
        loss = e_hat.sum(dim=1).mean()
        loss.backward()

        opt.step()
        scheduler.step()
        steps.append(x_hat.unflatten(dim=1, sizes=(-1, 3)).detach().cpu())
        e_steps.append(e_hat.detach().cpu())
        with torch.no_grad():
            latent /= torch.maximum(
                latent.norm(p=2, dim=1, keepdim=True), torch.tensor(1e-6)
            )

    output = model(latent)
    x_hat, e_hat = (
        output[:, :-energy_dim],
        output[:, -energy_dim:],
    )
    steps.append(x_hat.unflatten(dim=1, sizes=(-1, 3)).detach().cpu())
    e_steps.append(e_hat.detach().cpu() * model.std + model.mean)
    return steps, e_steps


def sample_low_energy_torsion(angles, amino, iter=500, lr=0.005, mode="full"):
    model = get_model(amino, f"checkpoints/{mode}/torsion_energy")
    model.eval()
    model.freeze()

    decoder = get_model(amino, f"checkpoints/{mode}/torsion_decoder")
    decoder.eval()
    decoder.freeze()
    train_mean, train_std = model.mean.cuda(), model.std.cuda()
    steps = []
    e_steps = []

    angles = angles.cuda()
    angles.requires_grad = True
    opt = torch.optim.AdamW([angles], lr)
    scheduler = LambdaLR(
        opt,
        lr_lambda=partial(lr_lambda, constant_steps=int(iter * 0.5), total_steps=iter),
    )

    for j in range(iter):
        opt.zero_grad()
        sin_cos_2dim = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
                torch.sin(3 * angles),
                torch.cos(3 * angles),
            ],
            dim=1,
        )
        sin_cos_1dim = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=1,
        )

        e_hat = model(sin_cos_2dim)
        e_hat = e_hat * train_std + train_mean
        loss = e_hat.sum(dim=1).mean()
        loss.backward()
        x_hat = decoder(sin_cos_1dim).unflatten(dim=1, sizes=(-1, 3)).detach().cpu()
        steps.append(x_hat)
        e_steps.append(e_hat.detach().cpu())

        opt.step()
        scheduler.step()
    sin_cos_2dim = torch.cat(
        [
            torch.sin(angles),
            torch.cos(angles),
            torch.sin(3 * angles),
            torch.cos(3 * angles),
        ],
        dim=1,
    )
    sin_cos_1dim = torch.cat(
        [
            torch.sin(angles),
            torch.cos(angles),
        ],
        dim=1,
    )

    e_hat = model(sin_cos_2dim)
    x_hat = decoder(sin_cos_1dim).unflatten(dim=1, sizes=(-1, 3)).detach().cpu()
    steps.append(x_hat)
    e_steps.append(e_hat.detach().cpu() * model.std + model.mean)
    return steps, e_steps


def sample_low_energy_mapping(latent, amino, iter=500, lr=0.01, mode="full"):
    model = get_model(
        amino,
        f"checkpoints/{mode}/mapping_network/",
        dim=latent.flatten(start_dim=1).shape[1],
    )
    model.eval()
    model.freeze()
    train_mean, train_std = model.mean.cuda(), model.std.cuda()
    latent = latent.cuda()
    latent.requires_grad = True
    opt = torch.optim.AdamW([latent], lr)
    scheduler = LambdaLR(
        opt,
        lr_lambda=partial(lr_lambda, constant_steps=int(iter * 0.5), total_steps=iter),
    )

    steps = []
    e_steps = []
    for j in range(iter):
        # print(latent.norm())
        opt.zero_grad()
        pred_sin_cos, x_hat, e_hat = model.reconstruct(latent)
        # print(e_hat)
        e_hat = e_hat * train_std + train_mean
        loss = e_hat.sum(dim=1).mean()
        loss.backward()

        opt.step()
        scheduler.step()
        steps.append(x_hat.detach().cpu())
        e_steps.append(e_hat.detach().cpu())

    pred_sin_cos, x_hat, e_hat = model.reconstruct(latent)
    steps.append(x_hat.detach().cpu())
    e_steps.append(e_hat.detach().cpu() * model.std + model.mean)
    return steps, e_steps
