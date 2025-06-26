import torch
import torch.nn.functional as F


def recon_regu(pred_batch, target_batch):
    pred_batch = pred_batch.flatten(start_dim=1)
    target_batch = target_batch.flatten(start_dim=1)
    pred_distance = torch.cdist(pred_batch, pred_batch, p=2)
    target_distance = torch.cdist(target_batch, target_batch, p=2)
    loss = F.mse_loss(pred_distance, target_distance)
    return loss


def normalize_latent(x):
    return F.normalize(x, p=2, dim=-1)


def distance_matrix_loss(pred_coords, true_coords):
    B, N, _ = pred_coords.shape

    # Precompute upper triangle indices (i < j) - calculated once per N
    i, j = torch.triu_indices(N, N, offset=1, device=pred_coords.device)

    # Calculate squared distances directly (more numerically stable)
    pred_diff = pred_coords[:, :, None] - pred_coords[:, None, :]  # [B, N, N, 3]
    true_diff = true_coords[:, :, None] - true_coords[:, None, :]

    # squared distance calculation
    pred_d2 = (pred_diff**2).sum(dim=-1)
    true_d2 = (true_diff**2).sum(dim=-1)

    # Select upper triangle elements using precomputed indices
    pred_upper = pred_d2[:, i, j]  # [B, M]
    true_upper = true_d2[:, i, j]  # [B, M]

    # Calculate RMSD on squared distances (avoids sqrt backprop)
    return F.mse_loss(pred_upper.sqrt(), true_upper.sqrt())


def hypersphere_combined_loss(
    latents,
    rmsd_matrix=None,
    temp_uniform=10.0,
    sigma_similar=1.2,
    lambda_similar=0.5,
):
    batch_size = latents.size(0)

    # 1. Uniformity Loss (log-sum-exp over pairwise dot products)
    if latents.dim() == 2:
        latents = latents.unsqueeze(1)  # Convert [B, d] → [B, 1, d]

    B, K, d = latents.shape
    device = latents.device

    # Compute pairwise dot products: [K, B, B]
    dots = torch.matmul(
        latents.transpose(0, 1), latents.transpose(0, 1).transpose(1, 2)
    )  # [K, B, B]

    # Mask out self-dot products
    mask = (
        ~torch.eye(B, dtype=torch.bool, device=device).unsqueeze(0).expand(K, B, B)
    )  # [1, B, B]
    dots_masked = dots[mask].view(K, B, B - 1)  # [K, B, B-1]

    # Compute loss
    uniformity_loss = (
        torch.logsumexp(temp_uniform * dots_masked, dim=-1).mean(dim=-1).mean()
    )

    if rmsd_matrix is None:
        return uniformity_loss
    # 2. Similarity Loss (pull low-RMSD pairs closer)
    dots_similar = torch.mm(latents, latents.t())  # [B, B]

    # Convert RMSD to similarity weights (Gaussian kernel)
    weights = torch.exp(-(rmsd_matrix**2) / (sigma_similar**2))  # [B, B]
    weights = weights * (
        1 - torch.eye(batch_size, device=latents.device)
    )  # ignore diagonal

    # Penalize dissimilarity between structurally similar pairs
    similarity_loss = (weights * (1 - dots_similar)).sum() / (
        batch_size * (batch_size - 1)
    )

    return (1 - lambda_similar) * uniformity_loss + lambda_similar * similarity_loss


def compute_rmsd_matrix(coords_batch):
    # Compute pairwise squared differences using broadcasting
    diffs = coords_batch[:, None, :, :] - coords_batch[None, :, :, :]  # [B, B, N, 3]

    # Sum squared coordinates, average over atoms, then sqrt
    sq_dists = torch.sum(diffs**2, dim=-1)  # [B, B, N]
    rmsd_matrix = torch.sqrt(torch.mean(sq_dists, dim=-1))  # [B, B]
    return rmsd_matrix


def calculate_3D_squared_distance(pred, target):
    diff = pred - target
    sd = torch.sum(diff * diff, dim=-1).detach()
    return sd.flatten()
