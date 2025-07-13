import math
import torch
from torch.nn import functional as F
import numpy as np
from data import utils as du

def calc_unit_vector(rigids, eps=1e-20):
    points = rigids.get_trans()[..., None, :, :]
    rigid_vec = rigids[..., None].invert_apply(points)

    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec**2, dim=-1))
    unit_vector = rigid_vec * inv_distance_scalar[..., None]
    unit_vector = torch.unbind(unit_vector[..., None, :], dim=-1)
    unit_vector = torch.cat(unit_vector, dim=-1)
    return unit_vector

def compute_distance_map(coords):

    coords_exp1 = coords.unsqueeze(2)  # (b, L, 1, 3)
    coords_exp2 = coords.unsqueeze(1)  # (b, 1, L, 3)

    dist_map = torch.norm(coords_exp1 - coords_exp2, dim=-1, keepdim=True)  # (b, L, L, 1)

    return dist_map

def calc_distogram(pos, min_bin, max_bin, num_bins):
    # pos: (b, L, 3)
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None] # (b, L, L, 1)
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram

def calc_all_atom_distogram(pos, min_bin, max_bin, num_bins):
    # pos: (b, L, 14, 3)
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :, :] - pos[:, None, :, :, :], axis=-1)[..., None] # (b, L, L, 14, 1)
    
    dists_2d = dists_2d.permute(0, 3, 1, 2, 4) # (b, 14, L, L, 1)
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram

def get_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> torch.Tensor:
    """
    Calculate the centers of the bins for a given range and number of bins.

    Args:
        min_bin (float): The minimum value of the bin range.
        max_bin (float): The maximum value of the bin range.
        no_bins (int): The number of bins.

    Returns:
        torch.Tensor: The centers of the bins.
            Shape: [no_bins]
    """
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = torch.linspace(
        start=min_bin,
        end=max_bin - bin_width,
        steps=no_bins,
    )
    bin_centers = boundaries + 0.5 * bin_width
    return bin_centers


def compute_contact_prob(
    distogram_logits: torch.Tensor,
    min_bin: float, # default 2.3125
    max_bin: float, # default 21.6875
    no_bins: int,
    thres=8.0,
) -> torch.Tensor:
    """
    Compute the contact probability from distogram logits.

    Args:
        distogram_logits (torch.Tensor): Logits for the distogram.
            Shape: [N_token, N_token, N_bins]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.
        thres (float): Threshold distance for contact probability. Defaults to 8.0.

    Returns:
        torch.Tensor: Contact probability.
            Shape: [N_token, N_token]
    """
    distogram_prob = torch.nn.functional.softmax(
        distogram_logits, dim=-1
    )  # [N_token, N_token, N_bins]
    distogram_bins = get_bin_centers(min_bin, max_bin, no_bins)
    thres_idx = (distogram_bins < thres).sum()
    contact_prob = distogram_prob[..., :thres_idx].sum(-1)
    return contact_prob

def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def t_stratified_loss(batch_t, batch_loss, num_bins=4, loss_name=None):
    """Stratify loss by binning t."""
    batch_t = du.to_numpy(batch_t)
    batch_loss = du.to_numpy(batch_loss)
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins+1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = 'loss'
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin+1]
        t_range = f'{loss_name} t=[{bin_start:.2f},{bin_end:.2f})'
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses
