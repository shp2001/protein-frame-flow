from typing import Optional, Union

import torch
import torch.nn as nn

from protenix.model.modules.pairformer import PairformerStack
from protenix.model.modules.primitives import LinearNoBias, Linear
from protenix.model.utils import one_hot
from protenix.openfold_local.model.primitives import LayerNorm


class AffinityHead(nn.Module):
    """
    Implements Algorithm 31 in AF3
    """

    def __init__(
        self,
        n_blocks: int = 4,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        pairformer_dropout: float = 0.0,
        blocks_per_ckpt: Optional[int] = None,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
        stop_gradient: bool = True,
        sigma=8.0,
    ) -> None:
        """
        Args:
            n_blocks (int, optional): number of blocks for ConfidenceHead. Defaults to 4.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s_inputs (int, optional): hidden dim [for single embedding from InputFeatureEmbedder]. Defaults to 449.
            pairformer_dropout (float, optional): dropout ratio for Pairformer. Defaults to 0.0.
            blocks_per_ckpt: number of Pairformer blocks in each activation checkpoint
            distance_bin_start (float, optional): Start of the distance bin range. Defaults to 3.25.
            distance_bin_end (float, optional): End of the distance bin range. Defaults to 52.0.
            distance_bin_step (float, optional): Step size for the distance bins. Defaults to 1.25.
            stop_gradient (bool, optional): Whether to stop gradient propagation. Defaults to True.
        """
        super(AffinityHead, self).__init__()
        self.n_blocks = n_blocks
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        self.stop_gradient = stop_gradient
        self.sigma = sigma

        # processing z
        self.linear_no_bias_z_intra = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_z_inter = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.input_ztrunk_ln = LayerNorm(self.c_z)
        self.linear_no_bias_z = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        
        self.linear_no_bias_s1 = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )
        self.linear_no_bias_s2 = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )
        lower_bins = torch.arange(
            distance_bin_start, distance_bin_end, distance_bin_step
        )
        upper_bins = torch.cat([lower_bins[1:], lower_bins.new_tensor([1e6])], dim=-1)
        self.lower_bins = nn.Parameter(lower_bins, requires_grad=False)
        self.upper_bins = nn.Parameter(upper_bins, requires_grad=False)
        self.num_bins = len(lower_bins)  # + 1

        self.linear_no_bias_d = LinearNoBias(
            in_features=self.num_bins, out_features=self.c_z
        )
        self.linear_no_bias_d_wo_onehot = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.pairformer_stack = PairformerStack(
            c_z=self.c_z,
            c_s=self.c_s,
            n_blocks=n_blocks,
            dropout=pairformer_dropout,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        self.affinity_out_mlp = nn.Sequential(
            Linear(self.c_z * 2, self.c_z, initializer='relu'),
            nn.ReLU(),
            Linear(self.c_z, self.c_z//2, initializer='relu'),
            nn.ReLU()
        )

        self.to_affinity_pred_value = nn.Sequential(
            Linear(self.c_z//2, self.c_z//4, initializer='relu'),
            nn.ReLU(),
            LinearNoBias(self.c_z//4, 1, initializer='zeros'),
        )

        self.to_affinity_pred_score = nn.Sequential(
            Linear(self.c_z//2, self.c_z//2, initializer='relu'),
            nn.ReLU(),
            Linear(self.c_z//2, self.c_z//2, initializer='relu'),
            nn.ReLU(),
            Linear(self.c_z//2, 1),
        )

        self.to_affinity_logits_binary = Linear(1, 1)

    def forward(
        self,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_bound: torch.Tensor,
        z_unbound: torch.Tensor,
        inter_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        x_pred_coords: torch.Tensor,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.stop_gradient:
            s_inputs = s_inputs.detach()
            s_trunk = s_trunk.detach()
            z_bound = z_bound.detach()
            z_unbound = z_unbound.detach()
        
        z_delta = z_bound - z_unbound # [L, L, 128]
        inter_mask = inter_mask[..., None]
        intra_mask = 1.0 - inter_mask

        inter_out = self.linear_no_bias_z_inter(z_delta * inter_mask) * inter_mask
        intra_out = self.linear_no_bias_z_intra(z_delta * intra_mask) * intra_mask

        z_trunk = inter_out + intra_out
        z_trunk = self.linear_no_bias_z(self.input_ztrunk_ln(z_trunk))

        z_init = (
            self.linear_no_bias_s1(s_inputs)[..., None, :, :]
            + self.linear_no_bias_s2(s_inputs)[..., None, :]
        )
        z_trunk = z_init + z_trunk

        if not self.training:
            del z_init
            torch.cuda.empty_cache()

        affinity_values, affinity_logits = (
            [],
            []
        )
        x_pred_rep_coords = x_pred_coords
        N_sample = x_pred_rep_coords.size(-3)
        for i in range(N_sample):
            affinity_value, affinity_logit = self.memory_efficient_forward(
                    s_trunk=s_trunk.clone() if inplace_safe else s_trunk,
                    z_pair=z_trunk.clone() if inplace_safe else z_trunk,
                    z_delta=z_delta.clone() if inplace_safe else z_delta,
                    edge_mask=edge_mask,
                    inter_mask=inter_mask,
                    x_pred_rep_coords=x_pred_coords[..., i, :, :],
            )
            affinity_values.append(affinity_value)
            affinity_logits.append(affinity_logit)

        affinity_values = torch.stack(affinity_values).squeeze(-1)
        affinity_logits = torch.stack(affinity_logits).squeeze(-1)

        return affinity_values, affinity_logits

    def memory_efficient_forward(
        self,
        s_trunk: torch.Tensor,
        z_pair: torch.Tensor,
        z_delta: torch.Tensor,
        edge_mask: torch.Tensor,
        inter_mask: torch.Tensor,
        x_pred_rep_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            ...
            x_pred_coords (torch.Tensor): predicted coordinates
                [..., N_atoms, 3] # Note: N_sample = 1 for avoiding CUDA OOM
        """
        
        # Embed pair distances of representative atoms:
        with torch.amp.autocast("cuda", enabled=False):
            x_pred_rep_coords = x_pred_rep_coords.to(torch.float32)
            distance_pred = torch.cdist(
                x_pred_rep_coords, x_pred_rep_coords
            )  # [..., N_tokens, N_tokens]
        z_pair = z_pair + self.linear_no_bias_d(
            one_hot(
                x=distance_pred,
                lower_bins=self.lower_bins,
                upper_bins=self.upper_bins,
            )
        )  # [..., N_tokens, N_tokens, c_z]

        z_pair = z_pair + self.linear_no_bias_d_wo_onehot(
            distance_pred.unsqueeze(dim=-1)
        )  # [..., N_tokens, N_tokens, c_z]


        # pairformer w/ inter pair mask (attention to off-diagonal part of the pair feature)
        s_single, z_pair = self.pairformer_stack(
            s_trunk,
            z_pair,
            edge_mask
        )

        # Upcast after pairformer
        z_pair = z_pair.to(torch.float32) # (L, L, 128)

        # [Inter: Distance-weighted pooling]
        mask = edge_mask[..., None] # (L, L, 1)
        dist_w = torch.exp(-(distance_pred**2) / (self.sigma**2))[..., None] # (L, L, 1)
        dist_w_inter = dist_w * inter_mask * mask
        g_inter = torch.sum(z_pair * dist_w_inter, dim=(0,1)) / torch.sum(dist_w_inter, dim=(0,1))

        # [Intra: Delta z weightd pooling]
        intra_mask = 1.0 - inter_mask
        delta_w = torch.linalg.vector_norm(z_delta, ord=2, dim=-1, keepdim=True)
        delta_w_intra = delta_w * intra_mask * mask
        g_intra = torch.sum(z_pair * delta_w_intra, dim=(0,1)) / torch.sum(delta_w_intra, dim=(0,1))

        # Final Output
        g = torch.cat([g_inter, g_intra], dim=-1)
        g = self.affinity_out_mlp(g) # (64)

        affinity_pred_value = self.to_affinity_pred_value(g)
        affinity_pred_score = self.to_affinity_pred_score(g)
        affinity_logits_binary = self.to_affinity_logits_binary(affinity_pred_score)

        return affinity_pred_value, affinity_logits_binary