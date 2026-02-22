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
        sigma: float = 10.0,
        pool_mutation: bool = False
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
        self.sigma = sigma
        self.pool_mutation = pool_mutation
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        self.stop_gradient = stop_gradient
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

        self.input_strunk_ln = LayerNorm(self.c_s)

        self.mut_emb_s = LinearNoBias(in_features=1, out_features=self.c_s)
        self.mut_emb_z = LinearNoBias(in_features=1, out_features=self.c_z)

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

        if pool_mutation:
            affinity_out_mlp_in = self.c_z * 2
        else:
            affinity_out_mlp_in = self.c_z

        self.norm_g = nn.LayerNorm(affinity_out_mlp_in)
        self.affinity_out_mlp = nn.Sequential(
            Linear(affinity_out_mlp_in, self.c_z, initializer='relu'),
            nn.ReLU(),
            Linear(self.c_z, self.c_z//2, initializer='relu'),
            nn.ReLU(),
            LinearNoBias(self.c_z//2, 1),
        )


    def forward(
        self,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        inter_pair_mask: torch.Tensor,
        x_pred_coords: torch.Tensor,
        mutation_mask: torch.Tensor,
        use_embedding: bool = True,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inter_pair_mask (torch.Tensor): inter-pair mask
                [..., N_token, N_token]
            x_pred_coords (torch.Tensor): predicted coordinates
                [..., N_atoms, 3]

        Returns:
            affinity_value (torch.Tensor): predicted affinity [..., 1]
        """

        if self.stop_gradient:
            s_inputs = s_inputs.detach()
            s_trunk = s_trunk.detach()
            z_trunk = z_trunk.detach()

        s_trunk = self.input_strunk_ln(s_trunk)

        if not use_embedding:
            if inplace_safe:
                z_trunk *= 0
            else:
                z_trunk = 0 * z_trunk

        z_init = (
            self.linear_no_bias_s1(s_inputs)[..., None, :, :]
            + self.linear_no_bias_s2(s_inputs)[..., None, :]
        )
        z_trunk = z_init + z_trunk
        if not self.training:
            del z_init
            torch.cuda.empty_cache()

        affinity_values = []
        x_pred_rep_coords = x_pred_coords

        N_sample = x_pred_rep_coords.size(-3)
        for i in range(N_sample):
            affinity_value = self.memory_efficient_forward(
                    s_trunk=s_trunk.clone() if inplace_safe else s_trunk,
                    z_pair=z_trunk.clone() if inplace_safe else z_trunk,
                    inter_pair_mask=inter_pair_mask,
                    x_pred_rep_coords=x_pred_coords[..., i, :, :],
                    mutation_mask=mutation_mask
            )
            affinity_values.append(affinity_value)
        affinity_values = torch.stack(affinity_values).squeeze(-1)
        return affinity_values

    def memory_efficient_forward(
        self,
        s_trunk: torch.Tensor,
        z_pair: torch.Tensor,
        inter_pair_mask: torch.Tensor,
        x_pred_rep_coords: torch.Tensor,
        mutation_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            ...
            x_pred_coords (torch.Tensor): predicted coordinates
                [..., N_atoms, 3] # Note: N_sample = 1 for avoiding CUDA OOM
        """
        mut_mask_expand = mutation_mask.to(s_trunk.dtype).unsqueeze(-1)
        s_trunk = s_trunk + self.mut_emb_s(mut_mask_expand)
        z_mut = self.mut_emb_z(mut_mask_expand)
        z_pair = z_pair + z_mut.unsqueeze(-2) + z_mut.unsqueeze(-3)
        
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
            inter_pair_mask
        )

        # Upcast after pairformer
        z_pair = z_pair.to(torch.float32) # (L, L, 128)

        # apply DistWeightedPooling 
        dist_w = torch.exp(-(distance_pred**2) / (self.sigma**2))[..., None]
        dist_w_inter = dist_w * inter_pair_mask[..., None]
        g = torch.sum(z_pair * dist_w_inter, dim=(0,1)) / torch.sum(dist_w_inter, dim=(0,1))

        # apply MutationWeightedPooling 
        if self.pool_mutation:
            mutation_mask_2d = (
                (mutation_mask.unsqueeze(-1) + mutation_mask.unsqueeze(-2)).clamp(max=1.0)
            )
            mutation_pooling_mask = dist_w_inter * mutation_mask_2d[..., None]
            g_mut = torch.sum(z_pair * mutation_pooling_mask, dim=(0, 1)) / torch.clamp(
                torch.sum(mutation_pooling_mask, dim=(0, 1)), min=1e-8
            )  # (c_z,)
            g = torch.cat([g, g_mut], dim=-1)  # (c_z * 2,)
        # # apply MeanPooling 
        # g = torch.sum(z_pair * inter_pair_mask[..., None], dim=(0,1)) / torch.sum(inter_pair_mask, dim=(0,1)) # (128)
        
        # Affinity MLP
        g = self.norm_g(g)
        affinity_pred_value = self.affinity_out_mlp(g) # (64)

        return affinity_pred_value