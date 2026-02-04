from typing import Optional, Union

import torch
import torch.nn as nn

from protenix.model.modules.pairformer import PairformerStack
from protenix.model.modules.primitives import LinearNoBias, Linear, BiasInitLinear
from protenix.model.utils import broadcast_token_to_atom, one_hot
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

        self.norm_g = nn.LayerNorm(self.c_z)

        self.affinity_out_mlp = nn.Sequential(
            Linear(self.c_z, self.c_z, initializer='relu'),
            nn.ReLU(),
            Linear(self.c_z, self.c_z//2, initializer='relu'),
            nn.ReLU()
        )

        self.to_affinity_pred_value = nn.Sequential(
            Linear(self.c_z//2, self.c_z//4, initializer='relu'),
            nn.ReLU(),
            Linear(self.c_z//4, self.c_z//4, initializer='relu'),
            nn.ReLU(),
            LinearNoBias(self.c_z//4, 1),
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
        z_trunk: torch.Tensor,
        inter_pair_mask: torch.Tensor,
        x_pred_coords: torch.Tensor,
        use_embedding: bool = True,
        inplace_safe: bool = False,
        use_coords: bool = True,
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

        s_trunk = self.input_strunk_ln(torch.clamp(s_trunk, min=-512, max=512))

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

        affinity_values, affinity_logits = (
            [],
            []
        )
        x_pred_rep_coords = x_pred_coords

        if use_coords:
            N_sample = x_pred_rep_coords.size(-3)
            for i in range(N_sample):
                affinity_value, affinity_logit = self.memory_efficient_forward(
                        s_trunk=s_trunk.clone() if inplace_safe else s_trunk,
                        z_pair=z_trunk.clone() if inplace_safe else z_trunk,
                        inter_pair_mask=inter_pair_mask,
                        x_pred_rep_coords=x_pred_coords[..., i, :, :],
                        use_coords=use_coords,
                )
                affinity_values.append(affinity_value)
                affinity_logits.append(affinity_logit)

        else:
            affinity_value, affinity_logit = self.memory_efficient_forward(
                    s_trunk=s_trunk.clone() if inplace_safe else s_trunk,
                    z_pair=z_trunk.clone() if inplace_safe else z_trunk,
                    inter_pair_mask=inter_pair_mask,
                    x_pred_rep_coords=x_pred_coords,
                    use_coords=use_coords,
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
        inter_pair_mask: torch.Tensor,
        x_pred_rep_coords: torch.Tensor,
        use_coords: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            ...
            x_pred_coords (torch.Tensor): predicted coordinates
                [..., N_atoms, 3] # Note: N_sample = 1 for avoiding CUDA OOM
        """
        
        if use_coords: 
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
        else:
            z_pair = z_pair 

        # pairformer w/ inter pair mask (attention to off-diagonal part of the pair feature)
        s_single, z_pair = self.pairformer_stack(
            s_trunk,
            z_pair,
            inter_pair_mask
        )

        # Upcast after pairformer
        z_pair = z_pair.to(torch.float32) # (L, L, 128)
        # apply MeanPooling 
        g = torch.sum(z_pair * inter_pair_mask[..., None], dim=(0,1)) / torch.sum(inter_pair_mask, dim=(0,1)) # (128)
        g = self.norm_g(g)
        
        # Affinity MLP 
        g = self.affinity_out_mlp(g) # (64)
        
        affinity_pred_value = self.to_affinity_pred_value(g)
        affinity_pred_score = self.to_affinity_pred_score(g)
        affinity_logits_binary = self.to_affinity_logits_binary(affinity_pred_score)

        return affinity_pred_value, affinity_logits_binary