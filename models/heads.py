from typing import Optional, Union 
import torch
from torch import nn
import numpy as np
from models import ipa_pytorch
from models.utils import calc_distogram, calc_unit_vector
from Protenix.protenix.model.modules.primitives import LayerNorm, LinearNoBias
from Protenix.protenix.model.modules.pairformer import PairformerStack

class AAContactHead(nn.Module):
    """
    Computes an all-atom contact map. 
    """

    def __init__(self, c_z, **kwargs):
        """
        Computes the all atom contact map 
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(AAContactHead, self).__init__()

        self.c_z = c_z
        self.linear = ipa_pytorch.Linear(self.c_z, 105, init="glorot")
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z):  
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
    
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """

        # [*, N, N, no_bins]
        # keep symmetry 
        left_half = self.linear(z)
        right_half = left_half.transpose(-2, -3)
        logits = left_half + right_half # (*, N, N, 14)
        logits = self.sigmoid(logits)

        return logits

class DistogramHead(nn.Module):
    """
    Computes an all-atom contact map.
    """

    def __init__(self, c_z, num_bins):
        """
        Args:
            c_z:
                Input channel dimension
            num_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()
        
        self.linear = ipa_pytorch.Linear(c_z, num_bins, init="glorot")

    def forward(self, z):  
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
            
            atom14_gt_exists:
                [*, N_res, 14]
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """

        # [*, N, N, no_bins]
        # keep symmetry 
        left_half = self.linear(z)
        right_half = left_half.transpose(-2, -3)
        logits = left_half + right_half # (*, N, N, 14)


        return logits

class AllAtomModule(nn.Module):
    """All-atom update resnet module."""

    def __init__(self, d_single, d_hidden, n_block, atom_num):
        super().__init__()
        self.relu = nn.ReLU()
        self.linear_in = ipa_pytorch.Linear(d_single, d_hidden)
        self.linear_initial = ipa_pytorch.Linear(d_single, d_hidden)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    ipa_pytorch.Linear(d_hidden, d_hidden, init="relu"),
                    nn.ReLU(),
                    ipa_pytorch.Linear(d_hidden, d_hidden, init="final"),
                )
                for _ in range(n_block)
            ]
        )
        self.linear_proj = ipa_pytorch.Linear(d_hidden, atom_num * 3)

    def forward(
        self,
        single: torch.Tensor,  # (..., L, d_single)
        init_single: torch.Tensor,  # (..., L, d_single)
    ) -> torch.Tensor:  # (..., L, atom_num, 3)
        single = self.linear_in(self.relu(single))
        init_single = self.linear_initial(self.relu(init_single))
        single = single + init_single

        for block in self.blocks:
            single = single + block(single)
        single = self.linear_proj(self.relu(single))

        local_atom_pos = single.view(single.shape[:-1] + (-1, 3))
        return local_atom_pos

class SmallMLP(nn.Module):
    def __init__(self, c, num_bins):
        super(SmallMLP, self).__init__()

        self.c = c
        self.num_bins = num_bins

        self.linear_1 = LinearNoBias(self.c, self.c, initializer="relu")
        self.linear_2 = LinearNoBias(self.c, self.c, initializer="relu")
        self.linear_3 = LinearNoBias(self.c, self.num_bins)
        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s


class ConfidenceHead(nn.Module):
    """
    Implements Algorithm 31 in AF3
    """

    def __init__(
        self,
        n_blocks: int = 3,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 384,
        min_bins: int = 2.0,
        max_bins: int = 32.0,
        num_bins: int = 32,
        pairformer_dropout: float = 0.0,
        blocks_per_ckpt: Optional[int] = None,
        stop_gradient: bool = True,
    ) -> None:
        """
        Args:
            n_blocks (int, optional): number of blocks for ConfidenceHead. Defaults to 4.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s_inputs (int, optional): hidden dim [for single embedding from InputFeatureEmbedder]. Defaults to 449.
            max_atoms_per_token (int, optional): max atoms in a token. Defaults to 20.
            pairformer_dropout (float, optional): dropout ratio for Pairformer. Defaults to 0.0.
            blocks_per_ckpt: number of Pairformer blocks in each activation checkpoint
            min_bins (float, optional): Start of the distance bin range. Defaults to 2.0.
            max_bins (float, optional): End of the distance bin range. Defaults to 32.0.
            num_bins (float, optional): The number of the bins. Defaults to 32.0.
            stop_gradient (bool, optional): Whether to stop gradient propagation. Defaults to True.
        """
        super(ConfidenceHead, self).__init__()
        self.n_blocks = n_blocks
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.num_bins = num_bins
        self.stop_gradient = stop_gradient
        self.linear_no_bias_s1 = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )
        self.linear_no_bias_s2 = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )

        self.linear_no_bias_d = LinearNoBias(
            in_features=self.num_bins+3, out_features=self.c_z
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
        self.linear_no_bias_pae = LinearNoBias(
            in_features=self.c_z, out_features=self.b_pae
        )
        self.linear_no_bias_pde = LinearNoBias(
            in_features=self.c_z, out_features=self.b_pde
        )

        self.input_strunk_ln = LayerNorm(self.c_s)
        self.plddt_ln = LayerNorm(self.c_s)
        self.plddt_transition = SmallMLP(self.c_s, num_bins=self.num_bins)

        with torch.no_grad():
            # Zero init for output layer (before softmax) to zero
            nn.init.zeros_(self.linear_no_bias_pae.weight)
            nn.init.zeros_(self.linear_no_bias_pde.weight)

            # # Zero init for trunk embedding input layer
            # nn.init.zeros_(self.linear_no_bias_s_trunk.weight)
            # nn.init.zeros_(self.linear_no_bias_z_trunk.weight)

    def forward(
        self,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_mask: torch.Tensor,
        pred_rigids,
        use_embedding: bool = True,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            use_memory_efficient_kernel (bool, optional): Whether to use memory-efficient kernel. Defaults to False.
            use_deepspeed_evo_attention (bool, optional): Whether to use DeepSpeed evolutionary attention. Defaults to False.
            use_lma (bool, optional): Whether to use low-memory attention. Defaults to False.
            inplace_safe (bool, optional): Whether to use inplace operations. Defaults to False.
            chunk_size (Optional[int], optional): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - plddt_preds: Predicted pLDDT scores [..., N_sample, N_atom, plddt_bins].
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

        pred_trans = pred_rigids.get_trans()
        N_sample = pred_trans.size(-3)

        z_init = (
            self.linear_no_bias_s1(s_inputs)[..., None, :, :]
            + self.linear_no_bias_s2(s_inputs)[..., None, :]
        )
        z_trunk = z_init + z_trunk
        if not self.training:
            del z_init
            torch.cuda.empty_cache()

        plddt_preds = []
        for i in range(N_sample):
            plddt_pred = (
                self.memory_efficient_forward(
                    s_trunk=s_trunk.clone() if inplace_safe else s_trunk,
                    z_pair=z_trunk.clone() if inplace_safe else z_trunk,
                    pair_mask=pair_mask,
                    pred_rigids=pred_rigids,
                    use_memory_efficient_kernel=use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            )

            plddt_preds.append(plddt_pred)

        plddt_preds = torch.stack(
            plddt_preds, dim=-3
        )  # [..., N_sample, N_res, plddt_bins]

        return plddt_preds


    def memory_efficient_forward(
        self,
        s_trunk: torch.Tensor,
        z_pair: torch.Tensor,
        pair_mask: torch.Tensor,
        pred_rigids,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            ...
            pred_trans (torch.Tensor): predicted coordinates
                [..., N_res, 3] # Note: N_sample = 1 for avoiding CUDA OOM
        """

        pred_trans = pred_rigids.get_trans()
        # Embed pair distances of representative atoms:
        with torch.cuda.amp.autocast(enabled=False):
            pred_trans = pred_trans.to(torch.float32)
            
            pred_distogarm, distance_pred = calc_distogram(pred_trans, 
                                            min_bin=self.min_bins, 
                                            max_bin=self.max_bins, 
                                            num_bins=self.num_bins,
                                            return_dist=True)
            pred_unit_vector = calc_unit_vector(pred_rigids)
            pred_z = torch.cat([pred_distogarm, pred_unit_vector], axis=-1)

            z_pair = z_pair + self.linear_no_bias_d(pred_z)  # [..., N_res, N_res, c_z]

            z_pair = z_pair + self.linear_no_bias_d_wo_onehot(
                distance_pred.unsqueeze(dim=-1)
            )  # [..., N_res, N_res, c_z]

        # Line 4
        s_single, z_pair = self.pairformer_stack(
            s_trunk,
            z_pair,
            pair_mask,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        # Upcast after pairformer
        z_pair = z_pair.to(torch.float32)
        s_single = s_single.to(torch.float32)

        plddt_logit = self.plddt_transition(self.plddt_ln(s_single))

        if not self.training and z_pair.shape[-2] > 2000:
            torch.cuda.empty_cache()

        return plddt_logit