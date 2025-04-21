
import torch
from torch import nn
import numpy as np
from models import ipa_pytorch

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

    def __init__(self, c_z, config):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.config = config 
        
        self.linear = ipa_pytorch.Linear(self.c_z, self.config.num_bins, init="glorot")
        self.softmax = nn.Softmax(dim=-1)

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
        probs = self.softmax(logits) 

        # breaks = np.linspace(self.config.first_break,
        #                         self.config.last_break,
        #                         self.config.num_bins-1)
        
        # bin_tops = np.append(breaks, breaks[-1] + (breaks[-1] + breaks[-2]))
        # bin_tops = torch.tensor(bin_tops)
        # threshold = 8 + 1e-3 # _CONTACT_THRESHOLD + _CONTACT_EPSILON
        # is_contact_bin = 1.0 * (bin_tops <= threshold)
        
        # contact_probs = torch.einsum(
        #     'ijk,k->ij', probs, is_contact_bin
        # )

        return probs
    
class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden, use_original_sm):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()

        self.c_hidden = c_hidden
        self.use_original_sm = use_original_sm

        if not self.use_original_sm:
            self.linear_1 = ipa_pytorch.Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = ipa_pytorch.Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_3 = ipa_pytorch.Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        s_initial = a

        if not self.use_original_sm:
            a = self.relu(a)
            a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)
        a = self.relu(a)
        a = self.linear_3(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon, use_original_sm):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
            use_original_sm:
                If True implement line 11 of algorithm 20 correctly else use the ABB3 implementation.
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon
        self.use_original_sm = use_original_sm

        if self.use_original_sm:
            self.linear_in = ipa_pytorch.Linear(self.c_in, self.c_hidden)
            self.linear_initial = ipa_pytorch.Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(
                c_hidden=self.c_hidden, use_original_sm=self.use_original_sm
            )
            self.layers.append(layer)

        self.linear_out = ipa_pytorch.Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor, s_initial: torch.Tensor
    ):
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        if self.use_original_sm:
            s_initial = self.relu(s_initial)
            s_initial = self.linear_initial(s_initial)
            s = self.relu(s)
            s = self.linear_in(s)
            s = s + s_initial
        else:
            s = torch.cat((s, s_initial), dim=-1)

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s