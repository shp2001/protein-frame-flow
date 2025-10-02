import torch
from torch import nn
from models.utils import get_time_embedding
from data import utils as du
from models.edge_feature_net import RelativePositionEncoding
from Protenix.protenix.model.modules.primitives import LayerNorm, LinearNoBias, Transition

class ConditioningModule(nn.Module):
    def __init__(self, model_conf):
        super(ConditioningModule, self).__init__()
        """
        Args:
            sigma_data (torch.float, optional): the standard deviation of the data. Defaults to 16.0.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_s_inputs (int, optional): input embedding dim from InputEmbedder. Defaults to 449.
            c_noise_embedding (int, optional): noise embedding dim. Defaults to 256.
        """

        self.c_z = model_conf.c_z
        self.c_s = model_conf.c_s
        self.c_s_inputs = model_conf.c_s_inputs
        self.c_noise_embedding = model_conf.c_noise_embedding

        # Line1-Line3:
        self.relpos_embedder = RelativePositionEncoding(
            r_max=model_conf.relpos.r_max,
            s_max=model_conf.relpos.s_max,
            c_z=model_conf.relpos.c_z,
            )
        self.layernorm_z = LayerNorm(2 * self.c_z, create_offset=False)
        self.linear_no_bias_z = LinearNoBias(
            in_features=2 * self.c_z, out_features=self.c_z, precision=torch.float32
        )
        # Line3-Line5:
        self.transition_z1 = Transition(c_in=self.c_z, n=2)
        self.transition_z2 = Transition(c_in=self.c_z, n=2)

        # Line6-Line7
        self.layernorm_s = LayerNorm(self.c_s + self.c_s_inputs, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s + self.c_s_inputs,
            out_features=self.c_s,
            precision=torch.float32,
        )

        # Line8-Line9
        self.layernorm_n = LayerNorm(self.c_noise_embedding, create_offset=False)
        self.linear_no_bias_n = LinearNoBias(
            in_features=self.c_noise_embedding,
            out_features=self.c_s,
            precision=torch.float32,
        )

        # Line10-Line12
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)

    def forward(
        self,
        t: torch.Tensor,
        asym_id:torch.Tensor,
        residue_index:torch.Tensor,
        entity_id:torch.Tensor,
        sym_id:torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        inplace_safe: bool = False,
        use_conditioning: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            trans_sc (torch.Tensor): trans vector from self conditioning 
            rotmats_sc (torch.Tensor): rotation matrix from self conditiong
            inplace_safe (bool): Whether it is safe to use inplace operations.
            use_conditioning (bool): Whether to drop the s/z embeddings.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: embeddings s and z
                - s (torch.Tensor): [..., N_sample, N_tokens, c_s]
                - z (torch.Tensor): [..., N_tokens, N_tokens, c_z]
        """
        if not use_conditioning:
            if inplace_safe:
                s_trunk *= 0
                z_trunk *= 0
            else:
                s_trunk = 0 * s_trunk
                z_trunk = 0 * z_trunk


        # Pair conditioning
        relative_position = self.relpos_embedder(
            asym_id,
            residue_index,
            entity_id,
            sym_id
        )
        pair_z = torch.cat(
            tensors=[z_trunk, relative_position], dim=-1
        )  # [..., N_tokens, N_tokens, 2*c_z]
        pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))
        if inplace_safe:
            pair_z += self.transition_z1(pair_z)
            pair_z += self.transition_z2(pair_z)
        else:
            pair_z = pair_z + self.transition_z1(pair_z)
            pair_z = pair_z + self.transition_z2(pair_z)
        
        # Single conditioning
        single_s = torch.cat(
            tensors=[s_trunk, s_inputs], dim=-1
        )  # [..., N_tokens, c_s + c_s_inputs]
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))

        time_embed = get_time_embedding(
            t[:, 0], 
            self.c_noise_embedding,
            2056
            )[:, None, :].repeat(1, single_s.shape[-2], 1).to(single_s.dtype)
        single_s = single_s + self.linear_no_bias_n(self.layernorm_n(time_embed))

        if inplace_safe:
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
        else:
            single_s = single_s + self.transition_s1(single_s)
            single_s = single_s + self.transition_s2(single_s)

        return single_s, pair_z