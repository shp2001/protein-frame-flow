import torch
from torch import nn

from models import ipa_pytorch
from Protenix.protenix.model.modules import transformer
from Protenix.protenix.model.modules.primitives import LayerNorm
from Proteus.model.ipa_pytorch import LocalTriangleAttentionNew

class StructureModuleBlock(nn.Module):
    def __init__(self, model_conf, update_pair):
        super().__init__()
        self._aa_enc_conf = model_conf.aa_enc
        self._ipa_conf = model_conf.ipa
        self.update_pair = update_pair
        self._local_triangle_attention_new_conf = model_conf.local_triangle_attention_new

        self.atom_attention_encoder = transformer.AtomAttentionEncoder(self._aa_enc_conf)
        self.ipa = ipa_pytorch.IPABlocks(self._ipa_conf,
                                            self._ipa_conf.depth)
        self.fuse_ln_aa_enc = LayerNorm(self._aa_enc_conf.c_token)
        self.fuse_ln_ipa = LayerNorm(self._ipa_conf.c_s)
        self.fuse_linear = ipa_pytorch.Linear(
            in_dim=self._ipa_conf.c_s + self._aa_enc_conf.c_token,
            out_dim=self._ipa_conf.c_s
            )
        tfmr_in = self._ipa_conf.c_s
        tfmr_layer = torch.nn.TransformerEncoderLayer(
            d_model=tfmr_in,
            nhead=self._ipa_conf.seq_tfmr_num_heads,
            dim_feedforward=tfmr_in,
            batch_first=True,
            dropout=0.0,
            norm_first=False
        )
        self.seq_tfmr = torch.nn.TransformerEncoder(
            tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
        self.post_tfmr = ipa_pytorch.Linear(
            tfmr_in, self._ipa_conf.c_s, init="final")
        self.node_transition = ipa_pytorch.StructureModuleTransition(
            c=self._ipa_conf.c_s)
        if self.update_pair:
            self.edge_update = LocalTriangleAttentionNew(**self._local_triangle_attention_new_conf)
        self.bb_update = ipa_pytorch.BackboneUpdate(
            self._ipa_conf.c_s, use_rot_updates=True)
    
    def forward(
        self,         
        s_single: torch.Tensor,
        z_pair: torch.Tensor,
        curr_rigids,
        ref_feature_dict,
        node_mask: torch.Tensor,
        diffuse_mask: torch.Tensor,
    ):

        a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
            input_feature_dict=ref_feature_dict,
            s=s_single,
            z=z_pair
        ) # [B, N_sample, N_token, c_token]
        a_token = a_token.to(dtype=torch.float32)

        # residue embed 
        ipa_embed = self.ipa(
            s_single,
            z_pair,
            curr_rigids,
            node_mask)
        ipa_embed = ipa_embed * node_mask[..., None]
        
        ipa_embed = self.fuse_ln_ipa(ipa_embed + s_single)
        a_token = self.fuse_ln_aa_enc(a_token)

        s = torch.cat([ipa_embed, a_token], dim=-1)
        s = self.fuse_linear(s)

        seq_tfmr_out = self.seq_tfmr(
            s, src_key_padding_mask=(1 - node_mask).to(torch.bool))
        s = s + self.post_tfmr(seq_tfmr_out)
        s = self.node_transition(s)
        s = s * node_mask[..., None]
        rigid_update = self.bb_update(
            s * node_mask[..., None])
        curr_rigids = curr_rigids.compose_q_update_vec(rigid_update)

        # pair update 
        if self.update_pair:
            edge_mask = node_mask[:, None] * node_mask[:, :, None]
            z_pair = self.edge_update(
                s, z_pair, curr_rigids, edge_mask
            )
            z_pair = z_pair * edge_mask[..., None]

        return curr_rigids, s, z_pair

class StructureModule(nn.Module):
    def __init__(self, model_conf):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                StructureModuleBlock(model_conf, pair_update=True)
                for _ in range(model_conf.n_blocks - 1)
            ] + [
                StructureModuleBlock(model_conf, pair_update=False)
            ]
        )

    def forward(
        self,
        s_single,
        z_pair,
        curr_rigids,  # (B, L)
        ref_feature_dict,
        node_mask: torch.Tensor,
        diffuse_mask: torch.Tensor
    ):
        for block in self.blocks:
            curr_rigids, s_single, z_pair = block(
                s_single, 
                z_pair, 
                curr_rigids, 
                ref_feature_dict, 
                node_mask, 
                diffuse_mask
                )
        return curr_rigids, s_single