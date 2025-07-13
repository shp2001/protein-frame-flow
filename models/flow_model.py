from typing import Optional, Union 

import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models import ipa_pytorch
from models.utils import get_time_embedding
from models.heads import DistogramHead
from data import utils as du
from openfold.utils.tensor_utils import dict_multimap
from openfold.utils.rigid_utils import local_to_global
from Proteus.model.ipa_pytorch import LocalTriangleAttentionNew
from Protenix.protenix.model.modules import transformer, pairformer
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
        self.relpos_dim = model_conf.relpos_dim
        self.feat_dim = model_conf.feat_dim

        # Line1-Line3:
        self.linear_relpos = LinearNoBias(self.relpos_dim, self.c_z)
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
        pair_init:torch.Tensor,
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
        relative_position = self.linear_relpos(pair_init)
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
        print(f"timestep shape: {t.shape}")
        time_embed = get_time_embedding(t[:, 0], 
                                        self.c_noise_embedding,
                                        2056
                                        ).to(single_s.dtype)
        time_embed = self.linear_no_bias_n(
                    self.layernorm_n(time_embed)
                    )
        single_s = torch.cat(
            tensors=[s_trunk, s_inputs], dim=-1
        )  # [..., N_tokens, c_s + c_s_inputs]
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
        single_s = single_s + time_embed

        if inplace_safe:
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
        else:
            single_s = single_s + self.transition_s1(single_s)
            single_s = single_s + self.transition_s2(single_s)
        if not self.training and pair_z.shape[-2] > 2000:
            torch.cuda.empty_cache()
        return single_s, pair_z
    
class FlowModel(nn.Module):
    def __init__(self, 
                 model_conf, 
                 training,
                 train_confidence):
        super(FlowModel, self).__init__()
        self.training = training
        self.train_confidence = train_confidence
        self._model_conf = model_conf
        self._pairformer_conf = model_conf.pairformer
        self._distogram_conf = model_conf.distogram_head

        self._condition_conf = model_conf.conditioning_module
        
        self._aa_enc_conf = model_conf.aa_enc
        self._ipa_conf = model_conf.ipa
        self._local_triangle_attention_new_conf = model_conf.local_triangle_attention_new
        self._all_atom_conf = model_conf.all_atom
        
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self._model_conf.edge_embed_size, 
            out_features=self._model_conf.edge_embed_size
        )
        self.layernorm_z_cycle = LayerNorm(self._model_conf.edge_embed_size)

        # recyling layer 
        self.linear_no_bias_s = LinearNoBias(
            in_features=self._model_conf.node_embed_size,
            out_features=self._model_conf.node_embed_size 
        )
        self.layernorm_s = LayerNorm(self._model_conf.node_embed_size)

        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)
        
        # Pairformer
        self.pairformer = pairformer.PairformerStack(
            self._pairformer_conf
        )
        self.distogram_head = DistogramHead(c_z=self._distogram_conf.c_z, 
                                            num_bins=self._distogram_conf.num_bins)

        # Condition Module 
        self.condition = ConditioningModule(self._condition_conf)

        # Structure Module 
        self.atom_attention_encoder = transformer.AtomAttentionEncoder(self._aa_enc_conf)
        self.ipa = ipa_pytorch.IPABlocks(self._ipa_conf,
                                         self._ipa_conf.depth)
        self.fuse_ln_aa_enc = LayerNorm(self._aa_enc_conf.c_token)
        self.fuse_ln_ipa = LayerNorm(self._ipa_conf.c_s)
        self.fuse_linear = ipa_pytorch.Linear(in_dim=self._ipa_conf.c_s + self._aa_enc_conf.c_token,
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
        self.bb_update = ipa_pytorch.BackboneUpdate(
            self._ipa_conf.c_s, use_rot_updates=True)

        self.allatom_tfmr = torch.nn.TransformerEncoder(
            tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
        self.allatom_proj = ipa_pytorch.Linear(self._ipa_conf.c_s, out_dim=14*3)
    
    def get_pairformer_output(self, 
                              s_init: torch.Tensor,
                              z_init: torch.Tensor,
                              edge_mask: torch.Tensor,
                              N_cycle: int):


        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training
                and (not self.train_confidence)
                and cycle_no == (N_cycle - 1)
            ):
                z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))

                s, z = self.pairformer(
                    s=s,
                    z=z,
                    pair_mask=edge_mask,
                    use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=True,
                    use_lma=False
                )
        
        distogram_logit = self.distogram_head(z)

        return s_init, s, z, distogram_logit
    
    def get_structure_output(self,         
                            t: torch.Tensor,
                            pair_init:torch.Tensor,
                            s_inputs: torch.Tensor,
                            s_single: torch.Tensor,
                            z_pair: torch.Tensor,
                            curr_rigids,
                            ref_feature_dict):

        # Main trunk
        all_atom_outputs = []

        curr_rigids = self.rigids_ang_to_nm(curr_rigids)

        a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
            input_feature_dict=ref_feature_dict,
            s=s_single,
            z=z_pair
        ) # [B, N_sample, N_token, c_token]
        a_token = a_token.to(dtype=torch.float32)

        # residue embed 
        ipa_embed = self.trunk[f'ipa_{b}'](
            node_embed,
            edge_embed,
            curr_rigids,
            node_mask)
        ipa_embed = ipa_embed * node_mask[..., None]
        
        ipa_embed = self.trunk[f'fuse_ln_aa_enc_{b}'](ipa_embed + node_embed)
        a_token = self.trunk[f'fuse_ln_ipa_{b}'](a_token)

        node_embed = torch.cat([ipa_embed, a_token], dim=-1)
        node_embed = self.trunk[f'fuse_linear_{b}'](node_embed)

        seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
            node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
        node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
        node_embed = self.trunk[f'node_transition_{b}'](node_embed)
        node_embed = node_embed * node_mask[..., None]
        rigid_update = self.trunk[f'bb_update_{b}'](
            node_embed * node_mask[..., None])
        curr_rigids = curr_rigids.compose_q_update_vec(
            rigid_update, (node_mask * diffuse_mask)[..., None])

        curr_rigids_unscaled = self.rigids_nm_to_ang(curr_rigids)
        allatom_embed = self.trunk[f"allatom_tfmr_{b}"](node_embed)
        local_atom_pos_pred = self.trunk[f"allatom_proj_{b}"](allatom_embed)
        local_atom_pos_pred = local_atom_pos_pred.view(local_atom_pos_pred.shape[:-1] + (-1, 3))
        pred_xyz = local_to_global(curr_rigids_unscaled, local_atom_pos_pred)
        all_atom_preds = {
            "positions": pred_xyz
        }
        
        all_atom_outputs.append(all_atom_preds)

        pred_trans = curr_rigids_unscaled.get_trans()
        pred_rotmats = curr_rigids_unscaled.get_rots().get_rot_mats()

        all_atom_outputs = dict_multimap(torch.stack, all_atom_outputs)
        input_for_confidence = {
            'node_embed': node_embed,
            'edge_embed': edge_embed,
            'curr_rigids': curr_rigids_unscaled
        }

        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            'backb_frame': curr_rigids_unscaled,
            'local_atom_pos': local_atom_pos_pred,
            'all_atom_preds': all_atom_outputs,
            'input_for_confidence': input_for_confidence,
        }

    def forward(self,
               input_feats,
               N_cycle):
        
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        pair_init = input_feats['pair_init']
        aatype = input_feats['aatype']
        ref_feature_dict = input_feats['ref_feature_dict']

        # Initialize node and edge embeddings
        s_init = self.node_feature_net(
            diffuse_mask,
            aatype
        )

        if 'trans_sc' not in input_feats:
            trans_sc = du.manage_missing_batch(trans_t, mask=~diffuse_mask.bool())
        else:
            trans_sc = input_feats['trans_sc']

        if 'rotmats_sc' not in input_feats:
            rotmats_sc = du.manage_missing_batch(rotmats_t, mask=~diffuse_mask.bool())
        else:
            rotmats_sc = input_feats['rotmats_sc']

        z_init = self.edge_feature_net(
            trans_t,
            trans_sc,
            rotmats_t,
            rotmats_sc,
            edge_mask,
            diffuse_mask,
            pair_init,
            ref_feature_dict
        )

        # Pairformer 
        s_init, s_trunk, z_trunk, distogram_logit = self.get_pairformer_output(s_init,
                                                                               z_init,
                                                                               edge_mask,
                                                                               N_cycle)

        # Conditioning 
        s_single, z_pair = self.condition(t=input_feats['t'],
                                          pair_init=input_feats['pair_init'],
                                          s_trunk=s_trunk,
                                          z_trunk=z_trunk)
        
        # Initialize rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

class ConfidenceModel(nn.Module):
    def __init__(self, model_conf):
        super(ConfidenceModel, self).__init__()
        self._model_conf = model_conf
        self._prmsd_conf = model_conf.prmsd

        self.prmsd = nn.ModuleDict()
  
        self.prmsd_node_transform = ipa_pytorch.Linear(self._prmsd_conf.c_s, self._prmsd_conf.c_s)
        self.prmsd_edge_transform = ipa_pytorch.Linear(
                self._prmsd_conf.c_z,
                self._prmsd_conf.c_z
            )

        for b in range(self._prmsd_conf.num_blocks):
            self.prmsd[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._prmsd_conf)
            self.prmsd[f'ipa_ln_{b}'] = nn.LayerNorm(self._prmsd_conf.c_s)
            
            if b < self._prmsd_conf.num_blocks - 1:
                self.prmsd[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                    c=self._prmsd_conf.c_s)
            else:
                self.prmsd[f'prmsd_transition_{b}'] = ipa_pytorch.NodeTransition(
                    c=self._prmsd_conf.c_s, num_bins=self._prmsd_conf.num_bins
                )
    
    def forward(self, input_feats, node_mask): 
    # input feats is a dictionary which includes node_embed, edge_embed, curr_rigids, node_mask
        node_embed = input_feats['node_embed']
        edge_embed = input_feats['edge_embed']
        curr_rigids = input_feats['curr_rigids']
        prmsd_node = self.prmsd_node_transform(node_embed) 
        prmsd_edge = self.prmsd_edge_transform(edge_embed)

        for b in range(self._prmsd_conf.num_blocks):
            prmsd_ipa_embed = self.prmsd[f'ipa_{b}'](
                prmsd_node,
                prmsd_edge,
                curr_rigids,
                node_mask
            )
            # prmsd_ipa_embed *= node_mask[..., None]
            prmsd_node = self.prmsd[f'ipa_ln_{b}'](prmsd_node + prmsd_ipa_embed)
            
            if b < self._prmsd_conf.num_blocks - 1:
                prmsd_node = self.prmsd[f'node_transition_{b}'](prmsd_node)
                prmsd_node = prmsd_node * node_mask[..., None]
            else:
                prmsd_node = self.prmsd[f'prmsd_transition_{b}'](prmsd_node)

        return prmsd_node 