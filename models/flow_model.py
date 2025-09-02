
import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models.heads import AAContactHead, DistogramHead, AllAtomModule
from models.utils import calc_distogram, calc_unit_vector
from models import ipa_pytorch
from data import utils as du
from openfold.utils.tensor_utils import dict_multimap
from openfold.utils.rigid_utils import local_to_global
from Proteus.model.ipa_pytorch import LocalTriangleAttentionNew

from Protenix.protenix.model.modules import transformer

class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._aa_enc_conf = model_conf.aa_enc
        self._ipa_conf = model_conf.ipa
        self._local_triangle_attention_new_conf = model_conf.local_triangle_attention_new
        self._all_atom_conf = model_conf.all_atom
        self._distogram_conf = model_conf.distogram_head
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._model_conf.num_blocks):
            self.trunk[f"atom_attention_encoder_{b}"] = transformer.AtomAttentionEncoder(self._aa_enc_conf)
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'fuse_ln_aa_enc_{b}'] = nn.LayerNorm(self._aa_enc_conf.c_token)
            self.trunk[f'fuse_ln_ipa_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            self.trunk[f"fuse_linear_{b}"] = ipa_pytorch.Linear(in_dim=self._ipa_conf.c_s + self._aa_enc_conf.c_token,
                                                                out_dim=self._ipa_conf.c_s)
            # self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            # 0, 1, 2
            if b < self._model_conf.num_blocks-1:
                if self._local_triangle_attention_new_conf.enable:
                    self.trunk[f'edge_transition_{b}'] = LocalTriangleAttentionNew(**self._local_triangle_attention_new_conf)
                    if self._distogram_conf.use_pair_head:
                        self.trunk[f'distogram_head_{b}'] = DistogramHead(
                            self._ipa_conf.c_z,
                            self._distogram_conf.num_bins
                            )

                else:
                    edge_in = self._model_conf.edge_embed_size
                    self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                        node_embed_size=self._ipa_conf.c_s,
                        edge_embed_in=edge_in,
                        edge_embed_out=self._model_conf.edge_embed_size,
                    )


            self.trunk[f"allatom_module_{b}"] = AllAtomModule(
                    self._all_atom_conf.d_single,
                    self._all_atom_conf.d_hidden,
                    self._all_atom_conf.n_blocks,
                    self._all_atom_conf.atom_num,
                )
            

    def forward(self, input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        loop_mask = input_feats['loop_mask']
        r3_t = input_feats['r3_t']
        trans_t = input_feats['trans_t']
        trans_template = input_feats['trans_template']
        rotmats_t = input_feats['rotmats_t']
        rotmats_template = input_feats['rotmats_template']
        pair_init = input_feats['pair_init']
        aatype = input_feats['aatype']
        ag_hotspot = input_feats['ag_hotspot']
        ref_feature_dict = input_feats['ref_feature_dict']

        # Initialize node and edge embeddings
        node_embed = self.node_feature_net(
            r3_t,
            node_mask,
            diffuse_mask,
            loop_mask,
            aatype,
            ag_hotspot
        )

        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']

        if 'rotmats_sc' not in input_feats:
            rotmats_sc = torch.zeros_like(rotmats_t)
        else:
            rotmats_sc = input_feats['rotmats_sc']

        edge_embed = self.edge_feature_net(
            trans_template,
            trans_sc,
            rotmats_template,
            rotmats_sc,
            edge_mask,
            diffuse_mask,
            loop_mask,
            pair_init,
            ref_feature_dict
        )

        # Initialize rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        all_atom_outputs = []
        pair_outputs = []

        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        node_embed = node_embed * node_mask[..., None]
        edge_embed = edge_embed * edge_mask[..., None]

        for b in range(self._model_conf.num_blocks):
            cb_distogram = None
            init_node_embed = node_embed
            
            # atom embed 
            a_token, q_skip, c_skip, p_skip = self.trunk[f"atom_attention_encoder_{b}"](
                input_feature_dict=ref_feature_dict,
                s=node_embed,
                z=edge_embed
            ) # [B, N_sample, N_token, c_token]
            a_token = a_token * node_mask[..., None]

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
            
            if b < self._model_conf.num_blocks-1:
                if self._local_triangle_attention_new_conf.enable:
                    curr_rigids_unscale = self.rigids_nm_to_ang(curr_rigids)
                    edge_embed = self.trunk[f'edge_transition_{b}'](
                        node_embed, edge_embed, curr_rigids_unscale, edge_mask
                    )
                    edge_embed = edge_embed * edge_mask[..., None]
    
                else:
                    edge_embed = self.trunk[f'edge_transition_{b}'](
                        node_embed, edge_embed)
                    edge_embed = edge_embed * edge_mask[..., None]
                if self._distogram_conf.use_pair_head:
                    cb_distogram = self.trunk[f'distogram_head_{b}'](edge_embed)
                    pair_outputs.append(cb_distogram)

                
            curr_rigids_unscaled = self.rigids_nm_to_ang(curr_rigids)
            local_atom_pos_pred = self.trunk[f"allatom_module_{b}"](node_embed, init_node_embed)
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
            'pair_outputs': pair_outputs
        }
    

class ConfidenceModel(nn.Module):
    def __init__(self, model_conf):
        super(ConfidenceModel, self).__init__()
        self._model_conf = model_conf
        self._aa_enc_conf = model_conf.aa_enc
        self._ipa_conf = model_conf.ipa
        self._local_triangle_attention_new_conf = model_conf.local_triangle_attention_new
        self._all_atom_conf = model_conf.all_atom
        self._distogram_conf = model_conf.distogram_head
        self._confidence_head = model_conf.confidence_head
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 

        self.s_transform = ipa_pytorch.Linear(self._confidence_head.c_s, self._confidence_head.c_s)
        self.str_2_pair = ipa_pytorch.Linear(self._distogram_conf.num_bins + 3, self._confidence_head.c_z)

        self.edge_transition = LocalTriangleAttentionNew(**self._local_triangle_attention_new_conf)
        self.distogram_error_head = DistogramHead(
            self._confidence_head.c_z,
            self._confidence_head.num_bins
            )


    def forward(self, input_feats, node_mask):
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        # Initialize node and edge embeddings
        node_embed = input_feats['node_embed']
        edge_embed = input_feats['edge_embed']
        curr_rigids = input_feats['curr_rigids']

        # transform single feature 
        node_embed = self.s_transform(node_embed)
        
        # transform pair feature
        pred_trans = curr_rigids.get_trans()
        pred_distogram = calc_distogram(
            pos=pred_trans, 
            min_bin=self._distogram_conf.min_bin,
            max_bin=self._distogram_conf.max_bin,
            num_bins=self._distogram_conf.num_bins
            )
        pred_unit_vector = calc_unit_vector(
            rigids=curr_rigids
        )
        edge_embed = edge_embed + self.str_2_pair(torch.concat([pred_distogram, pred_unit_vector], dim=-1))

        # apply local triangle 
        edge_embed = self.edge_transition(
            node_embed, edge_embed, curr_rigids, edge_mask
        )
        
        pde = self.distogram_error_head(edge_embed) # softmax 적용

        plddt_logit = None
        # # plddt 
        # curr_rigids_scaled = self.rigids_ang_to_nm(curr_rigids)
        # node_embed = self.ipa(
        #     node_embed,
        #     edge_embed,
        #     curr_rigids_scaled,
        #     node_mask)
        # node_embed = node_embed * node_mask[..., None]
        
        # seq_tfmr_out = self.seq_tfmr(
        #     node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
        # node_embed = node_embed + self.post_tfmr(seq_tfmr_out)
        # plddt_logit = self.plddt_head(node_embed) # softmax 미적용

        return plddt_logit, pde
