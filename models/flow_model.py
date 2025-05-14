
import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models import ipa_pytorch
from data import utils as du
from openfold.utils.tensor_utils import dict_multimap
from openfold.utils.rigid_utils import local_to_global
from Proteus.model.ipa_pytorch import LocalTriangleAttentionNew
from models.heads import AAContactHead, DistogramHead, AngleResnet

class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self._local_triangle_attention_new_conf = model_conf.local_triangle_attention_new
        self._angle_conf = model_conf.angle
        self._distogram_conf = model_conf.distogram_head
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
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
            if b < self._ipa_conf.num_blocks-1:
                if self._local_triangle_attention_new_conf.enable:
                    self.trunk[f'edge_transition_{b}'] = LocalTriangleAttentionNew(**self._local_triangle_attention_new_conf)
                    if b != self._ipa_conf.num_blocks-2:
                        self.trunk[f'distogram_head_{b}'] = DistogramHead(self._ipa_conf.c_z,
                                                                          self._distogram_conf)
                    else:
                        self.trunk[f'aa_contact_head_{b}'] = AAContactHead(self._ipa_conf.c_z)
                else:
                    edge_in = self._model_conf.edge_embed_size
                    self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                        node_embed_size=self._ipa_conf.c_s,
                        edge_embed_in=edge_in,
                        edge_embed_out=self._model_conf.edge_embed_size,
                    )


        self.angle_resnet = AngleResnet(
                self._ipa_conf.c_s,
                self._angle_conf.c_resnet,
                self._angle_conf.no_resnet_blocks,
                self._angle_conf.no_angles,
                self._angle_conf.epsilon,
                self._angle_conf.use_original_sm
            )
            
    def forward(self, input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        r3_t = input_feats['r3_t']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        pair_init = input_feats['pair_init']
        aatype = input_feats['aatype']

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            r3_t,
            node_mask,
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

        init_edge_embed = self.edge_feature_net(
            trans_t,
            trans_sc,
            rotmats_t,
            rotmats_sc,
            edge_mask,
            diffuse_mask,
            pair_init
        )

        # Initialize rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        all_atom_outputs = []
        pair_outputs = []


        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]

        for b in range(self._ipa_conf.num_blocks):
            cb_distogram = None
            all_atom_contact_map = None 

            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed = ipa_embed * node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None])
            if b < self._ipa_conf.num_blocks-1:
                if self._local_triangle_attention_new_conf.enable:
                    edge_embed = self.trunk[f'edge_transition_{b}'](
                        node_embed, edge_embed, curr_rigids, edge_mask
                    )
                    edge_embed = edge_embed * edge_mask[..., None]
    
                else:
                    edge_embed = self.trunk[f'edge_transition_{b}'](
                        node_embed, edge_embed)
                    edge_embed = edge_embed * edge_mask[..., None]

                if b < self._ipa_conf.num_blocks-2:
                    cb_distogram = self.trunk[f'distogram_head_{b}'](edge_embed)
                
                else:
                    all_atom_contact_map = self.trunk[f'aa_contact_head_{b}'](edge_embed)

            if cb_distogram != None:
                pair_outputs.append(cb_distogram)

            if all_atom_contact_map != None:
                pair_outputs.append(all_atom_contact_map)
                
            curr_rigids = self.rigids_nm_to_ang(curr_rigids)
            local_atom_pos_pred = self.allatom_module(node_embed, init_node_embed)
            pred_xyz = local_to_global(curr_rigids, local_atom_pos_pred)

            all_atom_preds = {
                "positions": pred_xyz
            }

            all_atom_outputs.append(all_atom_preds)

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

        all_atom_outputs = dict_multimap(torch.stack, all_atom_outputs)
        input_for_confidence = {
            'node_embed': node_embed,
            'edge_embed': edge_embed,
            'curr_rigids': curr_rigids
        }
        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            'all_atom_preds': all_atom_outputs,
            'input_for_confidence': input_for_confidence,
            'pair_outputs': pair_outputs # b-1개의 pair 기반 output (b-2개는 beta carbon distogram, 마지막은 all atom contact map)
        }
    

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
                self.prmsd[f'prmsd_transition_{b}'] = ipa_pytorch.pRMSDTransition(
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