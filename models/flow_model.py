
import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models.heads import AAContactHead, DistogramHead, AllAtomModule
from models.utils import calc_distogram, calc_unit_vector
from models import ipa_pytorch
from data import utils as du

from openfold.utils.tensor_utils import dict_multimap
from openfold.utils.rigid_utils import Rigid
from Proteus.model.ipa_pytorch import LocalTriangleAttentionNew

from Protenix.protenix.model.modules import transformer
from Protenix.protenix.openfold_local.model.primitives import LayerNorm
from Protenix.protenix.model.modules.primitives import LinearNoBias, Transition

class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._aa_enc_conf = model_conf.aa_enc
        self._diffusion_tfmr = model_conf.diffusion_transformer
        self._aa_dec_conf = model_conf.aa_decoder
        self._local_triangle_attention_new_conf = model_conf.local_triangle_attention_new
        self._distogram_conf = model_conf.distogram_head
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._model_conf.num_blocks):
            self.trunk[f"atom_attention_encoder_{b}"] = transformer.AtomAttentionEncoder(self._aa_enc_conf)
            self.trunk[f'layernorm_s_{b}'] = LayerNorm(self._model_conf.c_s, create_offset=False)
            self.trunk[f'linear_no_bias_s_{b}'] = LinearNoBias(
                in_features=self._model_conf.c_s,
                out_features=self._model_conf.c_token,
                initializer='zeros'
                )
            self.trunk[f'diffusion_transformer_{b}'] = transformer.DiffusionTransformer(
                c_a=self._diffusion_tfmr.c_token,
                c_s=self._diffusion_tfmr.c_s,
                c_z=self._diffusion_tfmr.c_z,
                blocks_per_ckpt=self._diffusion_tfmr.blocks_per_ckpt,
                n_blocks=self._diffusion_tfmr.n_blocks,
                n_heads=self._diffusion_tfmr.n_heads,
                drop_path_rate=self._diffusion_tfmr.drop_path_rate
                )
            self.trunk[f'layernorm_a_{b}'] = LayerNorm(self._diffusion_tfmr.c_token, create_offset=False)
            self.trunk[f'atom_attention_decoder_{b}'] = transformer.AtomAttentionDecoder(
                cfg=self._aa_dec_conf
            )

            # 0, 1, 2
            if b < self._model_conf.num_blocks-1:
                if self._local_triangle_attention_new_conf.enable:
                    self.trunk[f'edge_transition_{b}'] = LocalTriangleAttentionNew(**self._local_triangle_attention_new_conf)
                    if self._distogram_conf.use_pair_head:
                        self.trunk[f'distogram_head_{b}'] = DistogramHead(
                            self._distogram_conf.c_z,
                            self._distogram_conf.num_bins
                            )

                else:
                    edge_in = self._model_conf.edge_embed_size
                    self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                        node_embed_size=self._ipa_conf.c_s,
                        edge_embed_in=edge_in,
                        edge_embed_out=self._model_conf.edge_embed_size,
                    )
            

    def forward(self, input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        atom_diffuse_mask = input_feats['atom_diffuse_mask']
        loop_mask = input_feats['loop_mask']
        t = input_feats['t']
        
        trans_1 = input_feats['trans_1']
        rotmats_1 = input_feats['rotmats_1']
        pair_init = input_feats['pair_init']
        aatype = input_feats['aatype']
        ag_hotspot = input_feats['ag_hotspot']
        ref_feature_dict = input_feats['ref_feature_dict']
        r_t = input_feats['r_t']
        r_1 = input_feats['r_1']
        r_t = r_t * du.ANG_TO_NM_SCALE
        r_1 = r_1 * du.ANG_TO_NM_SCALE

        # Initialize node and edge embeddings
        s_init = self.node_feature_net(
            t,
            node_mask,
            diffuse_mask,
            loop_mask,
            aatype,
            ag_hotspot,
            ref_feature_dict
        )

        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_1)
        else:
            trans_sc = input_feats['trans_sc']

        if 'rotmats_sc' not in input_feats:
            rotmats_sc = torch.zeros_like(rotmats_1)
        else:
            rotmats_sc = input_feats['rotmats_sc']

        z_init = self.edge_feature_net(
            trans_1,
            trans_sc,
            rotmats_1,
            rotmats_sc,
            edge_mask,
            diffuse_mask,
            loop_mask,
            pair_init
            )

        # Main trunk
        all_atom_outputs = []
        pair_outputs = []
        z_update = z_init.clone()

        for b in range(self._model_conf.num_blocks):
            cb_distogram = None
            
            # atom embed 
            a_token, q_skip, c_skip, p_skip = self.trunk[f"atom_attention_encoder_{b}"](
                input_feature_dict=ref_feature_dict,
                r_l=r_t,
                s=s_init,
                z=z_update,
            ) # [B, N_sample, N_token, c_token]
            
            a_token = a_token + self.trunk[f"linear_no_bias_s_{b}"](
                self.trunk[f"layernorm_s_{b}"](s_init)
            )

            a_token = self.trunk[f"diffusion_transformer_{b}"](
                a=a_token,
                s=s_init,
                z=z_update
            )
            
            a_token = self.trunk[f"layernorm_a_{b}"](a_token)

            r_t = self.trunk[f"atom_attention_decoder_{b}"](
                input_feature_dict=ref_feature_dict,
                a=a_token,
                q_skip=q_skip,
                c_skip=c_skip,
                p_skip=p_skip
            )

            r_t = torch.where(atom_diffuse_mask.unsqueeze(-1).bool(), r_t, r_1)
            r_t_unflatten = du.atom_unflatten(r_t, input_feats['atom14_gt_exists']) # (B, L, 14, 3)
            r_t_unflatten = r_t_unflatten * du.NM_TO_ANG_SCALE # (B, L, 14, 3)

            all_atom_preds = {
                "positions": r_t_unflatten
            }

            if b < self._model_conf.num_blocks-1:
                if self._local_triangle_attention_new_conf.enable:

                    curr_rigids_unscaled = Rigid.from_3_points(
                        r_t_unflatten[:, :, 0],
                        r_t_unflatten[:, :, 1],
                        r_t_unflatten[:, :, 2])

                    z_update = self.trunk[f'edge_transition_{b}'](
                        a_token, z_update, curr_rigids_unscaled, edge_mask
                    )
    
                if self._distogram_conf.use_pair_head:
                    cb_distogram = self.trunk[f'distogram_head_{b}'](z_update)
                    pair_outputs.append(cb_distogram)

            all_atom_outputs.append(all_atom_preds)

        r_update = r_t * du.NM_TO_ANG_SCALE
        all_atom_outputs = dict_multimap(torch.stack, all_atom_outputs)

        curr_rigids_unscaled = Rigid.from_3_points(
            r_t_unflatten[:, :, 0],
            r_t_unflatten[:, :, 1],
            r_t_unflatten[:, :, 2],
        )
        pred_trans = curr_rigids_unscaled.get_trans()
        pred_rotmats = curr_rigids_unscaled.get_rots().get_rot_mats()

        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            'pred_r_1': r_update,
            'backb_frame': curr_rigids_unscaled,
            'all_atom_preds': all_atom_outputs,
            'pair_outputs': pair_outputs
        }
    

class ConfidenceModel(nn.Module):
    def __init__(self, model_conf):
        super(ConfidenceModel, self).__init__()
        self._model_conf = model_conf
        self._aa_enc_conf = model_conf.aa_enc
        self._local_triangle_attention_new_conf = model_conf.local_triangle_attention_new
        self._distogram_conf = model_conf.distogram_head
        self._confidence_head = model_conf.confidence_head

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

        return plddt_logit, pde
