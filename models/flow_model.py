
import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models.conditioning import ConditioningModule 
from models.heads import DistogramHead
from models.utils import calc_distogram, calc_unit_vector
from models import ipa_pytorch

from data import utils as du
from openfold.utils.rigid_utils import Rigid

from Proteus.model.ipa_pytorch import LocalTriangleAttentionNew

from Protenix.protenix.model.modules import transformer, pairformer
from Protenix.protenix.openfold_local.model.primitives import LayerNorm
from Protenix.protenix.model.modules.primitives import LinearNoBias

class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self.train_confidence = model_conf.train_confidence

        # Config 
        self._model_conf = model_conf
        self._pairformer_conf = model_conf.pairformer
        self._distogram_conf = model_conf.distogram_head

        self._condition_conf = model_conf.conditioning

        self._aa_enc_conf = model_conf.aa_enc
        self._diffusion_tfmr = model_conf.diffusion_transformer
        self._aa_dec_conf = model_conf.aa_decoder

        # Input Embedder 
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        # Pairformer
        self.pairformer = pairformer.PairformerStack(
            n_blocks=self._pairformer_conf.n_blocks,
            n_heads=self._pairformer_conf.n_heads,
            c_z=self._pairformer_conf.c_z,
            c_s=self._pairformer_conf.c_s,
            dropout=self._pairformer_conf.dropout,
            blocks_per_ckpt=self._pairformer_conf.blocks_per_ckpt
            )
        self.distogram_head_pairformer = DistogramHead(
            c_z=self._distogram_conf.c_z, 
            num_bins=self._distogram_conf.num_bins
            )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self._model_conf.c_z, 
            out_features=self._model_conf.c_z
            )
        self.layernorm_z_cycle = LayerNorm(self._model_conf.c_z)
        self.linear_no_bias_s_cycle = LinearNoBias(
            in_features=self._model_conf.c_s,
            out_features=self._model_conf.c_s,
            initializer='zeros'
            )
        self.layernorm_s_cycle = LayerNorm(self._model_conf.c_s, create_offset=False)
        # Condition Module 
        self.condition = ConditioningModule(self._condition_conf)

        # Structure Module
        self.atom_attention_encoder = transformer.AtomAttentionEncoder(self._aa_enc_conf)
        self.layernorm_s = LayerNorm(self._model_conf.c_s, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self._model_conf.c_s,
            out_features=self._model_conf.c_token,
            initializer='zeros'
            )
        self.diffusion_transformer = transformer.DiffusionTransformer(
            c_a=self._diffusion_tfmr.c_token,
            c_s=self._diffusion_tfmr.c_s,
            c_z=self._diffusion_tfmr.c_z,
            blocks_per_ckpt=self._diffusion_tfmr.blocks_per_ckpt,
            n_blocks=self._diffusion_tfmr.n_blocks,
            n_heads=self._diffusion_tfmr.n_heads,
            drop_path_rate=self._diffusion_tfmr.drop_path_rate
            )
        self.layernorm_a = LayerNorm(self._diffusion_tfmr.c_token, create_offset=False)
        self.atom_attention_decoder = transformer.AtomAttentionDecoder(
            cfg=self._aa_dec_conf
        )


    def embed_input(self, input_feats):
        diffuse_mask = input_feats['diffuse_mask'][0].unsqueeze(0)
        loop_mask = input_feats['loop_mask'][0].unsqueeze(0)
        aatype = input_feats['aatype'][0].unsqueeze(0)
        ag_hotspot = input_feats['ag_hotspot'][0].unsqueeze(0)
        trans_1 = input_feats['trans_1'][0].unsqueeze(0)
        rotmats_1 = input_feats['rotmats_1'][0].unsqueeze(0)
        asym_id = input_feats['asym_id'][0].unsqueeze(0)
        residue_index = input_feats['residue_index'][0].unsqueeze(0)
        entity_id = input_feats['entity_id'][0].unsqueeze(0)
        sym_id = input_feats['sym_id'][0].unsqueeze(0)

        ref_feature_dict = input_feats['ref_feature_dict']
        squeezed_ref_feature_dict = {}
        for k, v in ref_feature_dict.items():
            if k == 'atom_to_token_idx':
                squeezed_ref_feature_dict[k] = v
            else:
                squeezed_ref_feature_dict[k] = v[0].unsqueeze(0)

        # Input Embedding 
        s_init = self.node_feature_net(
            diffuse_mask,
            loop_mask,
            aatype,
            ag_hotspot,
            squeezed_ref_feature_dict
        )

        z_init = self.edge_feature_net(
            s_init,
            trans_1,
            rotmats_1,
            diffuse_mask,
            loop_mask,
            asym_id,
            residue_index,
            entity_id,
            sym_id
            )
    
        return s_init, z_init
    
    def do_pairformer(
        self, 
        s_init: torch.Tensor, # [1, N_res, c_s]
        z_init: torch.Tensor, # [1, N_res, N_res, c_z]
        edge_mask: torch.Tensor, # [1, N_res, N_res]
        N_cycle: int,
        B: int
        ):

        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                (not self.train_confidence)
                and cycle_no == (N_cycle - 1)
            ): # training을 하면서 confidence는 훈련하지 않고 마지막 cycle에서만 gradient
                z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                s = s_init + self.linear_no_bias_s_cycle(self.layernorm_s_cycle(s))

                s, z = self.pairformer(
                    s=s,
                    z=z,
                    pair_mask=edge_mask,
                    use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=False,
                    use_lma=False
                )

        distogram_logit = self.distogram_head_pairformer(z) # (1, N_res, N_res, num_bins)
        s_init = s.repeat(B, 1, 1) # (B, N_res, c_s)
        s = s.repeat(B, 1, 1) # (B, N_res, c_s)
        z = z.repeat(B, 1, 1, 1) # (B, N_res, N_res, c_z)
        distogram_logit = distogram_logit.repeat(B, 1, 1, 1) # (B, N_res, N_res, num_bins)

        return s_init, s, z, distogram_logit

    
    def get_structure(self, input_feats, s_init, s_trunk, z_trunk):
        # conditioning 
        s_single, z_pair = self.condition(
            t=input_feats['t'],
            asym_id=input_feats['asym_id'],
            residue_index=input_feats['residue_index'],
            entity_id=input_feats['entity_id'],
            sym_id=input_feats['sym_id'],
            s_inputs=s_init,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            )
        
        # atom embed 
        a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
            input_feature_dict=input_feats["ref_feature_dict"],
            r_l=input_feats["r_t"],
            s=s_trunk,
            z=z_pair,
        ) # [B, N_sample, N_token, c_token]
        a_token = a_token + self.linear_no_bias_s(
            self.layernorm_s(s_single)
        )
        a_token = self.diffusion_transformer(
            a=a_token,
            s=s_single,
            z=z_pair
        )
        a_token = self.layernorm_a(a_token)
        r_update = self.atom_attention_decoder(
            input_feature_dict=input_feats["ref_feature_dict"],
            a=a_token,
            q_skip=q_skip,
            c_skip=c_skip,
            p_skip=p_skip
        )
        r_update = torch.where(input_feats["atom_diffuse_mask"].unsqueeze(-1).bool(), r_update, input_feats["r_1"])
        r_update_unflatten = du.atom_unflatten(r_update, input_feats['atom14_gt_exists']) # (B, L, 14, 3)

        curr_rigids_unscaled = Rigid.from_3_points(
            r_update_unflatten[:, :, 0],
            r_update_unflatten[:, :, 1],
            r_update_unflatten[:, :, 2],
        )
        pred_trans = curr_rigids_unscaled.get_trans()
        pred_rotmats = curr_rigids_unscaled.get_rots().get_rot_mats()

        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            'pred_r_1': r_update,
            'pred_r_1_unflatten': r_update_unflatten,
        }
    
    def forward(self, input_feats, N_cycle):
        s_init, z_init = self.embed_input(input_feats)

        edge_mask = input_feats['edge_mask'][0].unsqueeze(0)
        s_init, s_trunk, z_trunk, distogram_logit_pairformer = self.do_pairformer(
            s_init, 
            z_init, 
            edge_mask,
            N_cycle,
            input_feats['res_mask'].shape[0]
            )
        structure_output = self.get_structure(input_feats, s_init, s_trunk, z_trunk)
        structure_output['pair_outputs'] = distogram_logit_pairformer
        return structure_output
    

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
        edge_mask = input_feats['edge_mask']

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
