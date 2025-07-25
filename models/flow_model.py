import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models.conditioning_module import ConditioningModule 
from models.structure_module import StructureModule
from models.all_atom import AllAtomModule

from models import ipa_pytorch
from models.heads import DistogramHead, ConfidenceHead
from data import utils as du
from openfold.utils.rigid_utils import local_to_global
from Protenix.protenix.model.modules import pairformer
from Protenix.protenix.model.modules.primitives import LayerNorm, LinearNoBias


class FlowModel(nn.Module):
    def __init__(
            self, 
            model_conf, 
            train_confidence):
        super(FlowModel, self).__init__()
        self.train_confidence = train_confidence
        self._model_conf = model_conf
        self._pairformer_conf = model_conf.pairformer
        self._distogram_conf = model_conf.distogram_head
        self._condition_conf = model_conf.conditioning_module
        self._confidence_conf = model_conf.confidence_head

        self._aa_enc_conf = model_conf.aa_enc
        self._ipa_conf = model_conf.ipa
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
        self.distogram_head_pairformer = DistogramHead(
            c_z=self._distogram_conf.c_z, 
            num_bins=self._distogram_conf.num_bins)
    
        # Condition Module 
        self.condition = ConditioningModule(self._condition_conf)

        # Structure Module 
        self.structure_module = StructureModule(self._model_conf)

        # AllAtom Module 
        self.all_atom = AllAtomModule(self._all_atom_conf)

        # confidence head 
        if self._confidence_conf.use_confidence:
            self.confidence_head = ConfidenceHead(
                n_blocks=self._confidence_conf.n_blocks,
                c_s=self._confidence_conf.c_s,
                c_z=self._confidence_conf.c_z,
                c_s_inputs=self._confidence_conf.c_s_inputs,
                min_bins=self._confidence_conf.min_bins,
                max_bins=self._confidence_conf.max_bins,
                num_bins=self._confidence_conf.num_bins,
                blocks_per_ckpt=self._confidence_conf.blocks_per_ckpt
            )

    def get_pairformer_output(self, 
                              s_init: torch.Tensor, # [1, N_res, c_s]
                              z_init: torch.Tensor, # [1, N_res, N_res, c_z]
                              edge_mask: torch.Tensor, # [1, N_res, N_res]
                              N_cycle: int,
                              B: int):

        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                (not self.train_confidence)
                and cycle_no == (N_cycle - 1)
            ): # training을 하면서 confidence는 훈련하지 않고 마지막 cycle에서만 gradient
                z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))

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
    
    def get_structure_output(
        self,         
        s_single: torch.Tensor,
        s_trunk: torch.Tensor,
        z_pair: torch.Tensor,
        z_trunk: torch.Tensor,
        curr_rigids,
        ref_feature_dict,
        node_mask: torch.Tensor,
        diffuse_mask: torch.Tensor
        ):

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)

        curr_rigids, s_single = self.structure_module(
            s_single,
            z_pair,
            curr_rigids,
            ref_feature_dict,
            node_mask,
            diffuse_mask
        )
        
        curr_rigids_unscaled = self.rigids_nm_to_ang(curr_rigids)

        local_atom_pos_pred = self.all_atom(s_single, s_trunk)
        pred_xyz = local_to_global(curr_rigids_unscaled, local_atom_pos_pred)
        
        all_atom_preds = {
            "positions": pred_xyz
        }

        pred_trans = curr_rigids_unscaled.get_trans()
        pred_rotmats = curr_rigids_unscaled.get_rots().get_rot_mats()

        input_for_confidence = {
            'node_embed': s_trunk,
            'edge_embed': z_trunk,
            'curr_rigids': curr_rigids_unscaled
        }

        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            'backb_frame': curr_rigids_unscaled,
            'all_atom_preds': all_atom_preds,
            'input_for_confidence': input_for_confidence,
        }

    def preprocess_input(self, input_feats, N_cycle):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        pair_init = input_feats['pair_init']
        aatype = input_feats['aatype']
        ref_feature_dict = input_feats['ref_feature_dict']

        B = node_mask.shape[0]

        # Initialize node and edge embeddings
        s_init = self.node_feature_net(
            diffuse_mask[0].unsqueeze(0),
            aatype[0].unsqueeze(0)
        ) # (1, N_res, c_s)

        squeezed_dict = {}
        for k, v in ref_feature_dict.items():
            if k == 'atom_to_token_idx':
                squeezed_dict[k] = v
            else:
                squeezed_dict[k] = v[0].unsqueeze(0)

        z_init = self.edge_feature_net(
            trans_t=trans_t[0].unsqueeze(0),
            trans_sc=None,
            rotmats_t=rotmats_t[0].unsqueeze(0),
            rotmats_sc=None,
            p_mask=edge_mask[0].unsqueeze(0),
            diffuse_mask=diffuse_mask[0].unsqueeze(0),
            pair_init=pair_init[0].unsqueeze(0),
            input_feature_dict=squeezed_dict
        ) # (1, N_res, N_res, c_z)

        # Pairformer 
        s_init, s_trunk, z_trunk, distogram_logit_pairformer = self.get_pairformer_output(
            s_init,
            z_init,
            edge_mask[0].unsqueeze(0),
            N_cycle,
            B) # each output is expanded to batch dimension 
        
        return s_init, s_trunk, z_trunk, distogram_logit_pairformer
    
    def forward(self,
               input_feats,
               N_cycle,
               do_pairformer=True,
               mode='structure' # [structure, pairformer]
               ):

        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        ref_feature_dict = input_feats['ref_feature_dict']

        if mode == 'pairformer':
            return self.preprocess_input(input_feats, N_cycle)
        
        if do_pairformer:
            s_init, s_trunk, z_trunk, distogram_logit_pairformer = self.preprocess_input(input_feats, N_cycle)
        else:
            s_init = input_feats['s_init']
            s_trunk = input_feats['s_trunk']
            z_trunk = input_feats['z_trunk']
            distogram_logit_pairformer = input_feats['distogram_logit_pairformer']
            
        # Conditioning 
        # self-condition
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t, device=trans_t.device)
        else:
            trans_sc = input_feats['trans_sc']

        if 'rotmats_sc' not in input_feats:
            rotmats_sc = torch.zeros_like(rotmats_t, device=rotmats_t.device)
        else:
            rotmats_sc = input_feats['rotmats_sc']

        s_single, z_pair = self.condition(
            t=input_feats['t'],
            pair_init=input_feats['pair_init'],
            s_inputs=s_init,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            trans_sc=trans_sc,
            rotmats_sc=rotmats_sc
            )

        # Structure Module
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        structure_output = self.get_structure_output(
            s_single=s_single,
            s_trunk=s_trunk,
            z_pair=z_pair,
            z_trunk=z_trunk,
            curr_rigids=curr_rigids,
            ref_feature_dict=ref_feature_dict,
            node_mask=node_mask,
            diffuse_mask=diffuse_mask
        )
        structure_output['distogram_logit_pairformer'] = distogram_logit_pairformer

        # Confidence Head
        plddt_preds = None 
        if self._confidence_conf.use_confidence:
            plddt_preds = self.confidence_head(
                s_inputs=s_init,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_mask=edge_mask,
                pred_rigids=structure_output['backb_frame']
            )
        structure_output['plddt'] = plddt_preds

        return structure_output
        