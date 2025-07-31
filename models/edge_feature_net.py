import torch
from torch import nn

from models.utils import calc_distogram, calc_unit_vector
from data.utils import create_rigid
from Protenix.protenix.model.modules.transformer import RefPosEmbedder

class EdgeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeFeatureNet, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_z = self._cfg.c_p
        self.ref_pos_dim = self._cfg.ref_pos_dim
        self.feat_dim = self._cfg.feat_dim
        self.relpos_dim = self._cfg.relpos_dim

        self.linear_relpos = nn.Linear(self.relpos_dim, self.feat_dim, bias=False)

        # total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2
        total_edge_feats = 0
        if self._cfg.embed_ref_pos: 
            self.ref_pos_embedder = RefPosEmbedder(c_atompair=self.ref_pos_dim)
            total_edge_feats += self._cfg.ref_pos_dim
        if self._cfg.embed_chain:
            total_edge_feats += 1
        if self._cfg.embed_diffuse_mask:
            total_edge_feats += 1
        if self._cfg.embed_distogram:
            total_edge_feats += self._cfg.num_bins
        if self._cfg.embed_unit_vector:
            total_edge_feats += 3 
        if self._cfg.embed_self_condition:
            total_edge_feats += self._cfg.num_bins + 3

        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_z),
            nn.ReLU(),
            nn.Linear(self.c_z, self.c_z),
            nn.ReLU(),
            nn.Linear(self.c_z, self.c_z),
            nn.LayerNorm(self.c_z),
        )

    def forward(self, 
                trans_t, 
                trans_sc, 
                rotmats_t, 
                rotmats_sc, 
                p_mask, 
                diffuse_mask,
                pair_init,
                input_feature_dict):
        """
        trans_sc, rotmats_sc : if there was self-condition value, it is sc-value.
                                If not, it is cdr_masked (cdr masked to the closeast residues) value 
        """

        relpos_feats = self.linear_relpos(pair_init)
        all_edge_feats = []

        if self._cfg.embed_ref_pos:
            ref_pos = self.ref_pos_embedder(input_feature_dict)
            all_edge_feats.append(ref_pos)

        if self._cfg.embed_diffuse_mask:
            diff_feat = (1-diffuse_mask[:, :, None]) * (1-diffuse_mask[:, None, :]) # cdr: 0 non_cdr: 1 -> 하나라도 cdr이면 0 아니면 1
            all_edge_feats.append(diff_feat[..., None])

        if self._cfg.embed_distogram:
            distogram_t = calc_distogram(
                trans_t, min_bin=2.0, max_bin=32.0, num_bins=self._cfg.num_bins)
            distogram_t = distogram_t * diff_feat[..., None]
            all_edge_feats.append(distogram_t)
            if self._cfg.embed_self_condition:
                distogram_sc = calc_distogram(
                    trans_sc, min_bin=2.0, max_bin=32.0, num_bins=self._cfg.num_bins)
                all_edge_feats.append(distogram_sc)

        if self._cfg.embed_unit_vector:
            rigid_t = create_rigid(rotmats_t, trans_t)
            unit_vec_t = calc_unit_vector(rigid_t)
            unit_vec_t = unit_vec_t * diff_feat[..., None]
            all_edge_feats.append(unit_vec_t)

            if self._cfg.embed_self_condition:
                rigid_sc = create_rigid(rotmats_sc, trans_sc)
                unit_vec_sc = calc_unit_vector(rigid_sc)
                all_edge_feats.append(unit_vec_sc)

        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= p_mask.unsqueeze(-1)
        edge_feats = edge_feats + relpos_feats
        return edge_feats