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

        self.linear_relpos = nn.Linear(self.relpos_dim, self.feat_dim)

        # total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2
        total_edge_feats = self.feat_dim
        if self._cfg.ref_pos_dim: 
            total_edge_feats += self._cfg.ref_pos_dim
        if self._cfg.embed_chain:
            total_edge_feats += 1
        if self._cfg.embed_loop_mask:
            total_edge_feats += 1
        if self._cfg.embed_diffuse_mask:
            total_edge_feats += 1
        if self._cfg.embed_distogram:
            total_edge_feats += self._cfg.num_bins * 2
        if self._cfg.embed_unit_vector:
            total_edge_feats += 3 * 2
        
        self.ref_pos_embedder = RefPosEmbedder(c_atompair=self.ref_pos_dim)

        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_z),
            nn.ReLU(),
            nn.Linear(self.c_z, self.c_z),
            nn.ReLU(),
            nn.Linear(self.c_z, self.c_z),
            nn.LayerNorm(self.c_z),
        )

    def embed_relpos(self, pair_init):
        return self.linear_relpos(pair_init)

    def forward(self, 
                trans_template, trans_sc,
                rotmats_template, rotmats_sc, 
                p_mask, diffuse_mask, loop_mask, 
                pair_init,
                input_feature_dict):
        """
        trans_sc, rotmats_sc : if there was self-condition value, it is sc-value.
                                If not, it is cdr_masked (cdr masked to the closeast residues) value 
        """
        # [b, n_res, c_z]
        relpos_feats = self.embed_relpos(pair_init)
        all_edge_feats = [relpos_feats]

        if self._cfg.embed_ref_pos:
            ref_pos = self.ref_pos_embedder(input_feature_dict)
            all_edge_feats.append(ref_pos)

        if self._cfg.embed_loop_mask:
            loop_feat = (1-loop_mask[:, :, None]) * (1-loop_mask[:, None, :]) # cdr: 0 non_cdr: 1 -> 하나라도 cdr이면 0 아니면 1
            all_edge_feats.append(loop_feat[..., None])

        if self._cfg.embed_diffuse_mask:
            diffuse_mask_i = diffuse_mask[:, :, None]  # (B, L, 1)
            diffuse_mask_j = diffuse_mask[:, None, :]  # (B, 1, L)
            diffuse_mask = (diffuse_mask_i == diffuse_mask_j).float() 
            all_edge_feats.append(diffuse_mask[..., None])

        if self._cfg.embed_distogram:
            distogram_t = calc_distogram(
                trans_template, min_bin=self._cfg.min_bin, max_bin=self._cfg.max_bin, num_bins=self._cfg.num_bins)
            distogram_t = distogram_t * loop_feat[..., None]
            all_edge_feats.append(distogram_t)

            distogram_sc = calc_distogram(
                trans_sc, min_bin=self._cfg.min_bin, max_bin=self._cfg.max_bin, num_bins=self._cfg.num_bins)
            all_edge_feats.append(distogram_sc)

        if self._cfg.embed_unit_vector:
            rigid_t = create_rigid(rotmats_template, trans_template)
            unit_vec_t = calc_unit_vector(rigid_t)
            unit_vec_t = unit_vec_t * loop_feat[..., None]
            all_edge_feats.append(unit_vec_t)

            rigid_sc = create_rigid(rotmats_sc, trans_sc)
            unit_vec_sc = calc_unit_vector(rigid_sc)
            all_edge_feats.append(unit_vec_sc)

        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= p_mask.unsqueeze(-1)
        return edge_feats