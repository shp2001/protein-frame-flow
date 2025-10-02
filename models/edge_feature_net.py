import torch
from torch import nn

from models.utils import calc_distogram, calc_unit_vector
from data.utils import create_rigid

from Protenix.protenix.model.modules.primitives import LinearNoBias

class RelativePositionEncoding(nn.Module):
    """
    Implements Algorithm 3 in AF3
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128) -> None:
        """
        Args:
            r_max (int, optional): Relative position indices clip value. Defaults to 32.
            s_max (int, optional): Relative chain indices clip value. Defaults to 2.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        """
        super(RelativePositionEncoding, self).__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear_no_bias = LinearNoBias(
            in_features=(2 * self.r_max + 2 * self.s_max + 5), out_features=self.c_z
        )

    def one_hot(self, x, v_bins):
        reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
        diffs = x[..., None] - reshaped_bins
        am = torch.argmin(torch.abs(diffs), dim=-1)

        return torch.nn.functional.one_hot(am, num_classes=len(v_bins)).float()
    
    def forward(
        self,
        asym_id: torch.Tensor,
        residue_index: torch.Tensor,
        entity_id: torch.Tensor,
        sym_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            asym_id / residue_index / entity_id / sym_id / token_index
                [..., N_tokens]
        Returns:
            torch.Tensor: relative position encoding
                [..., N_token, N_token, c_z]
        """
        device = residue_index.device

        pos = residue_index
        asym_id_same = (asym_id[..., None] == asym_id[..., None, :])
        offset = pos[..., None] - pos[..., None, :]

        clipped_offset = torch.clamp(
            offset + self.r_max, 0, 2 * self.r_max
        )

        rel_feats = []

        final_offset = torch.where(
            asym_id_same, 
            clipped_offset,
            (2 * self.r_max + 1) * 
            torch.ones_like(clipped_offset)
        )

        boundaries = torch.arange(
            start=0, end=2 * self.r_max + 2
        ).to(device)

        rel_pos = self.one_hot(
            final_offset,
            boundaries,
        )

        rel_feats.append(rel_pos)
        entity_id_same = (entity_id[..., None] == entity_id[..., None, :])
        rel_feats.append(entity_id_same[..., None].to(dtype=rel_pos.dtype))
        rel_sym_id = sym_id[..., None] - sym_id[..., None, :]

        clipped_rel_chain = torch.clamp(
            rel_sym_id + self.s_max,
            0,
            2 * self.s_max,
        )

        final_rel_chain = torch.where(
            entity_id_same,
            clipped_rel_chain,
            (2 * self.s_max + 1) *
            torch.ones_like(clipped_rel_chain)
        )

        boundaries = torch.arange(
            start=0, end=2 * self.s_max + 2
        ).to(device)
        rel_chain = self.one_hot(
            final_rel_chain,
            boundaries,
        )

        rel_feats.append(rel_chain)
        rel_feat = torch.cat(rel_feats, dim=-1)

        p = self.linear_no_bias(rel_feat)  
        
        return p

class EdgeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_z = self._cfg.c_z

        # total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2
        total_edge_feats = self.c_z

        if self._cfg.embed_loop_mask:
            total_edge_feats += 1
        if self._cfg.embed_diag_mask:
            total_edge_feats += 1
        if self._cfg.embed_distogram_diag:
            total_edge_feats += self._cfg.distogram_diag.num_bins
        if self._cfg.embed_distogram_off_diag:
            total_edge_feats += self._cfg.distogram_off_diag.num_bins
        if self._cfg.embed_unit_vector:
            total_edge_feats += 3 

        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=module_cfg.c_s, out_features=module_cfg.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=module_cfg.c_s, out_features=module_cfg.c_z
        )
        self.relpos_embedder = RelativePositionEncoding(
            r_max=module_cfg.relpos.r_max,
            s_max=module_cfg.relpos.s_max,
            c_z=module_cfg.relpos.c_z
        )
        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_z),
            nn.ReLU(),
            nn.Linear(self.c_z, self.c_z),
            nn.ReLU(),
            nn.Linear(self.c_z, self.c_z),
            nn.LayerNorm(self.c_z),
        )

    def forward(self, 
                s_init,
                trans_template, 
                rotmats_template,   
                diffuse_mask,
                loop_mask, 
                asym_id: torch.Tensor,
                residue_index: torch.Tensor,
                entity_id: torch.Tensor,
                sym_id: torch.Tensor,
                ):
        """
        trans_sc, rotmats_sc : if there was self-condition value, it is sc-value.
                                If not, it is cdr_masked (cdr masked to the closeast residues) value 
        """
        # [b, n_res, c_z]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  #  [..., N_token, N_token, c_z]

        all_edge_feats = [z_init]
        if self._cfg.embed_loop_mask:
            loop_mask_2d = (1-loop_mask[:, :, None]) * (1-loop_mask[:, None, :]) # cdr: 0 non_cdr: 1 -> 하나라도 cdr이면 0 아니면 1
            all_edge_feats.append(loop_mask_2d[..., None])

        if self._cfg.embed_diag_mask:
            diffuse_mask_i = diffuse_mask[:, :, None]  # (B, L, 1)
            diffuse_mask_j = diffuse_mask[:, None, :]  # (B, 1, L)
            diag_mask = (diffuse_mask_i == diffuse_mask_j).float() # 1: diag / 0: off-diag
            all_edge_feats.append(diag_mask[..., None])

        if self._cfg.embed_distogram_diag:
            distogram_diag = calc_distogram(
                trans_template, 
                min_bin=self._cfg.distogram_diag.min_bin, 
                max_bin=self._cfg.distogram_diag.max_bin, 
                num_bins=self._cfg.distogram_diag.num_bins
                )
            distogram_diag = distogram_diag * loop_mask_2d[..., None]
            distogram_diag = distogram_diag * diag_mask[..., None]
            all_edge_feats.append(distogram_diag)

        if self._cfg.embed_distogram_off_diag:
            distogram_off_diag = calc_distogram(
                trans_template, 
                min_bin=self._cfg.distogram_off_diag.min_bin,
                max_bin=self._cfg.distogram_off_diag.max_bin, 
                num_bins=self._cfg.distogram_off_diag.num_bins
                )
            distogram_off_diag = distogram_off_diag * loop_mask_2d[..., None]
            distogram_off_diag = distogram_off_diag * (1-diag_mask)[..., None]
            all_edge_feats.append(distogram_off_diag)

        if self._cfg.embed_unit_vector:
            rigid_t = create_rigid(rotmats_template, trans_template)
            unit_vec_t = calc_unit_vector(rigid_t)
            unit_vec_t = unit_vec_t * loop_mask_2d[..., None]
            unit_vec_t = unit_vec_t * diag_mask[..., None]
            all_edge_feats.append(unit_vec_t)

        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        relpos_feats = self.relpos_embedder(
            asym_id,
            residue_index,
            entity_id,
            sym_id
        )
        edge_feats = edge_feats + relpos_feats  
        return edge_feats