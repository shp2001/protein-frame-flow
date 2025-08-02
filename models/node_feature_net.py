import torch
from torch import nn
from data.featurizer import get_ref_pos, compute_residue_side_chain_distance

    
class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        embed_size = 1 + 21

        if self._cfg.embed_sc_dist:
            embed_size += self._cfg.sc_dist_dim
        
        self.linear_sc_dist = nn.Linear(14*14, self._cfg.sc_dist_dim)
        self.ln_sc_dist = nn.LayerNorm(self._cfg.sc_dist_dim)
        self.linear_s = nn.Linear(embed_size, self.c_s)

    def forward(self, diffuse_mask, aatype):
        # aatype embedding 
        single = torch.nn.functional.one_hot(aatype, 21).float()

        # side chain conformation embedding 
        B, N = aatype.shape
        aatype_one = aatype[0]
        ref_pos = get_ref_pos(aatype_one)
        sc_dist = compute_residue_side_chain_distance(ref_pos)
        sc_dists = sc_dist.unsqueeze(0).expand(B, -1, -1, -1).clone()
        sc_dists = sc_dists.reshape(B, N, -1).to(aatype.device)
        single_dist = self.linear_sc_dist(sc_dists)
        single_dist = self.ln_sc_dist(single_dist)

        input_feats = [
            single,
            single_dist,
            diffuse_mask[..., None]
        ]
        return self.linear_s(torch.cat(input_feats, dim=-1))
