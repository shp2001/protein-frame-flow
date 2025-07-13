import torch
from torch import nn


class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        embed_size = 1 + 21
        self.linear = nn.Linear(embed_size, self.c_s)


    def forward(self, diffuse_mask, aatype):
        single = torch.nn.functional.one_hot(aatype, 21).float()
        # [b, n_res, c_timestep_emb]
        input_feats = [
            single,
            diffuse_mask[..., None]
        ]
        return self.linear(torch.cat(input_feats, dim=-1))
