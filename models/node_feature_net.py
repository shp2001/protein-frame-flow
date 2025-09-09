import torch
from torch import nn
from models.utils import get_time_embedding
from Protenix.protenix.model.modules.transformer import AtomAttentionEncoder

class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self._aa_enc_cfg = module_cfg.aa_enc
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        embed_size = 21 + 1 + 1 + 1 + self._cfg.c_timestep_emb + self._aa_enc_cfg.c_token
        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb
        
        self.aa_enc = AtomAttentionEncoder(self._aa_enc_cfg)
        self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, r3_t, res_mask, diffuse_mask, loop_mask, aatype, ag_hotspot, ref_feature_dict):
        single = torch.nn.functional.one_hot(aatype, 21).float()
        a_token, _, _, _ = self.aa_enc(
            input_feature_dict=ref_feature_dict,
        )  # [..., N_token, c_token]

        # [b, n_res, c_timestep_emb]
        input_feats = [
            single,
            diffuse_mask[..., None],
            loop_mask[..., None],
            ag_hotspot[..., None],
            self.embed_t(r3_t, res_mask),
            a_token
        ]
        
        return self.linear(torch.cat(input_feats, dim=-1))
