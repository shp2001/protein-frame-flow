import torch.nn as nn
from Protenix.protenix.model.modules.primitives import LayerNorm, LinearNoBias


class AllAtomModule(nn.Module):
    """All-atom update resnet module."""

    def __init__(self, config):
        super().__init__()
        self.relu = nn.ReLU()
        self.linear_in = LinearNoBias(config.d_single, config.d_hidden)
        self.linear_initial = LinearNoBias(config.d_single, config.d_hidden)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm(config.d_hidden),
                    nn.GELU(),
                    LinearNoBias(config.d_hidden, config.d_hidden, initializer="relu"),
                    nn.GELU(),
                    LinearNoBias(config.d_hidden, config.d_hidden, initializer="zeros"),
                )
                for _ in range(config.n_blocks)
            ]
        )
        self.mlp_out = nn.Sequential(
            LayerNorm(config.d_hidden),
            nn.GELU(),
            LinearNoBias(config.d_hidden, config.atom_num * 3),
        )

    def forward(
        self,
        single,
        init_single,
    ):
        single = self.linear_in(self.relu(single))
        single = single + self.linear_initial(self.relu(init_single))

        for block in self.blocks:
            single = single + block(single)
        single = self.mlp_out(self.relu(single))
        return single.view(single.shape[:-1] + (-1, 3))