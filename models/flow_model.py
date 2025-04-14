
import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models import ipa_pytorch
from data import utils as du
from data import all_atom
from openfold.utils.tensor_utils import dict_multimap
from Proteus.model.ipa_pytorch import LocalTriangleAttentionNew

class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden, use_original_sm):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()

        self.c_hidden = c_hidden
        self.use_original_sm = use_original_sm

        if not self.use_original_sm:
            self.linear_1 = ipa_pytorch.Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = ipa_pytorch.Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_3 = ipa_pytorch.Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        s_initial = a

        if not self.use_original_sm:
            a = self.relu(a)
            a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)
        a = self.relu(a)
        a = self.linear_3(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon, use_original_sm):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
            use_original_sm:
                If True implement line 11 of algorithm 20 correctly else use the ABB3 implementation.
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon
        self.use_original_sm = use_original_sm

        if self.use_original_sm:
            self.linear_in = ipa_pytorch.Linear(self.c_in, self.c_hidden)
            self.linear_initial = ipa_pytorch.Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(
                c_hidden=self.c_hidden, use_original_sm=self.use_original_sm
            )
            self.layers.append(layer)

        self.linear_out = ipa_pytorch.Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor, s_initial: torch.Tensor
    ):
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        if self.use_original_sm:
            s_initial = self.relu(s_initial)
            s_initial = self.linear_initial(s_initial)
            s = self.relu(s)
            s = self.linear_in(s)
            s = s + s_initial
        else:
            s = torch.cat((s, s_initial), dim=-1)

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s
    
class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self._local_triangle_attention_new_conf = model_conf.local_triangle_attention_new
        self._angle_conf = model_conf.angle
        self._prmsd_conf = model_conf.prmsd
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                if self._local_triangle_attention_new_conf.enable:
                    self.trunk[f'edge_transition_{b}'] = LocalTriangleAttentionNew(**self._local_triangle_attention_new_conf)
                
                else:
                    edge_in = self._model_conf.edge_embed_size
                    self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                        node_embed_size=self._ipa_conf.c_s,
                        edge_embed_in=edge_in,
                        edge_embed_out=self._model_conf.edge_embed_size,
                    )

        self.angle_resnet = AngleResnet(
                self._ipa_conf.c_s,
                self._angle_conf.c_resnet,
                self._angle_conf.no_resnet_blocks,
                self._angle_conf.no_angles,
                self._angle_conf.epsilon,
                self._angle_conf.use_original_sm
            )
        
        self.prmsd = nn.ModuleDict()
        if self._prmsd_conf.use_prmsd:
            self.prmsd_node_transform = ipa_pytorch.Linear(self._prmsd_conf.c_s, self._prmsd_conf.c_s)
            
            self.prmsd_edge_transform = ipa_pytorch.Linear(
                    self._prmsd_conf.c_z,
                    self._prmsd_conf.c_z
                )

            for b in range(self._prmsd_conf.num_blocks):
                self.prmsd[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._prmsd_conf)
                self.prmsd[f'ipa_ln_{b}'] = nn.LayerNorm(self._prmsd_conf.c_s)
                tfmr_in = self._prmsd_conf.c_s
                tfmr_layer = torch.nn.TransformerEncoderLayer(
                    d_model=tfmr_in,
                    nhead=self._prmsd_conf.seq_tfmr_num_heads,
                    dim_feedforward=tfmr_in,
                    batch_first=True,
                    dropout=0.0,
                    norm_first=False
                )
                self.prmsd[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                    tfmr_layer, self._prmsd_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
                self.prmsd[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                    tfmr_in, self._prmsd_conf.c_s, init="final")
                self.prmsd[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                    c=self._prmsd_conf.c_s)

            self.prmsd_linear = ipa_pytorch.Linear(
                self._prmsd_conf.c_s,
                50, # prmsd bins 개수 
                )
            
    def forward(self, input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        r3_t = input_feats['r3_t']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        pair_init = input_feats['pair_init']
        aatype = input_feats['aatype']
        
        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            r3_t,
            node_mask,
            diffuse_mask,
            aatype
        )

        if 'trans_sc' not in input_feats:
            trans_sc = du.manage_missing_batch(trans_t, mask=~diffuse_mask.bool())
        else:
            trans_sc = input_feats['trans_sc']

        if 'rotmats_sc' not in input_feats:
            rotmats_sc = du.manage_missing_batch(rotmats_t, mask=~diffuse_mask.bool())
        else:
            rotmats_sc = input_feats['rotmats_sc']

        init_edge_embed = self.edge_feature_net(
            trans_t,
            trans_sc,
            rotmats_t,
            rotmats_sc,
            edge_mask,
            diffuse_mask,
            pair_init
        )

        # Initialize rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        all_atom_outputs = []


        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]

        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None])
            if b < self._ipa_conf.num_blocks-1:
                if self._local_triangle_attention_new_conf.enable:
                    edge_embed = self.trunk[f'edge_transition_{b}'](
                        node_embed, edge_embed, curr_rigids, edge_mask
                    )
                    edge_embed *= edge_mask[..., None]
               
                else:
                    edge_embed = self.trunk[f'edge_transition_{b}'](
                        node_embed, edge_embed)
                    edge_embed *= edge_mask[..., None]

            unnormalized_angles, angles = self.angle_resnet(node_embed, init_node_embed)

            backb_to_global = self.rigids_nm_to_ang(curr_rigids)
            all_frames_to_global = all_atom.torsion_angles_to_frames(backb_to_global, angles, aatype)
            pred_xyz = all_atom.frames_to_atom14_pos(all_frames_to_global, aatype)
            all_atom_preds = {
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "rigids": backb_to_global.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4()
            }

            all_atom_outputs.append(all_atom_preds)

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

        all_atom_outputs = dict_multimap(torch.stack, all_atom_outputs)

        if self._prmsd_conf.use_prmsd:

            prmsd_node = self.prmsd_node_transform(init_node_embed) 
            prmsd_edge = self.prmsd_edge_transform(init_edge_embed)

            rots = curr_rigids.get_rots().get_rot_mats().detach()
            trans = curr_rigids.get_trans().detach()
            prmsd_rigids = du.create_rigid(rots, trans)

            for b in range(self._prmsd_conf.num_blocks):
                prmsd_ipa_embed = self.prmsd[f'ipa_{b}'](
                    prmsd_node,
                    prmsd_edge,
                    prmsd_rigids,
                    node_mask
                )
                prmsd_ipa_embed *= node_mask[..., None]
                prmsd_node = self.prmsd[f'ipa_ln_{b}'](prmsd_node + prmsd_ipa_embed)
                prmsd_seq_tfmr_out = self.prmsd[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))

                prmsd_node = prmsd_node + self.prmsd[f'post_tfmr_{b}'](prmsd_seq_tfmr_out)
                prmsd_node = self.prmsd[f'node_transition_{b}'](prmsd_node)
                prmsd_node = prmsd_node * node_mask[..., None]

            # print(f'prmsd_node after attention: {prmsd_node}')
            prmsd_node = self.prmsd_linear(prmsd_node) # (b, L, 50)
            # print(f'prmsd_before_relu: {prmsd_node}')
            all_atom_outputs["prmsd"] = prmsd_node
        else:
            all_atom_outputs['prmsd'] = torch.zeros(node_embed.shape[0], node_embed.shape[1])

        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            'all_atom_preds': all_atom_outputs
        }
