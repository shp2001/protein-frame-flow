
import pandas as pd
import torch
import yaml
from data import utils as du


from openfold.data import data_transforms
from openfold.utils import rigid_utils
import json 

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

from data.motif_index import crop_antigen
import argparse

from typing import Any
import torch


import pandas as pd
from data import utils as du
from models.loss import *
from omegaconf import OmegaConf

import sys 
sys.path.append('/home/psh/project_v2/local_ImmuneBuilder/igfold_utils')
from einops import rearrange

parser = argparse.ArgumentParser(
    description='Merging two csv files')

parser.add_argument(
    '--csv',
    type=str,
    default='/home/psh/benchmark_after210930/meta/metadata.csv')

parser.add_argument(
    '--cfg',
    type=str,
    default='/home/psh/protein-frame-flow/configs/base.yaml')

def kabsch(
    mobile,
    stationary,
    return_translation_rotation=False,
):
    X = rearrange(
        mobile,
        "... l d -> ... d l",
    )
    Y = rearrange(
        stationary,
        "... l d -> ... d l",
    )

    #  center X and Y to the origin
    XT, YT = X.mean(dim=-1, keepdim=True), Y.mean(dim=-1, keepdim=True)
    X_ = X - XT
    Y_ = Y - YT

    # calculate convariance matrix
    C = torch.einsum("... x l, ... y l -> ... x y", X_, Y_)

    # Optimal rotation matrix via SVD
    if int(torch.__version__.split(".")[1]) < 8:
        # warning! int torch 1.<8 : W must be transposed
        V, S, W = torch.svd(C)
        W = rearrange(W, "... a b -> ... b a")
    else:
        V, S, W = torch.linalg.svd(C)

    # determinant sign for direction correction
    v_det = torch.det(V.to("cpu")).to(X.device)
    w_det = torch.det(W.to("cpu")).to(X.device)
    d = (v_det * w_det) < 0.0
    if d.any():
        S[d] = S[d] * (-1)
        V[d, :] = V[d, :] * (-1)

    # Create Rotation matrix U
    U = torch.matmul(V, W)  #.to(device)

    U = rearrange(
        U,
        "... d x -> ... x d",
    )
    XT = rearrange(
        XT,
        "... d x -> ... x d",
    )
    YT = rearrange(
        YT,
        "... d x -> ... x d",
    )

    if return_translation_rotation:
        return XT, U, YT

    transform = lambda coords: torch.einsum(
        "... l d, ... x d -> ... l x",
        coords - XT,
        U,
    ) + YT
    mobile = transform(mobile)

    return mobile, transform


def do_kabsch(
    mobile,
    stationary,
    align_mask=None,
):
    mobile_, stationary_ = mobile.clone(), stationary.clone()
    if align_mask != None:
        # print("align_mask: ", align_mask.shape)
        # print("mobile_: ", mobile_.shape)
        # print("stationary_: ", stationary_.shape)
        # print(f"mobile_[align_mask]: {mobile_[align_mask].shape}")
        mobile_[~align_mask] = mobile_[align_mask].mean(dim=-2)
        stationary_[~align_mask] = stationary_[align_mask].mean(dim=-2)
        _, kabsch_xform = kabsch(
            mobile_,
            stationary_,
        )
    else:
        _, kabsch_xform = kabsch(
            mobile_,
            stationary_,
        )

    return kabsch_xform(mobile)


def sample_scaffold_mask(batch):
    aatype = batch['aatype']
    mode = batch['mode']
    num_res = aatype.shape[0]
    scaffold_idx = batch['scaffold_idx']
    scaffold_mask = torch.zeros(num_res)

    if mode == 'general': # general loop PPI
        loop_indices = []
        for scf, idx in scaffold_idx.items():
            loop_indices.append(idx)
        loop_indices = sorted(loop_indices)

        scaffold_mask[loop_indices[0]:loop_indices[1]+1] = 1.0

    elif mode == 'ab': # antibody-antigen
        cdr_indices = []
        for scf, idx in scaffold_idx.items():
            cdr_indices.append(idx)
        cdr_indices = sorted(cdr_indices)
        for i in range(6):
            scaffold_mask[cdr_indices[2*i]:cdr_indices[2*i+1]+1] = 1.0

    elif mode == 'nanobody': # Nanobody-antigen
        cdr_indices = []
        for scf, idx in scaffold_idx.items():
            cdr_indices.append(idx)
        cdr_indices = sorted(cdr_indices)
        for i in range(3):
            scaffold_mask[cdr_indices[2*i]:cdr_indices[2*i+1]+1] = 1.0
            
    return scaffold_mask * batch['res_mask']

def setup_inpainting(feats):
    diffuse_mask = sample_scaffold_mask(feats)
    if 'plddt_mask' in feats:
        diffuse_mask = diffuse_mask * feats['plddt_mask']
    if torch.sum(diffuse_mask) < 1:
        # Should only happen rarely.
        diffuse_mask = torch.ones_like(diffuse_mask)
    feats['diffuse_mask'] = diffuse_mask

def _process_csv_row(processed_file_path, raw_path, scaffold_idx):
    processed_feats = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(processed_feats)

    # make chain sequence list (for the multimer relpos embedding)
    chain_seq_list = []
    pdb_file = raw_path

    p = PDBParser()
    structure = p.get_structure(
        'protein',
        pdb_file,
    )

    for chain in structure.get_chains():
        pdb_seq = "".join([seq1(r.get_resname()) for r in chain.get_residues()])
        chain_seq_list.append(pdb_seq)

    # Run through OpenFold data transforms.
    chain_feats = {
        'aatype': torch.tensor(processed_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(processed_feats['atom_positions']).float(),
        'all_atom_mask': torch.tensor(processed_feats['atom_mask']).float(),
        'seq_mask': torch.tensor(processed_feats['bb_mask']).int()
    }
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles(chain_feats)
    chain_feats = data_transforms.get_chi_angles(chain_feats)
    chain_feats = data_transforms.get_backbone_frames(chain_feats)
    chain_feats['pseudo_beta'] = data_transforms.pseudo_beta_fn(
                                                                chain_feats['aatype'],
                                                                chain_feats['all_atom_positions'],
                                                                None)
    res_plddt = processed_feats['b_factors'][:, 1]
    res_mask = torch.tensor(processed_feats['bb_mask']).int()

    chain_idx = torch.tensor(processed_feats['chain_index'])
    res_idx = processed_feats['residue_index']

    return {
        'res_plddt': torch.tensor(res_plddt),
        'aatype': chain_feats['aatype'],
        'res_mask': res_mask,
        'chain_idx': chain_idx,
        'res_idx': res_idx,
        'scaffold_idx': scaffold_idx,
        'chain_seq_list': chain_seq_list,
        'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'],
        'alt_torsion_angles_sin_cos': chain_feats['alt_torsion_angles_sin_cos'],
        'torsion_angles_mask': chain_feats['torsion_angles_mask'],
        'chi_angles_sin_cos': chain_feats['chi_angles_sin_cos'],
        'chi_mask': chain_feats['chi_mask'],
        'atom14_gt_exists': chain_feats['atom14_gt_exists'],
        'atom14_gt_positions': chain_feats['atom14_gt_positions'], # (L, 14, 3)
        'residx_atom37_to_atom14': chain_feats['residx_atom37_to_atom14'],
        'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
        'atom37_atom_exists': chain_feats['atom37_atom_exists'],
        'pseudo_beta': chain_feats['pseudo_beta'], # require centering 
        'atom14_alt_gt_positions': chain_feats['atom14_alt_gt_positions'], # require centering 
        'atom14_alt_gt_exists': chain_feats['atom14_alt_gt_exists'],
        'atom14_atom_is_ambiguous': chain_feats['atom14_atom_is_ambiguous'],
        'backbone_rigid_mask': chain_feats['backbone_rigid_mask'],
        'rigidgroups_gt_frames': chain_feats['rigidgroups_gt_frames'], # require centering  (L, 8, 4, 4)
        'rigidgroups_gt_exists': chain_feats['rigidgroups_gt_exists'],
        'rigidgroups_alt_gt_frames': chain_feats['rigidgroups_alt_gt_frames'], # require centering (L, 8, 4, 4)
    }

def process_csv_row(csv_row, idx):
    path = csv_row['processed_path']
    raw_path = csv_row['raw_path']
    seq_len = csv_row['seq_len']
    
    masked_chain = None
    first_chain_len = None

    scaffold_idx = {}

    if csv_row['mode'] == 'ab':
        cdr_types = ['h1', 'h2', 'h3', 'l1', 'l2', 'l3']
        for cdr in cdr_types:
            scaffold_idx[f'{cdr}_start'] = int(csv_row[f'{cdr}_start'])
            scaffold_idx[f'{cdr}_end'] = int(csv_row[f'{cdr}_end'])

    if csv_row['mode'] == 'nanobody':
        cdr_types = ['h1', 'h2', 'h3']
        for cdr in cdr_types:
            scaffold_idx[f'{cdr}_start'] = int(csv_row[f'{cdr}_start'])
            scaffold_idx[f'{cdr}_end'] = int(csv_row[f'{cdr}_end'])
    
    processed_row = _process_csv_row(path, raw_path, scaffold_idx)
    processed_row['masked_chain'] = masked_chain
    processed_row['first_chain_len'] = first_chain_len
    processed_row['raw_path'] = raw_path
    processed_row['mode'] = csv_row['mode']
    return processed_row

def get_item(csv, row_idx, res_idx):
    csv_row = csv.iloc[row_idx]
    chain_feats = process_csv_row(csv_row, row_idx)
    feats = chain_feats.copy()

    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
    rotmats_1 = torch.tensor(rigids_1.get_rots().get_rot_mats(), device=rigids_1.device)
    trans_1 = torch.tensor(rigids_1.get_trans(), device=rigids_1.device)

    # Center based on motif locations
    setup_inpainting(feats)

    feats['trans_1'] = trans_1
    feats['rotmats_1'] = rotmats_1
    feats['backbone_rigid_tensor'] = du.create_rigid(rots=feats['rotmats_1'],
                                                        trans=trans_1).to_tensor_4x4()

    # create res_idx for cropping 
    mode = feats['mode']
    
    if mode not in ['ab', 'nanobody', 'general']:
        raise ValueError('Mode should be one of [ab, nanobody, general]')

    if mode == 'ab' and res_idx == None:
        feats['res_idx'] = crop_antigen(feats['trans_1'],
                                        cdr_mask=feats['diffuse_mask'],
                                        nan_mask=feats['res_mask'],
                                        max_len=300,
                                        seq_list=feats['chain_seq_list'],
                                        crop_ab=True,
                                        mode=mode
                                        )
    else:
        feats['res_idx'] = res_idx

    feats['diffuse_mask'] = feats['diffuse_mask'].int()
    
    return feats

def collate_fn(feat):
    cropped_batch = []
    # res_idxs = []
    # trans_1s = []
    # raw_paths = []
    # masking_ratio = self.trainer.datamodule.masking_ratio if hasattr(self, 'trainer') else self.masking_ratio

    cropped_feat = {}
    not_crop_key = ['res_idx', 'scaffold_idx', 'chain_seq_list', 'csv_idx', 'masked_chain', 'first_chain_len', 'raw_path', 'mode']

    for key in feat.keys():
        if key not in not_crop_key:
            cropped_feat[key] = feat[key][feat['res_idx']]

    cropped_feat['res_idx'] = torch.tensor(feat['res_idx'])


    cropped_batch.append(cropped_feat)
    cropped_batch = {key: [d[key] for d in cropped_batch] for key in cropped_batch[0].keys()}   

    for key in cropped_batch.keys():     
        cropped_batch[key] = torch.stack(cropped_batch[key], dim=0)  

    cropped_batch['mode'] = feat['mode']
    cropped_batch['raw_path'] = feat['raw_path']
    cropped_batch['original_diffuse_mask'] = cropped_batch['diffuse_mask']

    return cropped_batch

def calculate_loss(batch, pred_batch, training_cfg):
    loss_mask = batch['res_mask'] * batch['diffuse_mask']
    if torch.any(torch.sum(loss_mask, dim=-1) < 1):
        raise ValueError('Empty batch encountered')

    # Ground truth labels
    gt_trans_1 = batch['trans_1']
    gt_chi_angle = batch['chi_angles_sin_cos']
    gt_atom14_pos = batch['atom14_gt_positions']
    alt_atom14_pos = batch['atom14_alt_gt_positions']
    gt_pseudo_beta = batch['pseudo_beta']
    backbone_rigid_tensor = batch['backbone_rigid_tensor']
    rigidgroups_gt_frames = batch['rigidgroups_gt_frames']
    rigidgroups_alt_gt_frames = batch['rigidgroups_alt_gt_frames']

    # if torch.any(torch.isnan(gt_rot_vf)):
    #     raise ValueError('NaN encountered in gt_rot_vf')

    # Timestep used for normalization.
    r3_t = torch.tensor([1])

    r3_norm_scale = 1 - torch.min(
        r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip))

    
    gt_atom14_pos = gt_atom14_pos * training_cfg.bb_atom_scale / r3_norm_scale[..., None] # scaling 

    alt_atom14_pos = alt_atom14_pos * training_cfg.bb_atom_scale / r3_norm_scale[..., None]
    gt_pseudo_beta = gt_pseudo_beta * training_cfg.bb_atom_scale / r3_norm_scale

    backbone_rigid_tensor[..., :3, 3] = backbone_rigid_tensor[..., :3, 3] * training_cfg.bb_atom_scale / r3_norm_scale
    rigidgroups_gt_frames[..., :3, 3] = rigidgroups_gt_frames[..., :3, 3] * training_cfg.bb_atom_scale / r3_norm_scale[..., None]
    rigidgroups_alt_gt_frames[..., :3, 3] = rigidgroups_alt_gt_frames[..., :3, 3] * training_cfg.bb_atom_scale / r3_norm_scale[..., None]

    # Model output predictions.
    pred_atom14 = pred_batch['atom14_gt_positions']
    align_mask = (1-pred_batch['diffuse_mask'].unsqueeze(-1)).repeat(1, 1, 14)   # align_mask가 0이면 해당 부분은 align 고려 제외 
    align_mask[:, :, 4:] = 0

    aligned_pred_atom14 = do_kabsch(
        rearrange(
            pred_atom14, 
            "b l a d -> b (l a) d",
            a=14),
        rearrange(
            batch['atom14_gt_positions'],
            "b l a d -> b (l a) d",
            a=14),
        rearrange(
            align_mask.bool(),
            "b l a -> b (l a)",
            a=14
        )
        )
    pred_atom14 = rearrange(
        aligned_pred_atom14, "b (l a) d -> b l a d", a=14
    )

    pred_trans_1 = pred_atom14[:, :, 1]
    pred_atom_14_list = [pred_atom14]
    pred_atom_14_list = [pred * training_cfg.bb_atom_scale / r3_norm_scale[..., None] for pred in pred_atom_14_list]
    pred_atom_14_list = torch.stack(pred_atom_14_list, dim=0) # (O, B, L, A, 3)


    pred_angles_list = [pred_batch['chi_angles_sin_cos']]
    norm_denom = torch.sqrt(
        torch.clamp(
            torch.sum(pred_batch['chi_angles_sin_cos']**2, dim=-1, keepdim=True),
            min=1.0e-07
        )
    )

    pred_unnormalized_angles_list = [pred_batch['chi_angles_sin_cos'] / norm_denom] 
    pred_angles_list = torch.stack(pred_angles_list, dim=0)
    pred_unnormalized_angles_list = torch.stack(pred_unnormalized_angles_list, dim=0)

    # if torch.any(torch.isnan(pred_rots_vf)):
    #     raise ValueError('NaN encountered in pred_rots_vf')

    # Get the renamed ground truth 
    renamed_dict = compute_renamed_ground_truth(batch,
                                                atom14_pred_positions=pred_batch['atom14_gt_positions'])

    alt_naming_is_better = renamed_dict['alt_naming_is_better'].clone()
    renamed_atom14_gt_exists = renamed_dict['renamed_atom14_gt_exists'].clone()
    renamed_atom14_gt_positions = renamed_dict['renamed_atom14_gt_positions'] * training_cfg.bb_atom_scale / r3_norm_scale[..., None]

    # Translation VF loss
    loss_denom = torch.sum(loss_mask, dim=-1) * 3
    trans_error = (gt_trans_1 - pred_trans_1) / r3_norm_scale * training_cfg.trans_scale
    trans_loss = training_cfg.translation_loss_weight * torch.sum(
        trans_error ** 2 * loss_mask[..., None],
        dim=(-1, -2)
    ) / loss_denom
 


    # Backbone atom loss
    bb_atom_loss = torch.zeros(gt_atom14_pos.shape[0], device=gt_atom14_pos.device)
    if training_cfg.aux_loss_use_bb_loss:
        bb_atom_loss = compute_rmsd(pred_atom_14_list,
                                renamed_atom14_gt_positions,
                                cdr_mask=batch['diffuse_mask'],
                                atom14_gt_exists=renamed_atom14_gt_exists,
                                mode='bb',
                                compute_non_cdr=False
                                )
                            
    # sc atom loss (final layer만 계산)
    sc_atom_loss = torch.zeros(gt_atom14_pos.shape[0], device=gt_atom14_pos.device)
    if training_cfg.aux_loss_use_sc_atom_loss:
        sc_atom_loss = compute_rmsd(pred_atom_14_list[-1].unsqueeze(0),
                                renamed_atom14_gt_positions,
                                cdr_mask=batch['diffuse_mask'],
                                atom14_gt_exists=renamed_atom14_gt_exists,
                                mode='sc',
                                compute_non_cdr=True
                                )    

    # torsion angle loss 
    chi_loss = torch.zeros(gt_atom14_pos.shape[0], device=gt_atom14_pos.device)
    if training_cfg.aux_loss_use_chi_loss:
        chi_loss = supervised_chi_loss(pred_angles_list,
                                    pred_unnormalized_angles_list,
                                    batch['aatype'],
                                    batch['res_mask'],
                                    batch['chi_mask'],
                                    gt_chi_angle,
                                    chi_weight=0.5,
                                    angle_norm_weight=0.02,
                                    cdr_mask=batch['diffuse_mask']
                                    )

    # final layer backbone rmsd loss
    final_bb_rmsd = compute_rmsd(pred_atom_14_list[-1].unsqueeze(0),
                            renamed_atom14_gt_positions,
                            cdr_mask=batch['diffuse_mask'],
                            atom14_gt_exists=renamed_atom14_gt_exists,
                            mode='bb',
                            compute_non_cdr=False
                            )

    final_layer_rmsd = final_bb_rmsd * (training_cfg.aux_loss_bb_atom_loss_weight/2)

    # local Pairwise distance loss (final layer만 계산)
    local_dist_mat_loss = torch.zeros(gt_atom14_pos.shape[0], device=gt_atom14_pos.device)
    scale_factor = training_cfg.bb_atom_scale / (1 - torch.min(
    r3_t, torch.tensor(training_cfg.t_normalize_clip)))

    if training_cfg.aux_loss_use_local_dist_mat_loss:
        pred_atom_14 = pred_atom_14_list[-1]
        local_dist_mat_loss, neighbor_indices, cdr_residues = local_distance_loss(
            pred_atom_14, # scaled 
            renamed_atom14_gt_exists,
            renamed_atom14_gt_positions, # scaled 
            batch['original_diffuse_mask'][0],
            scale_factor,
            batch['mode']
        )   # local_loss_mask: (B, N, N, 14)
    

    # all atom clash loss 
    all_atom_clash_loss = torch.zeros(gt_atom14_pos.shape[0], device=gt_atom14_pos.device)
    if training_cfg.aux_loss_use_all_atom_clash_loss:
        all_atom_clash_loss = compute_all_atom_clash_loss(
                                                        pred_batch['atom14_gt_positions'],
                                                        batch['atom14_gt_exists'],
                                                        batch['res_idx'],
                                                        batch['residx_atom14_to_atom37'])


    # calculate auxiliary loss 
    se3_vf_loss = trans_loss 
    auxiliary_loss = (
        bb_atom_loss * training_cfg.aux_loss_use_bb_loss * training_cfg.aux_loss_bb_atom_loss_weight
        + sc_atom_loss * training_cfg.aux_loss_use_sc_atom_loss * training_cfg.aux_loss_sc_atom_loss_weight
        + chi_loss * training_cfg.aux_loss_use_chi_loss * training_cfg.aux_loss_chi_loss_weight 
        + local_dist_mat_loss * training_cfg.aux_loss_use_local_dist_mat_loss * training_cfg.aux_loss_local_dist_mat_loss_weight
        + final_layer_rmsd * training_cfg.aux_loss_use_final_layer_rmsd * training_cfg.aux_loss_final_layer_rmsd_weight
        # + bb_fape_loss * training_cfg.aux_loss_use_fape_bb_loss * training_cfg.aux_loss_fape_bb_loss_weight 
        # + sc_fape_loss * training_cfg.aux_loss_use_fape_sc_loss * training_cfg.aux_loss_fape_sc_loss_weight 
    )

    # calculate violation loss
    violation_loss = (
        all_atom_clash_loss * training_cfg.aux_loss_use_all_atom_clash_loss * training_cfg.aux_loss_all_atom_clash_loss_weight
    )

    auxiliary_loss *= training_cfg.aux_loss_weight

    se3_vf_loss = se3_vf_loss + auxiliary_loss + violation_loss

    return {
        "trans_loss": trans_loss,
        "bb_atom_loss": bb_atom_loss,
        'sc_atom_loss': sc_atom_loss,
        'chi_loss': chi_loss,
        'all_atom_clash_loss': all_atom_clash_loss,
        'local_dist_mat_loss': local_dist_mat_loss,
        "se3_vf_loss": se3_vf_loss,
    }

def main(args):
    csv_path = args.csv 
    cfg_path = args.cfg

    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.experiment.training
    df = pd.read_csv(csv_path)
    num_rows = len(df)

    for row_idx in range(num_rows):
        print(f'{row_idx}-th data starts to be analyzed')
        csv_row = df.iloc[row_idx]

        # construct gt batch
        feats = get_item(df, row_idx, None)
        res_idx = feats['res_idx']
        batch = collate_fn(feats)

        # contruct pred batch and calculate loss
        pdb_name = csv_row['pdb_name']
        pred_csv_path = f'/home/psh/protein-frame-flow/pred_pkls/{pdb_name}/metadata.csv'
        pred_df = pd.read_csv(pred_csv_path)
        pred_num_rows = len(pred_df)

        for pred_row_idx in range(pred_num_rows):
            pred_csv_row = pred_df.iloc[pred_row_idx]

            if 'af3' not in pred_csv_row['raw_path']:
                continue
            write_path = pred_csv_row['raw_path'].replace('.pdb', '_loss.json')
            
            # if os.path.exists(write_path):
            #     continue
            
            pred_feats = get_item(pred_df, pred_row_idx, res_idx)
            pred_batch = collate_fn(pred_feats)

            loss_dict = calculate_loss(batch, pred_batch, training_cfg=cfg)
            loss_dict = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in loss_dict.items()}
            
            # 딕셔너리를 JSON 파일로 저장
            with open(write_path, 'w') as f:
                json.dump(loss_dict, f, indent=4)  # indent=4는 예쁘게 들여쓰기
            
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
            
        