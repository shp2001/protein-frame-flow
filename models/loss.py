import torch 
from typing import Optional, Dict

from data import residue_constants
from openfold.utils.loss import between_residue_clash_loss, within_residue_violations
from openfold.data.data_transforms import pseudo_beta_fn
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.utils.tensor_utils import permute_final_dims

from models.utils import calc_distogram
from data.motif_index import find_anchor
import analysis.utils as au

from itertools import combinations_with_replacement

import torch

import os 
import matplotlib.pyplot as plt 
import seaborn as sns 

def get_unique_filepath(output_path):
    """
    Checks if output_path exists and appends _number if it does.
    
    Args:
        output_path (str): Desired file path (e.g., 'contact_map.png')
    
    Returns:
        str: Unique file path (e.g., 'contact_map_1.png' if 'contact_map.png' exists)
    """
    base, ext = os.path.splitext(output_path)
    counter = 0
    new_path = output_path
    
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{ext}"
    
    return new_path

def visualize_contact_map(contact_map, cdr_residues=None, neighbor=None, title="Contact Map", output_path="/home/psh/protein-frame-flow/experiments/loss_mask_Cb_CH2/contact_map.png"):
    """
    Visualizes a [L, L, 14] contact map as a 2D heatmap by reducing the 14 atom-pair dimension
    and saves it to a unique file path.
    
    Args:
        contact_map (torch.Tensor): [L, L, 14] tensor (gt_contact_map * local_loss_mask)[0]
        title (str): Title of the plot
        output_path (str): Path to save the output image (e.g., 'contact_map.png')
    """
    # [L, L, 14] -> [L, L]로 축소: 14개 원자 쌍 중 하나라도 1이면 1로 설정
    contact_map_2d = contact_map[:, :, 50]  # [L, L]

    # 히트맵 그리기
    plt.figure(figsize=(6, 3))

    if cdr_residues != None:
        sns.heatmap(
            contact_map_2d.cpu().numpy(), 
            cmap="Reds", 
            cbar=True, 
            xticklabels=neighbor.cpu().numpy().tolist(), 
            yticklabels=cdr_residues.cpu().numpy().tolist(),
            square=True
        )
    else:
        sns.heatmap(
            contact_map_2d.cpu().numpy(), 
            cmap="Reds", 
            cbar=True, 
            square=True
        )

    plt.title(title)
    plt.xlabel("Neighbors Index", fontsize=6)
    plt.ylabel("H3 CDR Index", fontsize=6)
    plt.xticks(rotation=90, fontsize=4)
    plt.yticks(rotation=0, fontsize=4) 

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 고유한 파일 경로 생성
    unique_output_path = get_unique_filepath(output_path)

    # 이미지 저장
    plt.savefig(unique_output_path, dpi=300, bbox_inches="tight")

    # 플롯 닫기 (메모리 관리)
    plt.close()

def compute_renamed_ground_truth(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """
    Find optimal renaming of ground truth based on the predicted positions.

    Alg. 26 "renameSymmetricGroundTruthAtoms"

    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.

    Args:
      batch: Dictionary containing:
        * atom14_gt_positions: Ground truth positions.
        * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
        * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
            renaming swaps.
        * atom14_gt_exists: Mask for which atoms exist in ground truth.
        * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
            after renaming.
        * atom14_atom_exists: Mask for whether each atom is part of the given
            amino acid type.
      atom14_pred_positions: Array of atom positions in global frame with shape
    Returns:
      Dictionary containing:
        alt_naming_is_better: Array with 1.0 where alternative swap is better.
        renamed_atom14_gt_positions: Array of optimal ground truth positions
          after renaming swaps are performed.
        renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """

    pred_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_gt_positions = batch["atom14_gt_positions"]
    gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_gt_positions[..., None, :, None, :]
                - atom14_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_alt_gt_positions = batch["atom14_alt_gt_positions"]
    alt_gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_alt_gt_positions[..., None, :, None, :]
                - atom14_alt_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    lddt = torch.sqrt(eps + (pred_dists - gt_dists) ** 2)
    alt_lddt = torch.sqrt(eps + (pred_dists - alt_gt_dists) ** 2)

    atom14_gt_exists = batch["atom14_gt_exists"]
    atom14_atom_is_ambiguous = batch["atom14_atom_is_ambiguous"]
    mask = (
        atom14_gt_exists[..., None, :, None]
        * atom14_atom_is_ambiguous[..., None, :, None]
        * atom14_gt_exists[..., None, :, None, :]
        * (1.0 - atom14_atom_is_ambiguous[..., None, :, None, :])
    )

    per_res_lddt = torch.sum(mask * lddt, dim=(-1, -2, -3))
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))

    fp_type = atom14_pred_positions.dtype
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(fp_type)

    renamed_atom14_gt_positions = (
        1.0 - alt_naming_is_better[..., None, None]
    ) * atom14_gt_positions + alt_naming_is_better[
        ..., None, None
    ] * atom14_alt_gt_positions

    renamed_atom14_gt_mask = (
        1.0 - alt_naming_is_better[..., None]
    ) * atom14_gt_exists + alt_naming_is_better[..., None] * batch[
        "atom14_alt_gt_exists"
    ]

    return {
        "alt_naming_is_better": alt_naming_is_better,
        "renamed_atom14_gt_positions": renamed_atom14_gt_positions,
        "renamed_atom14_gt_exists": renamed_atom14_gt_mask,
    }

def masked_mean(mask, value, dim, eps=1e-4):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))

def supervised_chi_loss(
    angles_sin_cos: torch.Tensor,
    unnormalized_angles_sin_cos: torch.Tensor,
    aatype: torch.Tensor,
    seq_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    chi_angles_sin_cos: torch.Tensor,
    chi_weight: float,
    angle_norm_weight: float,
    cdr_mask: torch.Tensor,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    """
    Implements Algorithm 27 (torsionAngleLoss)

    Args:
        angles_sin_cos:
            [*, N, 7, 2] predicted angles
        unnormalized_angles_sin_cos:
            The same angles, but unnormalized
        aatype:
            [*, N] residue indices
        seq_mask:
            [*, N] sequence mask
        chi_mask:
            [*, N, 7] angle mask
        chi_angles_sin_cos:
            [*, N, 7, 2] ground truth angles
        chi_weight:
            Weight for the angle component of the loss
        angle_norm_weight:
            Weight for the normalization component of the loss
        cdr_mask:
            [*, N] cdr mask
    Returns:
        [*] loss tensor
    """

    pred_angles = angles_sin_cos[..., 3:, :] # (O, B, L, 4, 2)
    residue_type_one_hot = torch.nn.functional.one_hot(
        aatype,
        residue_constants.restype_num + 1,
    )
    chi_pi_periodic = torch.einsum(
        "...ij,jk->ik",
        residue_type_one_hot.type(angles_sin_cos.dtype),
        angles_sin_cos.new_tensor(residue_constants.chi_pi_periodic),
    )

    true_chi = chi_angles_sin_cos[None]  # (1, B, L, 4, 2)
    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum((true_chi_shifted - pred_angles) ** 2, dim=-1)
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)  # (O, B, L, 4)
    
    # The ol' switcheroo
    sq_chi_error = sq_chi_error.permute(
        *range(len(sq_chi_error.shape))[1:-2], 0, -2, -1
    ) # (B, O, L, 4)

    sq_chi_loss = masked_mean(chi_mask[..., None, :, :], sq_chi_error, dim=(-1, -2, -3))
    loss = chi_weight * sq_chi_loss
    print(f'sq_chi_loss: {sq_chi_loss}')
    
    ## cdr_mask for sq_chi_error
    cdr_chi_mask = cdr_mask.unsqueeze(-1) * chi_mask
    cdr_chi_loss = masked_mean(
        cdr_chi_mask[..., None, :, :], sq_chi_error, dim=(-1,-2,-3)
        )
    loss += chi_weight * cdr_chi_loss

    ## cdr_mask for sq_chi_loss
    # (B, n) -> (B, n, 7) or (B, n, 4)
    chi_cdr_mask = cdr_mask.unsqueeze(-1).repeat(1, 1, chi_mask.shape[-1])
    chi_cdr_mask = chi_cdr_mask * chi_mask
    sq_chi_loss += masked_mean(chi_cdr_mask[..., None, :, :], sq_chi_error, dim=(-1, -2, -3))

    angle_norm = torch.sqrt(torch.sum(unnormalized_angles_sin_cos**2, dim=-1) + eps)
    norm_error = torch.abs(angle_norm - 1.0)
    norm_error = norm_error.permute(*range(len(norm_error.shape))[1:-2], 0, -2, -1)
    angle_norm_loss = masked_mean(
        seq_mask[..., None, :, None], norm_error, dim=(-1, -2, -3)
    )

    loss = loss + angle_norm_weight * angle_norm_loss
    print(f'angle_norm_loss: {angle_norm_loss}')
    # # Average over the batch dimension
    # loss = torch.mean(loss)

    return loss

def compute_rmsd(
    pred, # (o, b, l, 14, 3)
    aligned_target, # (b, l, 14, 3)
    cdr_mask, # (b, l)  <- cdr: 1 fv: 0
    atom14_gt_exists, # (b, l, 14) <- exists: 1 non-exists: 0
    mode,
    cdr_clamp=30,
    compute_non_cdr=False
):

    if mode == 'bb':
        pred = pred[:, :, :, :3]
        aligned_target = aligned_target[:, :, :3]
        atom14_gt_exists = atom14_gt_exists[:, :, :3]
    else:
        pred = pred[:, :, :, 3:]
        aligned_target = aligned_target[:, :, 3:]
        atom14_gt_exists = atom14_gt_exists[:, :, 3:]

    mse = torch.nn.functional.mse_loss(
        pred,
        aligned_target[None, ...],
        reduction='none',
    ).mean(-1) # (o, b, L, a)

    mask = cdr_mask[..., None] * atom14_gt_exists # (b, L, a)
    cdr_mse = mse * mask[None, ...] # (o, b, L, a)

    cdr_mse = cdr_mse.permute(1,0,2,3) # (b, o, L, a)

    if cdr_clamp > 0:
        cdr_mse = torch.clamp(cdr_mse, max=cdr_clamp**2)

    cdr_mse = torch.sum(
        cdr_mse,
        dim=(-1,-2,-3),
    ) / (torch.sum(
        mask,
        dim=(-1,-2),
    ) * 3)
    
    cdr_mse = torch.sqrt(cdr_mse) # (b)

    if compute_non_cdr:
        mask = (1-cdr_mask[..., None]) * atom14_gt_exists # (b, L, a)
        non_cdr_mse = mse * mask[None, ...]

        non_cdr_mse = non_cdr_mse.permute(1,0,2,3) # (b, o, L, a)

        if cdr_clamp > 0:
            non_cdr_mse = torch.clamp(non_cdr_mse, max=cdr_clamp**2)

        non_cdr_mse = torch.sum(
            non_cdr_mse,
            dim=(-1,-2,-3),
        ) / (torch.sum(
            mask,
            dim=(-1,-2),
        ) * 3)
        

        non_cdr_mse = torch.sqrt(non_cdr_mse) # (b)
        cdr_mse = non_cdr_mse + cdr_mse 

    return cdr_mse

def cdr_clamp(error_dist, l1_clamp_distance, l1_clamp_distance_large, cdr_mask):
    intra_clamped = torch.clamp(error_dist, min=0, max=l1_clamp_distance_large)
    inter_clamped = torch.clamp(error_dist, min=0, max=l1_clamp_distance)
    error_dist = torch.where(cdr_mask.bool(), intra_clamped, inter_clamped)
    return error_dist

def compute_fape(
    pred_frames: Rigid,  
    target_frames: Rigid,
    frames_mask: torch.Tensor, # (1, B, L, 8)
    pred_positions: torch.Tensor, # (O, B, L, 14, 3)
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor, # (1, B, L, 14)
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    l1_clamp_distance_large: Optional[float] = None,
    cdr_mask: Optional[torch.Tensor] = None,
    eps=1e-8,
) -> torch.Tensor:
    """
    Computes FAPE loss.

    Args:
        pred_frames:
            [*, N_frames] Rigid object of predicted frames
        target_frames:
            [*, N_frames] Rigid object of ground truth frames
        frames_mask:
            [*, N_frames] binary mask for the frames
        pred_positions:
            [*, N_pts, 3] predicted atom positions
        target_positions:
            [*, N_pts, 3] ground truth positions
        positions_mask:
            [*, N_pts] positions mask
        length_scale:
            Length scale by which the loss is divided
        l1_clamp_distance:
            Cutoff above which distance errors are disregarded
        eps:
            Small value used to regularize denominators
    Returns:
        [*] loss tensor
    """

    # [*, N_frames, N_pts, 3] -> [O, B, L, L, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )
    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    ) # [O, B, L, L]

    if l1_clamp_distance is not None:
        if cdr_mask is not None and l1_clamp_distance_large is not None:
            error_dist = cdr_clamp(
                error_dist, l1_clamp_distance, l1_clamp_distance_large, cdr_mask
            )

        else:
            error_dist = torch.clamp(error_dist, max=l1_clamp_distance)

    normed_error = error_dist / length_scale # [*, N_frames, N_pts, 3] -> [O, B, L, L]
    normed_error = normed_error * frames_mask[..., None] # [O, B, L, L] * (1, B, L, 1)
    normed_error = normed_error * positions_mask[..., None, :] # [O, B, L, L] * (1, B, 1, L)

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter)

    normed_error = torch.sum(normed_error, dim=-1) # [O, B, L]
    normed_error = normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None] # [O, B, L]
    normed_error = torch.sum(normed_error, dim=-1) # (O, B)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1)) # (O, B)
    
    return normed_error

# calculate only cdr-backbone loss 
def backbone_fape_loss(
    backbone_rigid_tensor: torch.Tensor, # (B, L, 4, 4)
    backbone_rigid_mask: torch.Tensor, # (B, L)
    traj: torch.Tensor, # (O, B, L, 8, 7), pred frames 
    cdr_mask: torch.Tensor, # (B, L)
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    intercdr_distance: Optional[float] = 30.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    pred_aff = Rigid.from_tensor_7(traj)
    pred_aff = Rigid(
        Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
        pred_aff.get_trans(),
    )

    # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of
    # backbone tensor, normalizes it, and then turns it back to a rotation
    # matrix. To avoid a potentially numerically unstable rotation matrix
    # to quaternion conversion, we just use the original rotation matrix
    # outright. This one hasn't been composed a bunch of times, though, so
    # it might be fine.
    gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    
    if use_clamped_fape:
        l1_clamp_distance = clamp_distance
    else:
        l1_clamp_distance = None

    # fape_loss = compute_fape(
    #     pred_aff,
    #     gt_aff[None],
    #     backbone_rigid_mask[None],
    #     pred_aff.get_trans(),
    #     gt_aff[None].get_trans(),
    #     backbone_rigid_mask[None],
    #     l1_clamp_distance=l1_clamp_distance,
    #     length_scale=loss_unit_distance,
    #     eps=eps,
    # )

    cdr_fape_loss = compute_fape(
        pred_frames=pred_aff,
        target_frames=gt_aff[None],
        frames_mask=cdr_mask[None] * backbone_rigid_mask[None],
        pred_positions=pred_aff.get_trans(),
        target_positions=gt_aff[None].get_trans(),
        positions_mask=cdr_mask[None] * backbone_rigid_mask[None],
        l1_clamp_distance=l1_clamp_distance,
        l1_clamp_distance_large=intercdr_distance,
        cdr_mask=cdr_mask,
        length_scale=loss_unit_distance,
        eps=eps,
    ) # (O, B, L, 8)

    # fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
    #     1 - use_clamped_fape
    # )

    # Average over the dimensions except for batch dimension 
    cdr_fape_loss = torch.mean(cdr_fape_loss, axis=0) 

    return cdr_fape_loss


def sidechain_fape_loss(
    sidechain_frames: torch.Tensor,
    sidechain_atom_pos: torch.Tensor,
    rigidgroups_gt_frames: torch.Tensor,
    rigidgroups_alt_gt_frames: torch.Tensor,
    rigidgroups_gt_exists: torch.Tensor,
    renamed_atom14_gt_positions: torch.Tensor,
    renamed_atom14_gt_exists: torch.Tensor,
    alt_naming_is_better: torch.Tensor,
    cdr_mask: torch.Tensor,
    clamp_distance: float = 10.0,
    intercdr_distance: Optional[float] = 30.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    # renamed_gt_frames is same shape as rigidgroups_gt_frames and
    # rigidgroups_alt_gt_frames which is (B, n, 8, 4, 4)
    renamed_gt_frames = (
        1.0 - alt_naming_is_better[..., None, None, None]
    ) * rigidgroups_gt_frames + alt_naming_is_better[
        ..., None, None, None
    ] * rigidgroups_alt_gt_frames

    # Steamroll the inputs

    ## construct sidechain_frames
    # (L, B, n, 8, 4, 4) -> (B, n, 8, 4, 4)
    sidechain_frames = sidechain_frames[-1]
    # torch.size([B])
    batch_dims = sidechain_frames.shape[:-4]
    # (B, n, 8, 4, 4) -> (B, n * 8, 4, 4)
    sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
    # (B, n * 8, 4, 4) -> (B, n * 8) Rigids
    sidechain_frames = Rigid.from_tensor_4x4(sidechain_frames)

    ## construct renamed_gt_frames
    # (B, n, 8, 4, 4) -> (B, n * 8, 4, 4)
    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    # (B, n * 8, 4, 4) -> (B, n * 8) Rigids
    renamed_gt_frames = Rigid.from_tensor_4x4(renamed_gt_frames)

    ## rigidgroups_gt_exists
    # (B, n, 8) -> (B, n * 8)
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)

    ## sidechain_atom_pos
    # (L, B, n, 14, 3) -> (B, n, 14, 3)
    sidechain_atom_pos = sidechain_atom_pos[-1]
    # (B, n, 14, 3) -> (B, n * 14, 3)
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)

    ## renamed_atom14_gt_positions
    # (B, n, 14, 3) -> (B, n * 14, 3)
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(*batch_dims, -1, 3)

    ## renamed_atom14_gt_exists
    # (B, n, 14) -> (B, n * 14)
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)

    ## cdr_mask
    # (B, n) -> (B, n, 8)
    cdr_mask_frame_dim = cdr_mask.unsqueeze(-1).repeat(1, 1, 8)

    # (B, n, 8) -> (B, n * 8)
    cdr_mask_frame_dim = cdr_mask_frame_dim.view(*batch_dims, -1)
    sidechain_cdr_frames_mask = rigidgroups_gt_exists * cdr_mask_frame_dim

    # (B, n) -> (B, n, 14)
    cdr_mask_pos_dim = cdr_mask.unsqueeze(-1).repeat(1, 1, 14)

    # (B, n, 14) -> (B, n * 14)
    cdr_mask_pos_dim = cdr_mask_pos_dim.view(*batch_dims, -1)
    sidechain_cdr_points_mask = renamed_atom14_gt_exists * cdr_mask_pos_dim

    # (B, n) -> (B, n * 8, n * 14)
    cdr_mask = cdr_mask_frame_dim.unsqueeze(-1) != (
        cdr_mask_pos_dim.unsqueeze(-2)
    )
    fape = compute_fape(
        sidechain_frames,
        renamed_gt_frames,
        rigidgroups_gt_exists,
        sidechain_atom_pos,
        renamed_atom14_gt_positions,
        renamed_atom14_gt_exists,
        l1_clamp_distance=clamp_distance,
        l1_clamp_distance_large=intercdr_distance,
        cdr_mask=cdr_mask,
        length_scale=length_scale,
        eps=eps,
    )

    cdr_fape = compute_fape(
        sidechain_frames,
        renamed_gt_frames,
        sidechain_cdr_frames_mask,
        sidechain_atom_pos,
        renamed_atom14_gt_positions,
        sidechain_cdr_points_mask,
        l1_clamp_distance=clamp_distance,
        l1_clamp_distance_large=intercdr_distance,
        cdr_mask=cdr_mask,
        length_scale=length_scale,
        eps=eps,
    )


    return (fape + cdr_fape) / 2

# def compute_prmsd_loss(
#     pdev, # prmsd (b, l)
#     pred_position, # predicted structure (b, l, 14, 3)
#     atom14_gt_positions, # gt stucture  (b, l, 14, 3)
#     cdr_mask, # (b, l)
# ):
#     print(f'pdev: {pdev[0]}')
#     ca_pred = pred_position[:, :, 1] # (b, l, 3)
#     ca_target = atom14_gt_positions[:, :, 1]

#     bb_dev = (ca_pred - ca_target).norm(dim=-1) # (b, l)

#     loss = torch.nn.functional.l1_loss(
#         pdev,
#         bb_dev,
#         reduction='none',
#     )

#     cdr_loss = torch.sum(
#         loss * cdr_mask,
#         dim=-1,
#     ) / (torch.sum(
#         cdr_mask,
#         dim=-1,
#     )) # (b)
#     print(f'cdr_loss: {cdr_loss}')
#     debug_loss = torch.sum(loss, dim=-1) / bb_dev.shape[1]
#     return debug_loss + cdr_loss

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss

def compute_prmsd(prmsd: torch.Tensor,
                  cdr_mask: torch.Tensor) -> torch.Tensor:
    """Computes plddt from the model output. The output is a histogram of unnormalised
    plddt.

    Args:
        plddt (torch.Tensor): (B, n, 50) output from the model

    Returns:
        torch.Tensor: (B, n) plddt scores
    """
    pdf = torch.nn.functional.softmax(prmsd, dim=-1)
    vbins = torch.linspace(0, 15, steps=50).to(prmsd.device).float()
    output = pdf @ vbins  # (B, n)
    if cdr_mask is not None:
        output[cdr_mask == 0] = 0.0

    return output

def compute_prmsd_loss(
    logits: torch.Tensor, # prmsd (b, L, 50)
    all_atom_pred_pos: torch.Tensor, 
    all_atom_positions: torch.Tensor, # atom14_renamed_positions
    all_atom_mask: torch.Tensor, # atom14_renamed_exists
    cdr_mask: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos]  # keep dim

    dev = torch.sqrt(torch.sum((all_atom_pred_pos - all_atom_positions) ** 2, dim=-1))
    mask = cdr_mask * all_atom_mask # (b, L)

    # bin 경계: [0.0, 0.2, 0.4, ..., 10.0]
    bin_width = cutoff / (no_bins-1)
    bin_index = torch.floor(torch.minimum(dev, torch.tensor(cutoff)) / bin_width).long()
    dev_one_hot = torch.nn.functional.one_hot(bin_index, num_classes=no_bins)
    errors = softmax_cross_entropy(logits, dev_one_hot) # (B, L)


    loss = torch.sum(errors * mask, dim=-1) / (
        eps + torch.sum(mask, dim=-1)
    )

    return loss

def compute_plddt(logits: torch.Tensor, # (b, L, 50)
                  cdr_mask = None) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1) # (b, L)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    ) # (b, L)
    if cdr_mask is not None:
        pred_lddt_ca[cdr_mask == 0] = 1.0
    return pred_lddt_ca * 100

def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cdr_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_positions[..., None, :] - all_atom_positions[..., None, :, :])
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_pred_pos[..., None, :] - all_atom_pred_pos[..., None, :, :]) ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)
    # if len(torch.nonzero(cdr_mask[0])) > 20:
    #     print(dist_l1[0, 80:100, 80:100])
    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score

def lddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor, # (b, L, 14)
    cdr_mask: torch.Tensor, # (b, L)
    cutoff: float = 15.0,
    no_bins: int = 50,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]

    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    score = lddt(
        all_atom_pred_pos, all_atom_positions, all_atom_mask, cutoff=cutoff, eps=eps, cdr_mask=cdr_mask
    )

    score = score.detach()

    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(bin_index, num_classes=no_bins)
    # if len(torch.nonzero(cdr_mask[0])) > 20:
    #     print(f'lddt_ca_one_hot: {torch.nonzero(lddt_ca_one_hot[0])}')
    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)

    all_atom_mask = all_atom_mask * cdr_mask
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (
        eps + torch.sum(all_atom_mask, dim=-1)
    )

    return loss

def compute_all_atom_clash_loss(
        atom14_pred_positions,
        atom14_atom_exists,
        residue_index,
        residx_atom14_to_atom37,
        interface_mask):

    atomtype_radius = [
        residue_constants.van_der_waals_radius[name[0]]
        for name in residue_constants.atom_types
    ]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)
    atom14_atom_radius = (
        atom14_atom_exists
        * atomtype_radius[residx_atom14_to_atom37]
    )
    between_residue_clashes = between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=residue_index,
        interface_mask=interface_mask
    )

    # between residue clashes = {
    # "mean_loss": mean_loss,  # shape (B)
    # "per_atom_loss_sum": per_atom_loss_sum,  # shape (B, N, 14)
    # "per_atom_clash_mask": per_atom_clash_mask,  # shape (B, N, 14) }

    return between_residue_clashes['mean_loss']

def compute_within_clash_loss(       
        atom14_pred_positions,
        atom14_atom_exists,
        interface_mask,
        aatype,
        clash_overlap_tolerance=1.5,
        violation_tolerance_factor=12.0):

    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )
    atom14_dists_lower_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["lower_bound"]
    )[aatype]
    atom14_dists_upper_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["upper_bound"]
    )[aatype]
    
    within_residue_clashes = within_residue_violations(
        atom14_pred_positions,
        atom14_atom_exists,
        atom14_dists_lower_bound,
        atom14_dists_upper_bound,
    )['per_atom_loss_sum'] # ([B, N, 14])

    mean_loss = torch.sum(within_residue_clashes * atom14_atom_exists, dim=(1,2)) / (1e-6 + torch.sum(atom14_atom_exists, dim=(1,2)))
    if interface_mask is not None:
        # interface_mask: (B, L) → (B, L, 1) → (B, L, 14)
        interface_mask_exp = interface_mask[..., None].expand(-1, -1, 14)

        # 평균을 위해 존재하는 CDR atom 수 계산
        interface_exists = atom14_atom_exists * interface_mask_exp  # (B, L, 14)
        per_atom_interface_loss = within_residue_clashes * interface_exists  # (B, L, 14)
        interface_loss = torch.sum(per_atom_interface_loss, dim=(1, 2)) / (1e-6 + torch.sum(interface_exists, dim=(1, 2)))

        # 최종 loss에 더함 (필요시 가중치 사용 가능)
        mean_loss = mean_loss + interface_loss

    return mean_loss

def local_distance_loss(
    atom14_pred_positions, # (B, N, 14, 3)
    renamed_atom14_gt_exists, # (B, N, 14)
    renamed_atom14_gt_positions, # (B, N, 14, 3)
    original_diffuse_mask, # (N)
    scale_factor,
    mode
    ):
    """
    In order to update interface properly, this loss will scan distance among interface atoms.
    """
    cdr_residues, neighbor_indices = au.get_cdr_and_neighbors(
        atom14_pred_positions,
        renamed_atom14_gt_positions,
        renamed_atom14_gt_exists,
        original_diffuse_mask,
        mode,
        scale_factor,
        distance_threshold=5
    )

    # calculate gt distance map 
    device = renamed_atom14_gt_exists.device
    pair_indices = list(combinations_with_replacement(range(14), 2))  # 총 105쌍
    i_idx = torch.tensor([i for i, j in pair_indices], device=device)
    j_idx = torch.tensor([j for i, j in pair_indices], device=device)

    atom_i = renamed_atom14_gt_positions[:, :, i_idx].unsqueeze(2)  # (B, L, 1, 105, 3)
    atom_j = renamed_atom14_gt_positions[:, :, j_idx].unsqueeze(1)  # (B, 1, L, 105, 3)
    gt_distance_map = torch.norm(atom_i - atom_j, dim=-1)  # (B, L, L, 105)
    
    local_gt_pair_dists = gt_distance_map[:, cdr_residues][:, :, neighbor_indices] # (B, N_cdr, N, 105)

    # calculate pred distance map
    pred_atom_i = atom14_pred_positions[:, :, i_idx].unsqueeze(2)  # (B, L, 1, 105, 3)
    pred_atom_j = atom14_pred_positions[:, :, j_idx].unsqueeze(1)  # (B, 1, L, 105, 3)
    pred_distance_map = torch.norm(pred_atom_i - pred_atom_j, dim=-1)  # (B, L, L, 105)

    local_pred_pair_dists = pred_distance_map[:, cdr_residues][:, :, neighbor_indices] # (B, N_cdr, N, 105)

    # make loss mask
    exists_i = renamed_atom14_gt_exists[:, :, i_idx]  # (B, L, 105)
    exists_j = renamed_atom14_gt_exists[:, :, j_idx]  # (B, L, 105)

    mask_i = exists_i.unsqueeze(2)  # (B, L, 1, 105)
    mask_j = exists_j.unsqueeze(1)  # (B, 1, L, 105)
    loss_mask = mask_i * mask_j # (B, L, L, 105)
    local_loss_mask = loss_mask[:, cdr_residues][:, :, neighbor_indices] # (B, N_cdr, N, 105)

    # calculate loss (batch loss)
    dist_err = (local_gt_pair_dists - local_pred_pair_dists) ** 2
    dist_err = dist_err * local_loss_mask
    dist_mat_loss = torch.sum(
        dist_err,
        dim=(-1,-2,-3)
    )
    dist_mat_loss = dist_mat_loss / (torch.sum(local_loss_mask, dim=(-1,-2,-3)) + 1) # (B)
    return dist_mat_loss, neighbor_indices, cdr_residues

def b_carbon_distogram_loss(
    pred_cb_distogram: torch.Tensor,  # (O-2, B, L, L, 64)
    gt_pseudo_beta: torch.Tensor,     # (B, L, 3)
    res_mask: torch.Tensor,
    neighbor_indices, 
    cdr_residues: torch.Tensor, # (N)
    eps: float = 1e-10
):
    # 1. Ground truth distogram 계산 (one-hot 인코딩 포함)
    gt_cb_distogram = calc_distogram(  
        gt_pseudo_beta,
        min_bin=2.0,
        max_bin=22.0,
        num_bins=64
    ).unsqueeze(0) # (B, L, L, 64), one-hot

    # 2. Cross entropy: - sum y * log p
    loss_per_pair = -torch.sum(gt_cb_distogram * torch.log(pred_cb_distogram), dim=-1)  # (O-2, B, L, L)
    loss_per_pair = torch.mean(loss_per_pair, dim=0) # (B, L, L)

    # total loss 
    edge_mask = res_mask[:, None] * res_mask[:, :, None] # (B, L, L)
    masked_loss = loss_per_pair * edge_mask
    loss = torch.sum(masked_loss, dim=(1,2)) / (torch.sum(edge_mask, dim=(1,2)) + eps)

    # local loss  
    local_loss_per_pair = loss_per_pair[:, cdr_residues][:, :, neighbor_indices]
    local_mask = edge_mask[:, cdr_residues][:, :, neighbor_indices]
    local_masked_loss = local_loss_per_pair * local_mask
    local_loss = torch.sum(local_masked_loss, dim=(1,2)) / (torch.sum(local_mask, dim=(1,2)) + eps)

    return local_loss + loss
    

def aa_contact_map_loss(
    pred_aa_contact_map: torch.Tensor,  # (B, L, L, 14)
    renamed_atom14_gt_positions: torch.Tensor,  # (B, L, 14, 3)
    renamed_atom14_gt_exists: torch.Tensor, # (B, L, 14)
    neighbor_indices: torch.Tensor, # (N)
    cdr_residues: torch.Tensor, # (N)
    distance_threshold: float = 10.0,
    eps: float = 1e-10
):

    device = pred_aa_contact_map.device

    # make pairwise all atom contact map 
    pair_indices = list(combinations_with_replacement(range(14), 2))  # 총 105쌍
    i_idx = torch.tensor([i for i, j in pair_indices], device=device)
    j_idx = torch.tensor([j for i, j in pair_indices], device=device)

    atom_i = renamed_atom14_gt_positions[:, :, i_idx]  # (B, L, 105, 3)
    atom_j = renamed_atom14_gt_positions[:, :, j_idx]  # (B, L, 105, 3)

    atom_i = atom_i.unsqueeze(2)  # (B, L, 1, 105, 3)
    atom_j = atom_j.unsqueeze(1)  # (B, 1, L, 105, 3)
    gt_aa_distance_map = torch.norm(atom_i - atom_j, dim=-1)  # (B, L, L, 105)

    gt_contact_map = (gt_aa_distance_map < distance_threshold).float()  # (B, L, L, 105)

    # make pairwise all atom contact map mask 
    exists_i = renamed_atom14_gt_exists[:, :, i_idx]  # (B, L, 105)
    exists_j = renamed_atom14_gt_exists[:, :, j_idx]  # (B, L, 105)

    mask_i = exists_i.unsqueeze(2)  # (B, L, 1, 105)
    mask_j = exists_j.unsqueeze(1)  # (B, 1, L, 105)

    edge_mask = mask_i * mask_j  # (B, L, L, 105)

    # calculate BCE
    loss_per_pair = torch.nn.functional.binary_cross_entropy(
        pred_aa_contact_map,
        gt_contact_map,
        reduction="none"
    )  # (B, L, L, 105)

    # calculate loss
    masked_loss = loss_per_pair * edge_mask
    loss = torch.sum(masked_loss, dim=(1,2,3)) / (torch.sum(edge_mask, dim=(1,2,3)) + eps)

    # calculate local loss
    local_loss_per_pair = loss_per_pair[:, cdr_residues][:, :, neighbor_indices]
    local_loss_mask = edge_mask[:, cdr_residues][:, :, neighbor_indices]
    local_masked_loss = local_loss_per_pair * local_loss_mask
    local_loss = torch.sum(local_masked_loss, dim=(1,2,3)) / (torch.sum(local_loss_mask, dim=(1,2,3)) + eps)

    return loss + local_loss