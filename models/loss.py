import torch 
from typing import Optional

from data import residue_constants
from openfold.utils.loss import between_residue_clash_loss
from openfold.data.data_transforms import pseudo_beta_fn
from openfold.utils.rigid_utils import Rigid, Rotation

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

    ## cdr_mask for angle_norm_loss
    angle_cdr_mask = cdr_mask
    angle_cdr_mask = angle_cdr_mask * seq_mask
    angle_norm_loss += masked_mean(
        angle_cdr_mask[..., None, :, None], norm_error, dim=(-1, -2, -3)
    )

    loss = loss + angle_norm_weight * angle_norm_loss

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

    mse = torch.nn.functional.mse_loss(
        pred,
        aligned_target[None, ...],
        reduction='none',
    ).mean(-1) # (o, b, L, a)


    if mode == 'bb':
        atom14_gt_exists = atom14_gt_exists[:, :, :3]
    else:
        atom14_gt_exists = atom14_gt_exists[:, :, 3:]

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
    error_dist = torch.where(cdr_mask, intra_clamped, inter_clamped)
    return error_dist

def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    l1_clamp_distance_large: Optional[float] = None,
    cdr_mask: Optional[float] = None,
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

    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        if cdr_mask is not None and l1_clamp_distance_large is not None:
            error_dist = cdr_clamp(
                error_dist, l1_clamp_distance, l1_clamp_distance_large, cdr_mask
            )

        else:
            error_dist = torch.clamp(error_dist, max=l1_clamp_distance)

    normed_error = error_dist / length_scale

    normed_error = normed_error * frames_mask[..., None]

    
    normed_error = normed_error * positions_mask[..., None, :]

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter)

    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error

def cdr_clamp(error_dist, l1_clamp_distance, l1_clamp_distance_large, cdr_mask):
    intra_clamped = torch.clamp(error_dist, min=0, max=l1_clamp_distance_large)
    inter_clamped = torch.clamp(error_dist, min=0, max=l1_clamp_distance)
    error_dist = torch.where(cdr_mask, intra_clamped, inter_clamped)
    return error_dist

# calculate only cdr-backbone loss 
def backbone_fape_loss(
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    traj: torch.Tensor,
    cdr_mask: torch.Tensor,
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

    fape_loss = compute_fape(
        pred_aff,
        gt_aff[None],
        backbone_rigid_mask[None],
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        backbone_rigid_mask[None],
        l1_clamp_distance=l1_clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )

    cdr_fape_loss = compute_fape(
        pred_aff,
        gt_aff[None],
        cdr_mask[None], # backbone_rigid_mask[None]
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        cdr_mask[None], # backbone_rigid_mask[None]
        l1_clamp_distance=l1_clamp_distance,
        l1_clamp_distance_large=intercdr_distance,
        cdr_mask=cdr_mask,
        length_scale=loss_unit_distance,
        eps=eps,
    )

    # fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
    #     1 - use_clamped_fape
    # )

    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)
    cdr_fape_loss = torch.mean(cdr_fape_loss)

    return fape_loss + cdr_fape_loss


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
    side_chain_frames_cdr_mask = cdr_mask.unsqueeze(-1).repeat(1, 1, 8)
    # (B, n, 8) -> (B, n * 8)
    side_chain_frames_cdr_mask = side_chain_frames_cdr_mask.view(*batch_dims, -1)
    # (B, n) -> (B, n, 14)
    sidechain_atom_pos_cdr_mask = cdr_mask.unsqueeze(-1).repeat(1, 1, 14)
    # (B, n, 14) -> (B, n * 14)
    sidechain_atom_pos_cdr_mask = sidechain_atom_pos_cdr_mask.view(*batch_dims, -1)
    # (B, n) -> (B, n * 8, n * 14)
    cdr_mask = side_chain_frames_cdr_mask.unsqueeze(-1) != (
        sidechain_atom_pos_cdr_mask.unsqueeze(-2)
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
        side_chain_frames_cdr_mask, ####
        sidechain_atom_pos,
        renamed_atom14_gt_positions,
        sidechain_atom_pos_cdr_mask, ####
        l1_clamp_distance=clamp_distance,
        l1_clamp_distance_large=intercdr_distance,
        cdr_mask=cdr_mask,
        length_scale=length_scale,
        eps=eps,
    )
    
    return fape + cdr_fape

def compute_prmsd_loss(
    pdev, # prmsd (b, l)
    pred_position, # predicted structure (b, l, 14, 3)
    atom14_gt_positions, # gt stucture  (b, l, 14, 3)
    cdr_mask, # (b, l)
):

    ca_pred = pred_position[:, :, 1] # (b, l, 3)
    ca_target = atom14_gt_positions[:, :, 1]

    bb_dev = (ca_pred - ca_target).norm(dim=-1) # (b, l)
    loss = torch.nn.functional.l1_loss(
        pdev,
        bb_dev,
        reduction='none',
    )

    cdr_loss = torch.sum(
        loss * cdr_mask,
        dim=-1,
    ) / (torch.sum(
        cdr_mask,
        dim=-1,
    )*3) # (b)

    return cdr_loss

def compute_all_atom_clash_loss(
        atom14_pred_positions,
        atom14_atom_exists,
        residue_index,
        residx_atom14_to_atom37):

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
    )

    # between residue clashes = {
    # "mean_loss": mean_loss,  # shape ()
    # "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
    # "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14) }

    return between_residue_clashes['mean_loss']

def local_distance_loss(
        aatype,
        atom14_pred_positions, # (B, L, 14, 3)
        atom14_gt_exists, # (B, L, 14)
        atom14_gt_positions, # (B, L, 14, 3)
        diffuse_mask, # (L)
        ):
    """
    In order to update interface properly, this loss will scan distance among interface atoms.
    """
    # extract neighbor residues 
    pred_pseudo_beta = pseudo_beta_fn(
        aatype,
        atom14_pred_positions,
        None
    )
    cb_distance_map = torch.linalg.norm(
        pred_pseudo_beta[:, :, None, :] - pred_pseudo_beta[:, None, :, :], dim=-1) # (B, N, N)
    
    cdr_residues = torch.nonzero(diffuse_mask, as_tuple=True)[0]
    
    neighbor_mask = (cb_distance_map[:, cdr_residues] < 8)  # (B, N_cdr, L)
    neighbor_indices = torch.unique(torch.nonzero(neighbor_mask)[:, -1])  # (N_nb,)

    # calculate local all-atom log distance map  
    gt_pair_dists = torch.linalg.norm(
        atom14_gt_positions[:, :, None, :, :] - atom14_gt_positions[:, None, :, :, :], dim=-1) # (B, N, N, 14)
    # gt_pair_dists = torch.log10(gt_pair_dists+1)
    local_gt_pair_dists = gt_pair_dists[:, cdr_residues][:, :, neighbor_indices] # (B, N_cdr, N, 14)

    pred_pair_dists = torch.linalg.norm(
        atom14_pred_positions[:, :, None, :, :] - atom14_pred_positions[:, None, :, :, :], dim=-1) # (B, N, N, 14)
    # pred_pair_dists = torch.log10(pred_pair_dists+1)
    local_pred_pair_dists = pred_pair_dists[:, cdr_residues][:, :, neighbor_indices] # (B, N_cdr, N, 14)
    
    # make loss_mask with atom14_gt_exists
    loss_mask = (atom14_gt_exists[:, :, None, :].bool()) & (atom14_gt_exists[:, None, :, :].bool()) # (B, N, N, 14)
    local_loss_mask = loss_mask[:, cdr_residues][:, :, neighbor_indices] # (B, N_cdr, N, 14)

    # calculate loss (batch loss)
    dist_mat_loss = torch.sum(
        (local_gt_pair_dists - local_pred_pair_dists) ** 2 * local_loss_mask,
        dim=(-1,-2,-3)
    ) 
    dist_mat_loss /= (torch.sum(local_loss_mask, dim=(-1,-2,-3)) + 1) # (B)

    return dist_mat_loss

