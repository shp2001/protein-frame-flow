import torch 
from data import residue_constants
from openfold.utils.loss import between_residue_clash_loss
from openfold.data.data_transforms import pseudo_beta_fn

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

