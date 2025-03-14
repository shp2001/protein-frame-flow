import torch 
from data import residue_constants

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
    pred_angles = angles_sin_cos[..., 3:, :]
    residue_type_one_hot = torch.nn.functional.one_hot(
        aatype,
        residue_constants.restype_num + 1,
    )
    chi_pi_periodic = torch.einsum(
        "...ij,jk->ik",
        residue_type_one_hot.type(angles_sin_cos.dtype),
        angles_sin_cos.new_tensor(residue_constants.chi_pi_periodic),
    )

    true_chi = chi_angles_sin_cos[None]

    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum((true_chi_shifted - pred_angles) ** 2, dim=-1)
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)

    # The ol' switcheroo
    sq_chi_error = sq_chi_error.permute(
        *range(len(sq_chi_error.shape))[1:-2], 0, -2, -1
    )

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
    cdr_clamp=30, #
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