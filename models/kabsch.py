from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn.functional as F

# Backbone bond lengths
BL_N_CA = 1.459
BL_CA_C = 1.525
BL_C_N = 1.336
BL_C_O = 1.229

# Backbone bond angles
BA_N_CA_C = 111.0
BA_N_CA_CB = 110.6
BA_CA_C_N = 117.2
BA_CA_C_O = 120.1
BA_O_C_N = 122.7
BA_C_CA_CB = 110.6
BA_C_N_CA = 121.7

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
        # inplace 대체: 마스크를 활용해 새로운 텐서 생성
        S = torch.where(d.unsqueeze(-1), S * (-1), S)
        V = torch.where(d.unsqueeze(-1).unsqueeze(-1), V * (-1), V)

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
    if align_mask is not None:
        # batch-wise 평균 계산
        mobile_masked_mean = (
            (mobile_ * align_mask.unsqueeze(-1).float()).sum(dim=1, keepdim=True) /
            align_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
        )  # shape: [B, 1, 3]

        stationary_masked_mean = (
            (stationary_ * align_mask.unsqueeze(-1).float()).sum(dim=1, keepdim=True) /
            align_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
        )  # shape: [B, 1, 3]

        # 마스크 안은 그대로, 마스크 밖은 평균값으로 대체
        mobile_ = torch.where(
            align_mask.unsqueeze(-1), 
            mobile_, 
            mobile_masked_mean.expand_as(mobile_)
        )

        stationary_ = torch.where(
            align_mask.unsqueeze(-1), 
            stationary_, 
            stationary_masked_mean.expand_as(stationary_)
        )

        _, kabsch_xform = kabsch(mobile_, stationary_)
    else:
        _, kabsch_xform = kabsch(
            mobile_,
            stationary_,
        )

    return kabsch_xform(mobile)


def kabsch_mse(
    pred, # (b, L*14, 3)
    target, # (b, L*14, 3)
    align_mask=None, # (b, L*14, 3)
    mask=None,
    clamp=0.,
    sqrt=False,
):
    device = pred.device
    aligned_target = do_kabsch(
        mobile=target,
        stationary=pred.detach(),
        align_mask=align_mask,
    )
    mse = F.mse_loss(
        pred,
        aligned_target.to(device),
        reduction='none',
    ).mean(-1)

    mse = mse.to(device)
    if clamp > 0:
        mse = torch.clamp(mse, max=clamp**2)

    if mask != None:
        mask=mask.to(device)
        mse = torch.sum(
            mse * mask,
            dim=-1,
        ) / torch.sum(
            mask,
            dim=-1,
        )
    else:
        mse = mse.mean(-1)

    if sqrt:
        mse = mse.sqrt()

    return mse


def bb_prmsd_l1(
    pdev,
    pred,
    target,
    align_mask=None,
    mask=None,
):
    aligned_target = do_kabsch(
        mobile=target,
        stationary=pred,
        align_mask=align_mask,
    )
    bb_dev = (pred - aligned_target).norm(dim=-1)
    loss = F.l1_loss(
        pdev,
        bb_dev,
        reduction='none',
    )

    if mask != None:
        mask = repeat(mask, "b l -> b (l 4)")
        loss = torch.sum(
            loss * mask,
            dim=-1,
        ) / torch.sum(
            mask,
            dim=-1,
        )
    else:
        loss = loss.mean(-1)

    loss = loss.mean(-1).unsqueeze(0)

    return loss

def dist(x_1, x_2, eps=1e-8):
    d_sq = (x_1 - x_2)**2
    d = torch.sqrt(d_sq.sum(-1) + eps)

    return d

def normed_vec(vec, eps=1e-8):
    mag_sq = torch.sum(vec**2, dim=-1, keepdim=True)
    mag = torch.sqrt(mag_sq + eps)
    vec = vec / mag

    return vec

def normed_cross(vec1, vec2, eps=1e-8):
    vec1 = normed_vec(vec1, eps=eps)
    vec2 = normed_vec(vec2, eps=eps)
    cross = torch.cross(vec1, vec2, dim=-1)

    return cross

def dihedral(x_1, x_2, x_3, x_4, eps=1e-8):
    b1 = normed_vec(x_1 - x_2, eps=eps)
    b2 = normed_vec(x_2 - x_3, eps=eps)
    b3 = normed_vec(x_3 - x_4, eps=eps)
    n1 = normed_cross(b1, b2, eps=eps)
    n2 = normed_cross(b2, b3, eps=eps)
    m1 = normed_cross(n1, b2, eps=eps)
    x = (n1 * n2).sum(-1)
    y = (m1 * n2).sum(-1)

    dih = torch.atan2(y, x)

    return dih

def angle(x_1, x_2, x_3, eps=1e-8):
    a = normed_vec(x_1 - x_2, eps=eps)
    b = normed_vec(x_3 - x_2, eps=eps)
    ang = torch.arccos((a * b).sum(-1))

    return ang

def bond_len_loss(pred, seq_lens, mask, eps=1e-8):
    b, l, a, d = pred.shape

    pred_bb = pred[:, :, :3]
    mask = repeat(mask, "b l -> b (l 3)")
    for seq_len in seq_lens:
        mask[:, 3 * seq_len - 1] = 0
    mask_bb = mask[:, :-1] * mask[:, 1:]

    pred_bond_lens = dist(
        rearrange(pred_bb, "b l a d -> b (l a) d")[:, :-1],
        rearrange(pred_bb, "b l a d -> b (l a) d")[:, 1:],
    )
    lit_bond_lens = repeat(
        torch.tensor([BL_N_CA, BL_CA_C, BL_C_N]),
        "bl -> b (l bl)",
        b=b,
        l=l,
    )[:, :-1]
    lit_bond_lens = lit_bond_lens.to(pred_bond_lens.device)

    bl_loss = torch.abs(pred_bond_lens - lit_bond_lens) * mask_bb
    bl_loss = bl_loss.sum(-1) / (mask.sum(-1) + eps)

    return bl_loss


def bond_angle_loss(pred, seq_lens, mask, eps=1e-8):
    b, l, a, d = pred.shape

    for seq_len in seq_lens:
        mask[:, seq_len - 1] = 0
    mask_ = mask[:, 1:] * mask[:, :-1]

    N, CA, C, CB = pred.unbind(-2)
    ba_CA_C_N = angle(CA[:, :-1], C[:, :-1], N[:, 1:], eps=eps)
    ba_CA_C_N_loss = 1 - torch.cos(ba_CA_C_N - BA_CA_C_N * np.pi / 180)
    ba_CA_C_N_loss = ba_CA_C_N_loss * mask_

    ba_C_N_CA = angle(C[:, :-1], N[:, 1:], CA[:, 1:], eps=eps)
    ba_C_N_CA_loss = 1 - torch.cos(ba_C_N_CA - BA_C_N_CA * np.pi / 180)
    ba_C_N_CA_loss = ba_C_N_CA_loss * mask_

    loss = ba_CA_C_N_loss + ba_C_N_CA_loss
    loss = loss.sum(-1) / (mask_.sum(-1) + eps)

    return loss

def cis_peptide_loss(pred, seq_lens, mask, eps=1e-8):
    for seq_len in seq_lens:
        mask[:, seq_len - 1] = 0
    mask_ = mask[:, 1:] * mask[:, :-1]

    N, CA, C, _ = pred.unbind(-2)
    dih = dihedral(CA[:, :-1], C[:, :-1], N[:, 1:], CA[:, 1:], eps=0)

    loss = 1 - torch.cos(dih - np.pi)
    loss = loss.sum(dim=(-1, -2)) / (mask_.sum(dim=(-1, -2)) + eps)

    return loss


def violation_bond_loss(pred, seq_lens, mask, eps=1e-8):
    bl_loss = bond_len_loss(pred, seq_lens, mask, eps=eps)
    ba_loss = bond_angle_loss(pred, seq_lens, mask, eps=eps)
    cis_loss = cis_peptide_loss(pred, seq_lens, mask, eps=eps)

    loss = bl_loss + ba_loss + cis_loss

    return loss

