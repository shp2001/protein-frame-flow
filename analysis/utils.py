import numpy as np
import os
import re
from data import protein
from openfold.utils import rigid_utils
import matplotlib.pyplot as plt
import seaborn as sns

import torch 
from openfold.data.data_transforms import pseudo_beta_fn
from data.motif_index import find_anchor

from itertools import combinations_with_replacement

Rigid = rigid_utils.Rigid


def create_full_prot(
        atom37: np.ndarray,
        atom37_mask: np.ndarray,
        chain_index=None,
        aatype=None,
        b_factors=None,
        residue_index=None
    ):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    if residue_index is None:
        residue_index = np.arange(n)
    if chain_index is None:
        chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors)


def write_prot_to_pdb(
        prot_pos: np.ndarray,
        file_path: str,
        aatype: np.ndarray=None,
        chain_index=None,
        overwrite=False,
        no_indexing=False,
        b_factors=None,
        residue_index=None,  # 추가
    ):
    save_path = file_path

    with open(save_path, 'w') as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(
                    pos37, atom37_mask, chain_index=chain_index, aatype=aatype, b_factors=b_factors,
                    residue_index=residue_index  # 전달
                )
                pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(
                prot_pos, atom37_mask, chain_index=chain_index, aatype=aatype, b_factors=b_factors,
                residue_index=residue_index  # 전달
            )
            pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f'Invalid positions shape {prot_pos.shape}')
        f.write('END')
    return save_path

def get_cdr_and_neighbors(
        atom14_gt_positions, 
        atom14_gt_exists,
        loop_mask, 
        mode,
        distance_threshold=5
        ):
    
    device = atom14_gt_positions.device
    # find anchor residues 

    anchor_residues = find_anchor(loop_mask, only_h3=False)
    full_anchor_residues = anchor_residues

    if mode == 'ab' or mode == 'nanobody': # ab dataset -> extract only_h3 
        anchor_residues = anchor_residues[4:6]
    else: # ppi dataset -> use original residues  
        anchor_residues = anchor_residues

    cdr_residues = [i for i in range(anchor_residues[0]+1, anchor_residues[1]) if loop_mask[i]==1]
    cdr_residues = torch.tensor(cdr_residues, device=device)

    # make pairwise all atom contact map (gt)
    pair_indices = list(combinations_with_replacement(range(14), 2))  # 총 105쌍
    i_idx = torch.tensor([i for i, j in pair_indices], device=device)
    j_idx = torch.tensor([j for i, j in pair_indices], device=device)

    atom_i = atom14_gt_positions[0, :, i_idx]  # (L, 105, 3)
    atom_j = atom14_gt_positions[0, :, j_idx]  # (L, 105, 3)

    atom_i = atom_i.unsqueeze(1)  # (L, 1, 105, 3)
    atom_j = atom_j.unsqueeze(0)  # (1, L, 105, 3)
    gt_aa_distance_map = torch.norm(atom_i - atom_j, dim=-1)  # (L, L, 105)

    # make pairwise all atom contact map mask 
    exists_i = atom14_gt_exists[0, :, i_idx]  # (L, 105)
    exists_j = atom14_gt_exists[0, :, j_idx]  # (L, 105)

    mask_i = exists_i.unsqueeze(1)  # (L, 1, 105)
    mask_j = exists_j.unsqueeze(0)  # (1, L, 105)

    edge_mask = mask_i * mask_j  # (L, L, 105)

    # find gt neighbors
    gt_aa_distance_map = torch.where(edge_mask == 0, torch.tensor(1000, device=device), gt_aa_distance_map) # (L, L, 105)
    gt_neighbor_mask = torch.any((gt_aa_distance_map[cdr_residues] < distance_threshold), dim=-1)  # (N_cdr, N)
    gt_neighbor = torch.nonzero(gt_neighbor_mask)[:, -1]  # (N_nb,)
    neighbor_indices = torch.unique(gt_neighbor)
        
    return cdr_residues, neighbor_indices, full_anchor_residues

def visualize_contact_map(contact_map, cdr_residues, neighbor, title, output_path):
    """
    Visualizes a [L, L, 14] contact map as a 2D heatmap by reducing the 14 atom-pair dimension
    and saves it to a unique file path.
    
    Args:
        contact_map (torch.Tensor): [L, L] tensor (gt_contact_map * local_loss_mask)[0]
        title (str): Title of the plot
        output_path (str): Path to save the output image (e.g., 'contact_map.png')
    """

    contact_map_interface = contact_map[cdr_residues][:, neighbor]

    # 히트맵 그리기
    plt.figure(figsize=(6, 3))
    ax = sns.heatmap(
        contact_map_interface.detach().cpu().numpy(), 
        cmap="Reds", 
        cbar=True, 
        cbar_kws={"shrink": 0.4},
        xticklabels=neighbor.cpu().numpy().tolist(), 
        yticklabels=cdr_residues.cpu().numpy().tolist(),
        square=True
    )
    plt.title(title)
    plt.xlabel("Neighbors Index", fontsize=6)
    plt.ylabel("H3 CDR Index", fontsize=6)
    plt.xticks(rotation=90, fontsize=4)
    plt.yticks(rotation=0, fontsize=4) 
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 이미지 저장
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    # 플롯 닫기 (메모리 관리)
    plt.close()
