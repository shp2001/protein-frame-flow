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

Rigid = rigid_utils.Rigid


def create_full_prot(
        atom37: np.ndarray,
        atom37_mask: np.ndarray,
        chain_index=None,
        aatype=None,
        b_factors=None,
    ):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
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
    ):

    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip('.pdb')
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max([
            int(re.findall(r'_(\d+).pdb', x)[0]) for x in existing_files if re.findall(r'_(\d+).pdb', x)
            if re.findall(r'_(\d+).pdb', x)] + [0])
    if not no_indexing:
        save_path = file_path.replace('.pdb', '') + f'_{max_existing_idx+1}.pdb'
    else:
        save_path = file_path
    with open(save_path, 'w') as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(
                    pos37, atom37_mask, chain_index=chain_index, aatype=aatype, b_factors=b_factors)
                pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(
                prot_pos, atom37_mask, chain_index=chain_index, aatype=aatype, b_factors=b_factors)
            pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f'Invalid positions shape {prot_pos.shape}')
        f.write('END')
    return save_path

def get_cdr_and_neighbors(aatype, 
                          atom14_pred_positions,
                          atom14_gt_positions,
                          original_diffuse_mask, 
                          diffuse_mask,
                          gt_pseudo_beta,
                          mode
                          ):
    
    device = atom14_pred_positions.device

    # find anchor residues 
    anchor_residues = find_anchor(original_diffuse_mask, only_h3=False)
    if mode == 'ab' or mode == 'nanobody': # ab dataset -> extract only_h3 
        anchor_residues = anchor_residues[4:6]
    else: # ppi dataset -> use original residues  
        anchor_residues = anchor_residues

    # make pairwise all atom contact map 
    pair_indices = list(combinations_with_replacement(range(14), 2))  # 총 105쌍
    i_idx = torch.tensor([i for i, j in pair_indices], device=device)
    j_idx = torch.tensor([j for i, j in pair_indices], device=device)

    atom_i = atom14_gt_positions[:, :, i_idx]  # (B, L, 105, 3)
    atom_j = atom14_gt_positions[:, :, j_idx]  # (B, L, 105, 3)

    atom_i = atom_i.unsqueeze(2)  # (B, L, 1, 105, 3)
    atom_j = atom_j.unsqueeze(1)  # (B, 1, L, 105, 3)
    gt_aa_distance_map = torch.norm(atom_i - atom_j, dim=-1)  # (B, L, L, 105)




    cb_distance_map = torch.linalg.norm(
        pred_pseudo_beta[:, :, None, :] - pred_pseudo_beta[:, None, :, :], dim=-1) # (B, N, N)
    


    # print(f'anchor_residues: {anchor_residues}')
    # print(f'anchor_residues: {anchor_residues}')
    cdr_residues = [i for i in range(anchor_residues[0]+1, anchor_residues[1]) if diffuse_mask[i]==1]
    cdr_residues = torch.tensor(cdr_residues)

    neighbor_mask_list = []
    for b in range(cb_distance_map.shape[0]):
        neighbor_mask = (cb_distance_map[b, cdr_residues] < 8.0)  # (B, N_cdr, N)
        pred_neighbor_indices = torch.nonzero(neighbor_mask)[:, -1]  # (N_nb,)
        neighbor_mask_list.append(pred_neighbor_indices)
    
    pred_neighbor = torch.cat(neighbor_mask_list, dim=0)

    # extract neighbor residues from gt structure and add to pred neighbors 
    gt_cb_distance_map = torch.linalg.norm(
        gt_pseudo_beta[:, :, None, :] - gt_pseudo_beta[:, None, :, :], dim=-1) # (B, N, N)
    
    neighbor_mask_list = []
    for b in range(cb_distance_map.shape[0]):
        neighbor_mask = (gt_cb_distance_map[b, cdr_residues] < 8.0)  # (B, N_cdr, N)
        gt_neighbor_indices = torch.nonzero(neighbor_mask)[:, -1]  # (N_nb,)
        neighbor_mask_list.append(gt_neighbor_indices)

    gt_neighbor = torch.cat(neighbor_mask_list)

    neighbor_indices = torch.unique(torch.cat((pred_neighbor, gt_neighbor)))

    return cdr_residues, neighbor_indices

def visualize_contact_map(contact_map, cdr_residues, neighbor, title, output_path):
    """
    Visualizes a [L, L, 14] contact map as a 2D heatmap by reducing the 14 atom-pair dimension
    and saves it to a unique file path.
    
    Args:
        contact_map (torch.Tensor): [L, L, 14] tensor (gt_contact_map * local_loss_mask)[0]
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
    print(f"Contact map saved to: {output_path}")

    # 플롯 닫기 (메모리 관리)
    plt.close()
