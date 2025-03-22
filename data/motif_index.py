import sys
import os
import torch 

from os.path import splitext, basename
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from bisect import bisect_left, bisect_right

import random
import json 

def get_pdb_chain_seq(
    pdb_file,
    chain_id,
):
    p = PDBParser()
    file_name = splitext(basename(pdb_file))[0]
    structure = p.get_structure(
        file_name,
        pdb_file,
    )

    pdb_seq = None
    for chain in structure.get_chains():
        if chain.id == chain_id:
            pdb_seq = "".join(
                [seq1(r.get_resname()) for r in chain.get_residues()])

    return pdb_seq

def cdr_indices(chothia_pdb_file, cdr, offset_heavy=True):
    """Gets the index of a given CDR loop"""
    cdr_chothia_range_dict = {
        "h1": (26, 32),
        "h2": (52, 56),
        "h3": (95, 102),
        "l1": (24, 34),
        "l2": (50, 56),
        "l3": (89, 97)
    }

    cdr = str.lower(cdr)
    assert cdr in cdr_chothia_range_dict.keys()

    chothia_range = cdr_chothia_range_dict[cdr]

    parser = PDBParser(QUIET=True)
    pdb_id = basename(chothia_pdb_file).split('.')[0]
    structure = parser.get_structure(pdb_id, chothia_pdb_file)

    chains = list(structure.get_chains())
    if len(chains) < 2:
        print("PDB must have at least two chains.")
        sys.exit(-1)

    # 첫 번째 체인을 heavy chain으로, 두 번째 체인을 light chain으로 설정
    heavy_chain = chains[0]
    light_chain = chains[1]

    # CDR에 따라 올바른 체인을 선택
    chain = heavy_chain if cdr.startswith('h') else light_chain
    chain_id = chain.id

    residue_id_nums = [res.get_id()[1] for res in chain]

    # Binary search to find the start and end of the CDR loop (pdb 상에서 cdr index. anchor residue가 아니라 cdr 양 끝점의 computing index다.)
    cdr_start = bisect_left(
        residue_id_nums,
        chothia_range[0],
    )
    cdr_end = bisect_right(
        residue_id_nums,
        chothia_range[1],
    ) - 1

    if len(get_pdb_chain_seq(chothia_pdb_file, chain_id=chain_id)) != len(residue_id_nums):
        print('ERROR in PDB file ' + chothia_pdb_file)
        print('residue id len', len(residue_id_nums))

    if chain == light_chain and offset_heavy:
        heavy_seq_len = len(get_pdb_chain_seq(chothia_pdb_file, chain_id=heavy_chain.id))
        cdr_start += heavy_seq_len
        cdr_end += heavy_seq_len

    return cdr_start, cdr_end

def get_cdr_range_dict(chothia_pdb_file, heavy_only=False, light_only=False, offset_heavy=True):
    cdr_names = ["h1", "h2", "h3", "l1", "l2", "l3"]
    if heavy_only:
        cdr_names = cdr_names[:3]
    if light_only:
        cdr_names = cdr_names[3:]

    cdr_range_dict = {
        cdr: cdr_indices(chothia_pdb_file, cdr, offset_heavy=offset_heavy)
        for cdr in cdr_names
    }

    return cdr_range_dict

def load_loop_file(loop_file):
    """Gets the index of a given CDR loop"""

    with open(loop_file, 'r') as file:
        contact_idx_dict = json.load(file)

    loop_indices = contact_idx_dict['loop_list']
    masked_chain = contact_idx_dict['masked_chain']
    first_chain_length = contact_idx_dict['chain_A_len']

    loop_indices = random.choice(loop_indices) # [start, end]

    return int(loop_indices[0]), int(loop_indices[1]), masked_chain, first_chain_length

######################## crop_antigen ########################

def group_numbers(numbers, nan_mask):
    if not numbers:
        return []
    
    groups = []
    current_group = [numbers[0]]
    
    for i in range(1, len(numbers)):
        if numbers[i] - numbers[i - 1] < 15:
            current_group.append(numbers[i])
        else:
            groups.append(current_group)
            current_group = [numbers[i]]

    groups.append(current_group)
    
    for i, group in enumerate(groups):
        group_max = max(group)
        group_min = min(group)
        
        groups[i] = []
        for j in range(group_min, group_max+1):
            if nan_mask[j]==1:
                groups[i].append(j)
    return groups

def find_anchor(pattern, only_h3=True):
    anchor = []

    # 문자열의 길이를 확인
    for i in range(1, len(pattern)):
        if pattern[i - 1] == 0 and pattern[i] == 1:
            anchor.append(i-1)  # 0에서 1로 바뀌는 현재 인덱스 추가
        elif pattern[i - 1] == 1 and pattern[i] == 0:
            anchor.append(i)  # 1에서 0으로 바뀌는 현재 인덱스 추가

    if only_h3:
        anchor = anchor[4:6]
    return anchor

# get alpha carbon distance map with translation vector 
def get_distance_map(trans_1):
    residue_loc_1 = trans_1.unsqueeze(1)
    residue_loc_2 = trans_1.unsqueeze(0)

    distance_map = torch.norm(residue_loc_1 - residue_loc_2, dim=2)

    return distance_map

def crop_antigen(trans_1, threshold, cdr_mask, nan_mask, max_len, seq_list=None): 
    chain_len_list = [len(seq) for seq in seq_list]

    ab_len = sum(chain_len_list[:2])
    ag_len = sum(chain_len_list[2:])
    
    residue_indices = None

    # ag이 max_ag_len 미만인 경우 전부 포함
    if sum(chain_len_list) <= max_len:
        ab_idx = [i for i in range(ab_len) if nan_mask[i]==1]
        ag_idx = [i for i in range(ab_len, ab_len+ag_len) if nan_mask[i]==1]
        residue_indices = ab_idx + ag_idx

    # ag이 400 residue 이상인 경우 cropping   
    else:
        distance_map = get_distance_map(trans_1)
        anchor_indices = find_anchor(cdr_mask)
        
        distance_vectors = []
        for i in anchor_indices:
            distance_vectors.append(distance_map[i])

        distance_vectors = torch.stack(distance_vectors)
        distance_vector, _ = torch.min(distance_vectors, dim=0)
        
        distance_ag = distance_vector[ab_len:]
        values, indices = torch.topk(distance_ag, max_len-ab_len, largest=False)
        indices = sorted([i + ab_len for i in indices if nan_mask[i]==1])
        residue_indices = [i for i in range(ab_len) if nan_mask[i]==1] + indices

    return residue_indices

######################## crop_general_protein ########################

def crop_general_protein(trans_1, threshold, loop_mask, nan_mask, max_len, masked_chain, first_chain_len, seq_list=None):  
    first_chain_len = int(first_chain_len)
    chain_len_list = [len(seq) for seq in seq_list]
    L = sum(chain_len_list)

    residue_indices = None
    
    if len([i for i in range(L) if nan_mask[i]==1]) <= max_len:
        residue_indices = [i for i in range(L) if nan_mask[i]==1]

    else:
        distance_map = get_distance_map(trans_1)
        anchor = find_anchor(loop_mask, only_h3=False)
        start = anchor[0] ; end = anchor[1]
        
        distance_vectors = []
        loop_indices = [i for i in range(start+1, end)]
        for i in loop_indices:
            distance_vectors.append(distance_map[i])

        distance_vectors = torch.stack(distance_vectors)
        distance_vector, _ = torch.min(distance_vectors, dim=0)
        values, indices = torch.topk(distance_vector, max_len, largest=False)

        residue_indices = sorted(list((set(indices.tolist() + loop_indices))))

    return residue_indices

######################## relpos ########################

def assign_chain_entity(chain_seq_list):
    index_map = {}
    return [index_map.setdefault(item, len(index_map)) for item in chain_seq_list]

def assign_chain_sym(chain_seq_list):
    count_map = {}  # 원소별 등장 횟수를 기록하는 딕셔너리
    result = []     # 결과 리스트

    for item in chain_seq_list:
        count = count_map.get(item, 0)  # 해당 원소의 현재 등장 횟수 (기본값 0)
        result.append(count)            # 현재 등장 횟수를 결과에 추가
        count_map[item] = count + 1     # 등장 횟수 갱신

    return result


def one_hot(x, v_bins):
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)

    return torch.nn.functional.one_hot(am, num_classes=len(v_bins)).float()


def relpos(
    residue_index,
    asym_id,
    entity_id,
    sym_id,
    max_relative_idx=32,
    max_relative_chain=2
    ):

    residue_index = torch.tensor(residue_index)
    device = residue_index.device
    asym_id = torch.tensor(asym_id)
    entity_id = torch.tensor(entity_id)
    sym_id = torch.tensor(sym_id)

    pos = residue_index
    asym_id_same = (asym_id[..., None] == asym_id[..., None, :])
    offset = pos[..., None] - pos[..., None, :]

    clipped_offset = torch.clamp(
        offset + max_relative_idx, 0, 2 * max_relative_idx
    )

    rel_feats = []

    final_offset = torch.where(
        asym_id_same, 
        clipped_offset,
        (2 * max_relative_idx + 1) * 
        torch.ones_like(clipped_offset)
    )

    boundaries = torch.arange(
        start=0, end=2 * max_relative_idx + 2
    ).to(device)

    rel_pos = one_hot(
        final_offset,
        boundaries,
    )

    rel_feats.append(rel_pos)
    entity_id_same = (entity_id[..., None] == entity_id[..., None, :])
    rel_feats.append(entity_id_same[..., None].to(dtype=rel_pos.dtype))
    rel_sym_id = sym_id[..., None] - sym_id[..., None, :]

    max_rel_chain = max_relative_chain
    clipped_rel_chain = torch.clamp(
        rel_sym_id + max_rel_chain,
        0,
        2 * max_rel_chain,
    )

    final_rel_chain = torch.where(
        entity_id_same,
        clipped_rel_chain,
        (2 * max_rel_chain + 1) *
        torch.ones_like(clipped_rel_chain)
    )

    boundaries = torch.arange(
        start=0, end=2 * max_rel_chain + 2
    ).to(device)
    rel_chain = one_hot(
        final_rel_chain,
        boundaries,
    )

    rel_feats.append(rel_chain)
    rel_feat = torch.cat(rel_feats, dim=-1)
    return rel_feat

def embed_relpos(residue_index, seq_list):
    # AlphaFold-Miultimer
    # Based on OpenFold utils/tensor_utils.py
    chain_entity = assign_chain_entity(seq_list)
    chain_sym = assign_chain_sym(seq_list)
    chain_len_list = [len(seq) for seq in seq_list]

    asym_id = []
    entity_id = []
    sym_id = []
    for i, chain_len in enumerate(chain_len_list):
        for _ in range(chain_len):
            asym_id.append(i)
            entity_id.append(chain_entity[i])
            sym_id.append(chain_sym[i])


    relpos_emb = relpos(    
                    residue_index,
                    asym_id,
                    entity_id,
                    sym_id)
    
    return relpos_emb
