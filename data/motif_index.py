import sys
import torch 

from os.path import splitext, basename
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from bisect import bisect_left, bisect_right
import data.utils as du
import random
import numpy as np 
import json 
from scipy.spatial.distance import cdist
from collections import defaultdict

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


def convert_mask_info_index(mask_info, residue_index, chain_index):
    """
    mask_info의 pdb residue id를 전체 시퀀스 기반의 python index로 변환합니다.
    """
    
    if isinstance(residue_index, torch.Tensor):
        residue_index = residue_index.tolist()
    if isinstance(chain_index, torch.Tensor):
        chain_index = chain_index.tolist()

    # 1. Lookup Table 생성
    # 구조: {chain_int: {residue_id: python_index, ...}, ...}
    lookup = defaultdict(dict)
    
    for i, (c, r) in enumerate(zip(chain_index, residue_index)):
        lookup[c][r] = i
        
    converted_mask_info = {}

    for chain_str, blocks in mask_info.items():
        chain_int = du.CHAIN_TO_INT.get(chain_str)
            
        chain_map = lookup[chain_int]
        
        converted_blocks = []
        for block in blocks:
            converted_block = [chain_map[res_id] for res_id in block if res_id in chain_map]
            if converted_block:
                converted_blocks.append(converted_block)
        
        converted_mask_info[chain_str] = converted_blocks
        
    return converted_mask_info


def load_general_mask(
        mask_info_file,
        residue_index,
        chain_index, 
        pseudo_beta,
        threshold_min_dist,
        threshold_max_dist,
        threshold_length,
        seed=None
    ):

    # --- [헬퍼 함수 1] 길이 제한 및 랜덤 슬라이싱 ---
    def get_random_contiguous_segment(indices, limit):
        if len(indices) > limit:
            max_start_idx = len(indices) - limit
            start_idx = random.randint(0, max_start_idx)
            return indices[start_idx : start_idx + limit]
        return indices

    # --- [헬퍼 함수 2] 인덱스 리스트를 연속된 구간들로 분리 ---
    def split_into_contiguous_segments(indices):
        """
        [0, 1, 2, 5, 6] -> [[0, 1, 2], [5, 6]]
        """
        if len(indices) == 0:
            return []
        segments = []
        current_segment = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_segment.append(indices[i])
            else:
                segments.append(current_segment)
                current_segment = [indices[i]]
        segments.append(current_segment)
        return segments
    # ------------------

    # 1. JSON 로드
    with open(mask_info_file, 'r') as file:
        mask_info = json.load(file)

    # 2. 인덱스 변환
    mask_info = convert_mask_info_index(
        mask_info,
        residue_index,
        chain_index
    )
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 예외 처리
    chain_keys = list(mask_info.keys())
    if not chain_keys:
        return torch.zeros(len(residue_index))

    # 3. Pivot 및 주변 블록 선택 로직
    selected_chain = random.choice(chain_keys) # str 형태의 chain id 
    blocks_in_chain = mask_info[selected_chain]

    raw_pivot_block = random.choice(blocks_in_chain)
    pivot_block = get_random_contiguous_segment(raw_pivot_block, limit=threshold_length)
    
    loop_index_set = set(pivot_block)
    
    # 거리 계산을 위해 pivot 좌표 추출
    pivot_coords = pseudo_beta[pivot_block]    

    for c_key, blocks in mask_info.items():
        for block in blocks:
            target_coords = pseudo_beta[block]
            
            # Tensor/Numpy 호환성 처리
            if isinstance(pivot_coords, torch.Tensor):
                p_coords = pivot_coords.cpu().numpy()
                t_coords = target_coords.cpu().numpy()
            else:
                p_coords = pivot_coords
                t_coords = target_coords

            # 거리 행렬 계산 (Shape: [N_pivot, N_target])
            dists = cdist(p_coords, t_coords)

            # Case 1: 블록 전체가 조건 만족 (유지)
            if (np.min(dists) < threshold_min_dist) and (np.max(dists) < threshold_max_dist):
                selected_residues = get_random_contiguous_segment(block, limit=threshold_length)
                loop_index_set.update(selected_residues)
            
            # Case 2: 블록 내 일부 잔기만 조건 만족 -> 부분 선택
            elif np.max(dists) >= threshold_max_dist:
                
                max_dist_per_residue = np.max(dists, axis=0) # Shape: (N_target,)
                valid_relative_indices = np.where(max_dist_per_residue < threshold_max_dist)[0]

                segments = split_into_contiguous_segments(valid_relative_indices)

                for seg_indices in segments:
                    # [추가된 조건] 세그먼트 길이가 5 미만이면 건너뜀 (제거)
                    if len(seg_indices) < 5:
                        continue

                    segment_dists = dists[:, seg_indices]
                    
                    if np.min(segment_dists) < threshold_min_dist:
                        valid_residues = [block[i] for i in seg_indices]
                        final_residues = get_random_contiguous_segment(valid_residues, limit=threshold_length)
                        loop_index_set.update(final_residues)

    final_loop_index = sorted(list(loop_index_set))
    L = len(residue_index)
    loop_mask = torch.zeros(L) 
    
    if final_loop_index:
        loop_mask[final_loop_index] = 1.0
    
    return loop_mask, du.CHAIN_TO_INT.get(selected_chain)

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

def provide_anchor(diffuse_mask, res_mask, chain_index, mode):
    if mode == 'monomer':
        indices = torch.nonzero(diffuse_mask, as_tuple=False).squeeze()
        if indices.numel() > 0:
            start, end = indices[0].item(), indices[-1].item()

            # 시작 이전에 anchor(res_mask==1)가 없으면 앞을 마스킹
            if res_mask[:start].sum() == 0:
                for i in range(start, end + 1):
                    diffuse_mask[i] = 0
                    if res_mask[i] == 1:
                        break

            # 끝 이후에 anchor(res_mask==1)가 없으면 뒤를 마스킹
            if res_mask[end + 1:].sum() == 0:
                for i in range(end, start - 1, -1):
                    diffuse_mask[i] = 0
                    if res_mask[i] == 1:
                        break

    elif mode == 'polymer':
        unique_chains = torch.unique(chain_index)

        for chain_id in unique_chains:
            chain_mask = (chain_index == chain_id)
            chain_indices = torch.nonzero(chain_mask, as_tuple=False).squeeze()

            if chain_indices.numel() == 0:
                continue

            chain_diffuse = diffuse_mask[chain_mask]
            chain_res_mask = res_mask[chain_mask]

            if chain_diffuse.sum() == 0:
                continue

            start = torch.nonzero(chain_diffuse, as_tuple=False)[0].item()
            end = torch.nonzero(chain_diffuse, as_tuple=False)[-1].item()

            # 앞쪽 anchor가 없으면 앞을 마스킹
            if chain_res_mask[:start].sum() == 0:
                for i in range(start, end + 1):
                    diffuse_mask[chain_indices[i]] = 0
                    if chain_res_mask[i] == 1:
                        break

            # 뒤쪽 anchor가 없으면 뒤를 마스킹
            if chain_res_mask[end + 1:].sum() == 0:
                for i in range(end, start - 1, -1):
                    diffuse_mask[chain_indices[i]] = 0
                    if chain_res_mask[i] == 1:
                        break
    else:
        return diffuse_mask

    return diffuse_mask

# get alpha carbon distance map with translation vector 
def get_distance_map(trans_1):
    residue_loc_1 = trans_1.unsqueeze(1)
    residue_loc_2 = trans_1.unsqueeze(0)

    distance_map = torch.norm(residue_loc_1 - residue_loc_2, dim=2)

    return distance_map

def crop_antigen(trans_1, cdr_mask, nan_mask, max_len, seq_list, crop_ab, include_ag=True, mode='ab'): 
    chain_len_list = [len(seq) for seq in seq_list]

    if mode == 'ab':
        ab_len = sum(chain_len_list[:2])
        ag_len = sum(chain_len_list[2:])
    
    else:
        ab_len = sum(chain_len_list[:1])
        ag_len = sum(chain_len_list[1:])

    residue_indices = None
    all_anchors = find_anchor(cdr_mask, only_h3=False)

    ab_idx = []
    
    if crop_ab:
        for i in range(len(all_anchors)//2):
            residues = [i for i in range(all_anchors[2*i]-6, all_anchors[2*i+1]+6) if nan_mask[i]==1]
            ab_idx.extend(residues)
    else:
        ab_idx = [i for i in range(ab_len) if nan_mask[i]==1]

    # ag이 max_ag_len 미만인 경우 전부 포함
    if include_ag:
        if len(ab_idx) + ag_len <= max_len:
            ag_idx = [i for i in range(ab_len, ab_len+ag_len) if nan_mask[i]==1]
            residue_indices = ab_idx + ag_idx

        # ag이 max_ag_len 이상인 경우 cropping   
        else:
            distance_map = get_distance_map(trans_1)
            loop_residues = torch.nonzero(cdr_mask, as_tuple=False).squeeze(-1)
            
            distance_vectors = []
            for i in loop_residues:
                distance_vectors.append(distance_map[i])

            distance_vectors = torch.stack(distance_vectors)
            distance_vector, _ = torch.min(distance_vectors, dim=0)
            
            distance_ag = distance_vector[ab_len:]
            values, indices = torch.topk(distance_ag, max_len-len(ab_idx), largest=False)
            indices = sorted([i + ab_len for i in indices if nan_mask[i]==1])
            residue_indices = ab_idx + indices
    else:
        residue_indices = ab_idx
    return residue_indices

######################## crop_general_protein ########################
def crop_general_protein(trans_1, loop_mask, nan_mask, max_len, seq_list):  
    chain_len_list = [len(seq) for seq in seq_list]
    L = sum(chain_len_list)
    residue_indices = None
    num_loop_residues = torch.sum(loop_mask).item()

    if num_loop_residues <= 10:
        ratio = 0.5
    elif num_loop_residues >= 60:
        ratio = 1.0
    else:
        ratio = 0.5 + (num_loop_residues - 10) * (0.5 / 50)

    # 3. 새로운 max_len 결정 (정수형 변환)
    max_len = int(max_len * ratio)        
    if torch.sum(nan_mask) <= max_len:
        residue_indices = [i for i in range(L) if nan_mask[i]==1]

    else:
        distance_map = get_distance_map(trans_1)
        distance_vectors = []
        
        loop_indices = (
            (loop_mask == 1) & (nan_mask == 1)
        ).nonzero(as_tuple=True)[0].tolist()

        for i in loop_indices:
            distance_vectors.append(distance_map[i])

        distance_vectors = torch.stack(distance_vectors)
        distance_vector, _ = torch.min(distance_vectors, dim=0)
        values, indices = torch.topk(distance_vector, max_len, largest=False)

        residue_indices = sorted(list((set(indices.tolist() + loop_indices))))
        residue_indices = [i for i in residue_indices if nan_mask[i] == 1]

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

def get_relpos_input(seq_list):
    chain_entity = assign_chain_entity(seq_list)
    chain_sym = assign_chain_sym(seq_list)
    chain_len_list = [len(seq) for seq in seq_list]

    asym_id = [] # 서로 다른 체인이면 무조건 다른 id  
    entity_id = [] # 동일한 시퀀스의 체인이면 동일한 id 
    sym_id = [] # 동일한 시퀀스의 체인 상에서 몇 번째 체인인지. 즉 같은 시퀀스의 체인들을 구분 
    for i, chain_len in enumerate(chain_len_list):
        for _ in range(chain_len):
            asym_id.append(i)
            entity_id.append(chain_entity[i])
            sym_id.append(chain_sym[i])
    
    return asym_id, entity_id, sym_id

def get_ag_hotspot(
        pseudo_beta, 
        loop_mask, 
        diffuse_mask,
        threshold,            # True Hotspot 기준 거리
        masking_ratio,        # True Hotspot을 지울(drop) 확률
        false_hotspot_ratio,  # Candidate 영역에서 False Hotspot을 생성할 확률
        noise_range        # False Hotspot 후보군 거리 범위 (threshold ~ threshold + noise_range)
    ):
    """
    pseudo_beta:   (B, L, 3) 좌표
    loop_mask:     (B, L) 1이면 loop residue
    diffuse_mask:  (B, L) 0이면 antigen residue
    threshold:     contact 거리 cutoff
    """
    B, L, _ = pseudo_beta.shape
    device = pseudo_beta.device

    loop_mask_bool = loop_mask[0].bool()           # (num_loop,)
    ag_mask_bool   = (diffuse_mask[0] == 0).bool() # (num_ag,)
    dist_map = torch.cdist(pseudo_beta, pseudo_beta)
    d = dist_map[:, loop_mask_bool][:, :, ag_mask_bool]
    min_dist_to_loop = d.min(dim=1).values 

    is_true_hotspot = min_dist_to_loop < threshold
    drop_prob = torch.rand_like(min_dist_to_loop)
    keep_mask = drop_prob > masking_ratio 
    processed_true_hotspot = is_true_hotspot & keep_mask


    upper_threshold = threshold + noise_range
    is_candidate_region = (min_dist_to_loop >= threshold) & (min_dist_to_loop < upper_threshold)
    
    # Add Mask 생성: false_hotspot_ratio 확률로 True(1)
    # 독립적인 난수 생성
    add_prob = torch.rand_like(min_dist_to_loop)
    select_mask = add_prob < false_hotspot_ratio
    processed_false_hotspot = is_candidate_region & select_mask

    final_contact = processed_true_hotspot | processed_false_hotspot

    # 4. 전체 시퀀스 크기로 복원
    hotspot = torch.zeros((B, L), dtype=torch.float, device=device)
    ag_idx = torch.where(ag_mask_bool)[0]
    hotspot[:, ag_idx] = final_contact.float()

    return hotspot