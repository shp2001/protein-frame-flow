import sys
import torch 

from os.path import splitext, basename
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from bisect import bisect_left, bisect_right

import random
import numpy as np 
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

<<<<<<< HEAD
def load_antibody_mask(
    scaffold_idx,
    chain_index,
    residue_index,
    ab_chains,
    pseudo_beta
):
    """
    Antibody CDR 영역(start ~ end 포함)을 1로 설정하고, 
    CDR과 가까운 Antigen Interface 영역을 0으로 마스킹합니다.
    (말단 트리밍 및 길이 필터링 로직 포함)
    """
    
    # 0. Device 및 기초 설정
    device = pseudo_beta.device
    L = chain_index.shape[0]
    scaffold_mask = torch.zeros(L, dtype=torch.float32, device=device)
    
    # ab_chains를 Tensor로 변환
    ab_chains_tensor = torch.tensor(ab_chains, device=device)

    # ---------------------------------------------------------
    # 1. CDR 영역 설정 (Value = 1)
    # ---------------------------------------------------------
    cdr_loops = ['h1', 'h2', 'h3', 'l1', 'l2', 'l3']
    all_cdr_indices_list = []
    
    for loop in cdr_loops:
        start_key = f"{loop}_start"
        end_key = f"{loop}_end"
        
        if start_key in scaffold_idx and end_key in scaffold_idx:
            start = scaffold_idx[start_key]
            end = scaffold_idx[end_key]
            loop_indices = torch.arange(start, end + 1, device=device)
            loop_indices = loop_indices[loop_indices < L]
            scaffold_mask[loop_indices] = 1.0
            all_cdr_indices_list.append(loop_indices)
    
    all_cdr_indices = torch.cat(all_cdr_indices_list)

    is_ab = torch.isin(chain_index, ab_chains_tensor)
    ag_indices = torch.nonzero(~is_ab, as_tuple=True)[0] 
    
    if ag_indices.numel() == 0:
        return scaffold_mask

    cdr_coords = pseudo_beta[all_cdr_indices]
    ag_coords = pseudo_beta[ag_indices]       
    
    dists = torch.cdist(cdr_coords, ag_coords) 
    min_dists = torch.min(dists, dim=0).values 
    
    is_contact = min_dists < 8.0
    raw_interface_indices = ag_indices[is_contact]
    
    if raw_interface_indices.numel() == 0:
        return scaffold_mask

    # ---------------------------------------------------------
    # 3. Gap Filling & Trimming & Filtering
    # ---------------------------------------------------------
    # 3-1. 초기 Interface Mask 생성 (Gap 채우기 전)
    final_interface_mask = torch.zeros(L, dtype=torch.bool, device=device)
    final_interface_mask[raw_interface_indices] = True
    
    unique_ag_chains = torch.unique(chain_index[raw_interface_indices])
    
    for c_id in unique_ag_chains:
        chain_mask = (chain_index == c_id)
        
        # --- [Step A] Gap Filling ---
        c_interface_bool = final_interface_mask & chain_mask
        if not c_interface_bool.any():
            continue
            
        c_interface_indices = torch.nonzero(c_interface_bool, as_tuple=True)[0]
        c_pdb_ids = residue_index[c_interface_indices]
        sorted_pdbs, sort_idx = torch.sort(c_pdb_ids)
        
        if sorted_pdbs.numel() > 1:
            diffs = sorted_pdbs[1:] - sorted_pdbs[:-1]
            gap_mask = (diffs > 1) & (diffs < 5)
            
            if gap_mask.any():
                gap_starts = sorted_pdbs[:-1][gap_mask]
                gap_ends = sorted_pdbs[1:][gap_mask]
                
                chain_global_indices = torch.nonzero(chain_mask, as_tuple=True)[0]
                chain_global_pdbs = residue_index[chain_global_indices]
                
                g_s = gap_starts.unsqueeze(1)
                g_e = gap_ends.unsqueeze(1)
                c_p = chain_global_pdbs.unsqueeze(0)
                
                in_gap_mask = (c_p > g_s) & (c_p < g_e)
                fill_candidates_mask = in_gap_mask.any(dim=0)
                
                fill_indices = chain_global_indices[fill_candidates_mask]
                final_interface_mask[fill_indices] = True

        # --- [Step B] Trimming & Filtering ---
        # Gap Filling이 완료된 후, 해당 체인의 마스크를 다시 가져와서 블록 단위로 처리
        c_interface_bool_updated = final_interface_mask & chain_mask
        if not c_interface_bool_updated.any():
            continue
        
        # 현재 체인의 확정된 Interface 인덱스들
        current_indices = torch.nonzero(c_interface_bool_updated, as_tuple=True)[0]
        current_pdbs = residue_index[current_indices]
        
        # PDB ID 기준 정렬 (블록을 나누기 위해)
        sorted_pdbs, sort_idx = torch.sort(current_pdbs)
        sorted_indices = current_indices[sort_idx]
        
        # 체인의 전체 범위 확인 (N-term, C-term 판별용)
        chain_global_indices = torch.nonzero(chain_mask, as_tuple=True)[0]
        chain_global_pdbs = residue_index[chain_global_indices]
        chain_start_pdb = torch.min(chain_global_pdbs)
        chain_end_pdb = torch.max(chain_global_pdbs)
        
        # 블록 나누기 (PDB ID가 연속되지 않으면 분리)
        # diff != 1 인 지점이 블록의 경계
        if sorted_pdbs.numel() > 0:
            pdb_diffs = sorted_pdbs[1:] - sorted_pdbs[:-1]
            break_points = torch.nonzero(pdb_diffs != 1, as_tuple=True)[0] + 1
            
            # [0, break1, break2, ..., len] 형태의 split 포인트 생성
            splits = torch.cat([
                torch.tensor([0], device=device), 
                break_points, 
                torch.tensor([sorted_pdbs.numel()], device=device)
            ])
            
            valid_indices_list = []
            
            for i in range(len(splits) - 1):
                start_idx = splits[i]
                end_idx = splits[i+1]
                
                # 하나의 블록
                block_indices = sorted_indices[start_idx:end_idx]
                block_pdbs = sorted_pdbs[start_idx:end_idx]
                
                # 1. Trimming (N-term)
                # 블록의 첫 잔기가 체인의 시작 잔기라면 앞 4개 제거
                if block_pdbs[0] == chain_start_pdb:
                    block_indices = block_indices[4:]
                    # block_pdbs = block_pdbs[4:] # (인덱스만 슬라이싱하면 충분)
                
                if block_indices.numel() == 0: continue

                # 2. Trimming (C-term)
                # 블록의 마지막 잔기가 체인의 끝 잔기라면 뒤 4개 제거
                # (주의: N-term trimming으로 인해 block_pdbs가 바뀌었을 수 있으므로
                # 원본 sorted_pdbs의 해당 구간 마지막 값을 참조)
                if sorted_pdbs[end_idx-1] == chain_end_pdb:
                    block_indices = block_indices[:-4]
                
                # 3. Filtering (Length < 4)
                if block_indices.numel() >= 4:
                    valid_indices_list.append(block_indices)
            
            # 기존 마스크 초기화 후 검증된 인덱스만 다시 활성화
            # (해당 체인 영역만 False로 밀고 다시 씀)
            final_interface_mask[current_indices] = False 
            
            if valid_indices_list:
                all_valid_indices = torch.cat(valid_indices_list)
                final_interface_mask[all_valid_indices] = True
            else:
                pass

    scaffold_mask[final_interface_mask] = 1.0
    return scaffold_mask
=======
def load_loop_file(loop_file, seed=None):
    """Gets the index of a given CDR loop"""
>>>>>>> parent of 4266575 (generate loop mask incorporating recetpor interface)

    with open(loop_file, 'r') as file:
        contact_idx_dict = json.load(file)

    loop_indices = contact_idx_dict['loop_list']
    masked_chain = contact_idx_dict['masked_chain']
    first_chain_length = contact_idx_dict['chain_A_len']
    
    if seed != None:
        random.seed(seed)
    loop_index = random.choice(loop_indices) # [start, end]

    return int(loop_index[0]), int(loop_index[1]), masked_chain, first_chain_length

def load_monomer_mask(mask_info_file, seq_len, seed=None):
    with open(mask_info_file, 'r') as file:
<<<<<<< HEAD
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

    chain_keys = list(mask_info.keys())
    if not chain_keys:
        return torch.zeros(len(residue_index))

    # 3. Pivot 및 주변 블록 선택 로직
    selected_chain = random.choice(chain_keys) # str 형태의 chain id 
    blocks_in_chain = mask_info[selected_chain]

    raw_pivot_block = random.choice(blocks_in_chain)
    pivot_block = get_random_contiguous_segment(raw_pivot_block, limit=threshold_length)

    loop_index_set = set(pivot_block)
=======
        loop_indices = json.load(file)
    
    if seed is not None:
        random.seed(seed)
    
    loop_index = random.choice(loop_indices)

    # 시작과 끝을 clipping
    start = max(0, int(loop_index[0]))
    end = min(seq_len - 1, int(loop_index[-1]))

    return start, end

def load_polymer_mask(mask_info_file, select_num=1, seed=None):
    if seed is not None:
        random.seed(seed)

    with open(mask_info_file, 'r') as file:
        interface_indices = json.load(file)

    chain_lengths = interface_indices["chain_lengths"]
    chain_ids = list(chain_lengths.keys())  # 등장 순서 보장

    # interface_residues에 존재하는 체인 인덱스들만 추출
    interface_residues = interface_indices["interface_residues"]
    interface_chain_indices = [
        idx for idx, residues in interface_residues.items()
        if residues  # 빈 리스트가 아닌 경우
    ]

    interface_chains = [
        chain_ids[int(idx)] for idx in interface_chain_indices
    ]

    long_chains = [c for c in interface_chains if chain_lengths[c] >= 50]

    if long_chains:
        selected_chain = random.choice(long_chains)
    else:
        selected_chain = max(interface_chains, key=lambda c: chain_lengths[c])

    # 체인 오프셋 계산
    chain_offsets = {}
    offset = 0
    for cid in chain_ids:
        chain_offsets[cid] = offset
        offset += chain_lengths[cid]

    start_index = chain_offsets[selected_chain]
    end_index = start_index + chain_lengths[selected_chain] - 1

    # interface residue 리스트 가져오기
    chain_index = str(chain_ids.index(selected_chain))  # 등장 순서 기반 인덱싱
    interface_list = interface_indices["interface_residues"].get(chain_index, [])

    if not interface_list:
        raise ValueError(f"{mask_info_file}: No interface residues found for chain {selected_chain}")

    if len(interface_list) < select_num:
        select_num = len(interface_list)

    weights = np.array([len(g) for g in interface_list], dtype=float)
    weights /= weights.sum()  # 확률로 정규화

    idx = np.random.choice(
        len(interface_list),
        size=select_num,
        replace=False,
        p=weights
    )

    interfaces = [interface_list[i] for i in idx]
>>>>>>> parent of 4266575 (generate loop mask incorporating recetpor interface)
    
    # 5. clipping: 선택된 체인의 범위 안에서만 확장

    selected_residues_list = []
    for interface in interfaces:
        min_res = max(start_index, interface[0] - 3)
        max_res = min(end_index, interface[-1] + 3)
        selected_residues = [res for res in interface if min_res <= res <= max_res]

        if len(selected_residues) > 30:
            start = random.randint(0, len(selected_residues) - 30)
            selected_residues = selected_residues[start:start + 30]
        
        selected_residues_list.append(selected_residues)

<<<<<<< HEAD
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
                    # [추가된 조건] 세그먼트 길이가 4 미만이면 건너뜀 (제거)
                    if len(seg_indices) < 4:
                        continue

                    segment_dists = dists[:, seg_indices]
                    
                    if np.min(segment_dists) < threshold_min_dist:
                        valid_residues = [block[i] for i in seg_indices]
                        final_residues = get_random_contiguous_segment(valid_residues, limit=threshold_length)
                        loop_index_set.update(final_residues)

    final_loop_index = sorted(list(loop_index_set))
    L = len(residue_index)
    loop_mask = torch.zeros(L) 
=======
    return [(selected_residues[0], selected_residues[-1]) for selected_residues in selected_residues_list] 
>>>>>>> parent of 4266575 (generate loop mask incorporating recetpor interface)
    
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
<<<<<<< HEAD
def crop_general_protein(
        trans_1, 
        loop_mask,
        nan_mask, 
        max_len, 
        seq_list
        ):  
=======

def crop_general_protein(trans_1, loop_mask, nan_mask, max_len, seq_list):  
>>>>>>> parent of 4266575 (generate loop mask incorporating recetpor interface)
    chain_len_list = [len(seq) for seq in seq_list]
    L = sum(chain_len_list)
    residue_indices = None


    loop_indices = (
        (loop_mask == 1) & (nan_mask == 1)
    ).nonzero(as_tuple=True)[0].tolist()

    if torch.sum(loop_mask) < 10:
        max_len = max_len // 2
    elif torch.sum(loop_mask) < 15:
        max_len = (max_len * 3) // 5
    elif torch.sum(loop_mask) < 20:
        max_len = (max_len * 4) // 5
    else:
        max_len = max_len
        
    if torch.sum(nan_mask) <= max_len:
        residue_indices = [i for i in range(L) if nan_mask[i]==1]

    else:
        distance_map = get_distance_map(trans_1)
        distance_vectors = []
        
        for i in loop_indices:
            distance_vectors.append(distance_map[i])

        if distance_vectors == []:
            print()
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