import sys
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

def load_loop_file(loop_file, seed=None):
    """Gets the index of a given CDR loop"""

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
        loop_indices = json.load(file)
    
    if seed is not None:
        random.seed(seed)
    
    loop_index = random.choice(loop_indices)

    # 시작과 끝을 clipping
    start = max(0, int(loop_index[0]))
    end = min(seq_len - 1, int(loop_index[-1]))

    return start, end

def load_polymer_mask(mask_info_file, seed=None):
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

    interface = random.choices(interface_list, weights=[len(g) for g in interface_list])[0]

    # 5. clipping: 선택된 체인의 범위 안에서만 확장
    min_res = max(start_index, interface[0] - 5)
    max_res = min(end_index, interface[-1] + 5)
    
    interface = [res for res in interface if min_res <= res <= max_res]

    # 6. 길이 자르기 (30 초과시 연속 30개)
    if len(interface) > 30:
        start = random.randint(0, len(interface) - 30)
        interface = interface[start:start + 30]

    return int(min_res), int(max_res)
    
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

def crop_antigen(trans_1, cdr_mask, nan_mask, max_len, seq_list, crop_ab, mode='ab'): 
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
        
    return residue_indices

######################## crop_general_protein ########################

def crop_general_protein(trans_1, loop_mask, nan_mask, max_len, seq_list):  
    chain_len_list = [len(seq) for seq in seq_list]
    L = sum(chain_len_list)
    residue_indices = None

    anchor = find_anchor(loop_mask, only_h3=False)
    start = anchor[0] ; end = anchor[1]
    loop_indices = [i for i in range(start+1, end) if nan_mask[i] == 1]

    if len(loop_indices) < 10:
        max_len = max_len // 2
    elif len(loop_indices) < 15:
        max_len = (max_len * 3) // 5
    elif len(loop_indices) < 20:
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

def get_ag_hotspot(pseudo_beta, loop_mask, diffuse_mask, threshold=5.0):
    """
    pseudo_beta:   (B, L, 3) 좌표
    loop_mask:     (B, L) 1이면 loop residue
    diffuse_mask:  (B, L) 0이면 antigen residue
    threshold:     contact 거리 cutoff
    """
    B, L, _ = pseudo_beta.shape
    device = pseudo_beta.device

    # 모든 batch에서 동일하므로 첫 번째 것만 사용
    loop_mask = loop_mask[0].bool()          # (L,)
    ag_mask   = (diffuse_mask[0] == 0).bool() # (L,)

    # 거리 행렬 (B, L, L)
    dist_map = torch.cdist(pseudo_beta, pseudo_beta)  # (B, L, L)

    # loop vs antigen 거리만 추출
    d = dist_map[:, loop_mask][:, :, ag_mask]   # (B, num_loop, num_ag)

    # antigen residue별 contact 여부
    contact = (d < threshold).any(dim=1)        # (B, num_ag)

    # hotspot 초기화
    hotspot = torch.zeros((B, L), dtype=torch.float, device=device)
    ag_idx = torch.where(ag_mask)[0]
    hotspot[:, ag_idx] = contact.float()

    return hotspot  # (B, L)