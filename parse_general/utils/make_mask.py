import os
import json
import yaml
import numpy as np
import pandas as pd
import mdtraj as md
from scipy.spatial.distance import cdist
import dataclasses
from data import utils as du 
from tqdm import tqdm
import multiprocessing as mp

# protein-frame-flow/OpenFold 등에서 사용하는 표준 Alphanumeric
ALPHANUMERIC = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

def int_to_chain_str(chain_idx):
    """
    정수형 Chain Index를 문자열로 변환합니다.
    """
    if chain_idx < len(ALPHANUMERIC):
        return ALPHANUMERIC[chain_idx]
    else:
        return str(chain_idx)

def get_cb_or_ca(atom_positions, atom_mask):
    """
    (L, 37, 3) -> (L, 3)
    CB가 존재(mask=1)하면 CB, 아니면 CA 좌표 반환
    """
    has_cb = atom_mask[:, 3] > 0.5
    coords = np.zeros((atom_positions.shape[0], 3), dtype=np.float32)
    coords[has_cb] = atom_positions[has_cb, 3, :]
    coords[~has_cb] = atom_positions[~has_cb, 1, :]
    return coords

def fill_gaps_and_group(selected_indices, valid_residue_set, max_gap):
    """
    선택된 인덱스들 사이의 Gap을 채우되, 
    'valid_residue_set'에 실제로 존재하는 Residue ID만 포함시킵니다.
    """
    if len(selected_indices) == 0:
        return []

    # 1. 정렬
    sorted_idx = sorted(list(set(selected_indices)))
    
    # 2. Gap 채우기 (존재하는 잔기만 확인)
    filled_indices_set = set(sorted_idx)
    
    if len(sorted_idx) > 1:
        for i in range(1, len(sorted_idx)):
            prev = sorted_idx[i-1]
            curr = sorted_idx[i]
            diff = curr - prev
            
            # Gap 조건: 거리 max_gap 이하
            if diff > 1 and diff <= (max_gap + 1):
                # prev와 curr 사이의 모든 숫자에 대해 실제 존재하는지 확인
                for candidate_id in range(prev + 1, curr):
                    if candidate_id in valid_residue_set:
                        filled_indices_set.add(candidate_id)
    
    # 다시 리스트로 변환 및 정렬
    final_indices = sorted(list(filled_indices_set))
    
    # 3. 연속된 번호끼리 블록(Block)화
    if not final_indices:
        return []

    grouped_blocks = []
    current_block = [final_indices[0]]
    
    for i in range(1, len(final_indices)):
        if final_indices[i] == final_indices[i-1] + 1:
            current_block.append(final_indices[i])
        else:
            grouped_blocks.append(current_block)
            current_block = [final_indices[i]]
    grouped_blocks.append(current_block)
    
    return grouped_blocks

def trim_and_filter_blocks(blocks, chain_start_res, chain_end_res):
    """
    블록 리스트를 받아 말단 잔기 포함 여부에 따라 트리밍하고,
    길이가 4 미만인 블록을 제거합니다.
    """
    final_blocks = []
    
    for block in blocks:
        if not block:
            continue
            
        # 현재 블록의 N-term/C-term 포함 여부 확인
        # (트리밍 전 원본 블록 기준 확인)
        has_n_term = (chain_start_res in block)
        has_c_term = (chain_end_res in block)
        
        # 1. N-term 처리
        if has_n_term:
            if len(block) < 4:
                continue # 4 미만이면 전체 삭제
            block = block[4:] # 앞 4개 제거
            
        if not block: 
            continue # N-term 자르다가 다 없어졌으면 스킵

        # 2. C-term 처리
        if has_c_term:
            if len(block) < 4:
                continue # 4 미만이면 전체 삭제
            block = block[:-4] # 뒤 4개 제거
            
        # 3. 최종 길이 필터 (4 미만 제거)
        if len(block) >= 4:
            final_blocks.append(block)
            
    return final_blocks

def create_mdtraj_obj(features):
    """
    features 딕셔너리로부터 MDTraj Trajectory 생성
    """
    aatype = features['aatype']
    n_res = len(aatype)
    
    top = md.Topology()
    chain = top.add_chain()
    
    restypes = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    ]
    
    for aa_idx in aatype:
        if 0 <= aa_idx < 20:
            res_name = restypes[aa_idx]
        else:
            res_name = 'ALA' 
        
        residue = top.add_residue(res_name, chain)
        top.add_atom('N', md.element.nitrogen, residue)
        top.add_atom('CA', md.element.carbon, residue)
        top.add_atom('C', md.element.carbon, residue)
        top.add_atom('O', md.element.oxygen, residue)

    indices = [0, 1, 2, 4] # N, CA, C, O
    xyz = features['atom_positions'][:, indices, :] 
    xyz_flat = xyz.reshape(1, n_res * 4, 3) / 10.0
    
    traj = md.Trajectory(xyz_flat, top)
    return traj

def process_row(row):
    """
    개별 행(Row)을 처리하는 Worker 함수
    """
    try:
        pdb_name = row['pdb_name']
        pkl_path = row['processed_path']
        mode = row['mode']
        
        if not os.path.exists(pkl_path):
            return f"Skipping {pdb_name}: pkl not found."
            
        data = du.read_pkl(pkl_path)
        
        residue_index = data['residue_index'] 
        chain_index = data['chain_index']     
        atom_positions = data['atom_positions']
        atom_mask = data['atom_mask']
        
        result_json = {} 
        unique_chains = np.unique(chain_index)

        # 공통적으로 사용할 좌표 계산 (monomer 모드 제외하고는 필요함)
        coords = None
        interface_mask_all = None
        
        if mode in ['polymer', 'loop_ppi']:
            coords = get_cb_or_ca(atom_positions, atom_mask)
            dists = cdist(coords, coords)
            
            # 같은 체인 간의 거리는 무한대로 설정 (Interface만 보기 위함)
            c_matrix = chain_index[:, None] == chain_index[None, :]
            dists[c_matrix] = np.inf
            
            min_dists = np.min(dists, axis=1)
            interface_mask_all = min_dists < 8.0

        if mode == 'polymer':
            for c_idx in unique_chains:
                chain_residues_mask = (chain_index == c_idx)
                if not np.any(chain_residues_mask): continue

                chain_res_ids = residue_index[chain_residues_mask]
                valid_residue_set = set(chain_res_ids)
                
                chain_start_res = np.min(chain_res_ids)
                chain_end_res = np.max(chain_res_ids)
                
                c_mask = chain_residues_mask & interface_mask_all
                raw_res_ids = residue_index[c_mask]
                
                grouped_ids = fill_gaps_and_group(
                    raw_res_ids, 
                    valid_residue_set=valid_residue_set, 
                    max_gap=5
                )
                
                final_blocks = trim_and_filter_blocks(grouped_ids, chain_start_res, chain_end_res)
                
                if final_blocks:
                    c_str = int_to_chain_str(int(c_idx))
                    result_json[c_str] = final_blocks

        elif mode == 'monomer':
            traj = create_mdtraj_obj(data)
            dssp = md.compute_dssp(traj, simplified=True)[0] 
            loop_mask_all = (dssp == 'C')
            
            for c_idx in unique_chains:
                chain_residues_mask = (chain_index == c_idx)
                if not np.any(chain_residues_mask): continue

                chain_res_ids = residue_index[chain_residues_mask]
                valid_residue_set = set(chain_res_ids)
                
                chain_start_res = np.min(chain_res_ids)
                chain_end_res = np.max(chain_res_ids)
                
                c_mask = chain_residues_mask & loop_mask_all
                raw_res_ids = residue_index[c_mask]
                
                grouped_ids = fill_gaps_and_group(
                    raw_res_ids, 
                    valid_residue_set=valid_residue_set, 
                    max_gap=4
                )
                
                final_blocks = trim_and_filter_blocks(grouped_ids, chain_start_res, chain_end_res)
                
                if final_blocks:
                    c_str = int_to_chain_str(int(c_idx))
                    result_json[c_str] = final_blocks
        
        elif mode == 'loop_ppi':
            # 1. Loop 계산
            traj = create_mdtraj_obj(data)
            dssp = md.compute_dssp(traj, simplified=True)[0] 
            loop_mask_all = (dssp == 'C')

            for c_idx in unique_chains:
                chain_residues_mask = (chain_index == c_idx)
                if not np.any(chain_residues_mask): continue

                chain_res_ids = residue_index[chain_residues_mask]
                valid_residue_set = set(chain_res_ids)
                chain_start_res = np.min(chain_res_ids)
                chain_end_res = np.max(chain_res_ids)

                # 2. Loop Residue 추출 및 블록화 (Gap Filling 수행)
                c_loop_mask = chain_residues_mask & loop_mask_all
                raw_loop_ids = residue_index[c_loop_mask]
                
                grouped_loop_blocks = fill_gaps_and_group(
                    raw_loop_ids, 
                    valid_residue_set=valid_residue_set, 
                    max_gap=4
                )

                c_interface_mask = chain_residues_mask & interface_mask_all
                interface_res_ids_set = set(residue_index[c_interface_mask])

                valid_blocks = []
                for block in grouped_loop_blocks:
                    if not interface_res_ids_set.isdisjoint(block):
                        valid_blocks.append(block)

                final_blocks = trim_and_filter_blocks(valid_blocks, chain_start_res, chain_end_res)

                if final_blocks:
                    c_str = int_to_chain_str(int(c_idx))
                    result_json[c_str] = final_blocks

        # JSON 저장
        if 'mask_info_file' in row and pd.notna(row['mask_info_file']):
            save_path = row['mask_info_file']
            
            if result_json:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                clean_json = {}
                for k, v in result_json.items():
                    clean_json[k] = [[int(x) for x in block] for block in v]

                with open(save_path, 'w') as f:
                    json.dump(clean_json, f, indent=4)
        else:
            return f"Skipping {pdb_name}: mask_info_file path is missing."

    except Exception as e:
        import traceback
        return f"Error processing {row.get('pdb_name', 'unknown')}: {e}\n{traceback.format_exc()}"
    
    return None

def process_metadata(metadata_path, num_processes):
    df = pd.read_csv(metadata_path)
    
    if 'seq_len' in df.columns:
        original_count = len(df)
        df = df[df['seq_len'] <= 2000]
        filtered_count = len(df)
        print(f"Filtered metadata: {original_count} -> {filtered_count} (seq_len <= 2000)")
    else:
        print("Warning: 'seq_len' column not found. Skipping filter.")

    data_list = df.to_dict('records')
    total_files = len(data_list)

    if total_files == 0:
        print("No files to process after filtering.")
        return

    print(f"Start processing {total_files} files with {num_processes} processes...")

    with mp.Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_row, data_list), total=total_files))

    error_count = 0
    for res in results:
        if res is not None:
            print(res)
            error_count += 1
            
    print(f"Finished. Total: {total_files}, Errors: {error_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', type=str, default='/home/psh/data/loop_ppi/meta/metadata.csv')
    parser.add_argument('--num_processes', type=int, default=60, help='Number of parallel processes')
    args = parser.parse_args()
    
    process_metadata(args.metadata_path, args.num_processes)