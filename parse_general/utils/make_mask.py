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

def fill_gaps_and_group(indices, max_gap):
    """
    정수 리스트를 받아 Gap을 채우고, 연속된 구간을 블록화하여 이중 리스트로 반환
    """
    if len(indices) == 0:
        return []

    sorted_idx = sorted(list(set(indices)))
    filled_indices = []
    
    # 1. Fill Gaps
    if len(sorted_idx) > 0:
        filled_indices.append(sorted_idx[0])
        for i in range(1, len(sorted_idx)):
            prev = sorted_idx[i-1]
            curr = sorted_idx[i]
            diff = curr - prev
            
            if diff <= (max_gap + 1) and diff > 1:
                filled_indices.extend(range(prev + 1, curr))
            
            filled_indices.append(curr)
    
    # 2. Group Consecutive
    grouped_blocks = []
    if not filled_indices:
        return []

    current_block = [filled_indices[0]]
    
    for i in range(1, len(filled_indices)):
        if filled_indices[i] == filled_indices[i-1] + 1:
            current_block.append(filled_indices[i])
        else:
            grouped_blocks.append(current_block)
            current_block = [filled_indices[i]]
    grouped_blocks.append(current_block)
    
    return grouped_blocks

def create_mdtraj_obj(features):
    """
    features 딕셔너리로부터 MDTraj Trajectory 생성
    """
    aatype = features['aatype']
    n_res = len(aatype)
    
    # Create Topology
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
    xyz = features['atom_positions'][:, indices, :] # (L, 4, 3)
    
    xyz_flat = xyz.reshape(1, n_res * 4, 3) / 10.0
    
    traj = md.Trajectory(xyz_flat, top)
    return traj

def process_row(row):
    """
    개별 행(Row)을 처리하는 Worker 함수 (병렬 처리용)
    """
    try:
        pdb_name = row['pdb_name']
        pkl_path = row['processed_path']
        mode = row['mode']
        
        # Pickle 로드
        if not os.path.exists(pkl_path):
            return f"Skipping {pdb_name}: pkl not found."
            
        data = du.read_pkl(pkl_path)
        
        residue_index = data['residue_index'] 
        chain_index = data['chain_index']     
        atom_positions = data['atom_positions']
        atom_mask = data['atom_mask']
        
        result_json = {} 

        if mode == 'polymer':
            coords = get_cb_or_ca(atom_positions, atom_mask)
            dists = cdist(coords, coords)
            
            c_matrix = chain_index[:, None] == chain_index[None, :]
            dists[c_matrix] = np.inf
            
            min_dists = np.min(dists, axis=1)
            interface_mask = min_dists < 8.0
            
            unique_chains = np.unique(chain_index)
            
            for c_idx in unique_chains:
                c_mask = (chain_index == c_idx) & interface_mask
                raw_res_ids = residue_index[c_mask]
                grouped_ids = fill_gaps_and_group(raw_res_ids, max_gap=5)
                c_str = int_to_chain_str(int(c_idx))
                result_json[c_str] = grouped_ids

        elif mode == 'monomer':
            traj = create_mdtraj_obj(data)
            dssp = md.compute_dssp(traj, simplified=True)[0] 
            loop_mask = (dssp == 'C')
            
            unique_chains = np.unique(chain_index)
            
            for c_idx in unique_chains:
                c_mask = (chain_index == c_idx) & loop_mask
                raw_res_ids = residue_index[c_mask]
                grouped_ids = fill_gaps_and_group(raw_res_ids, max_gap=4)
                c_str = int_to_chain_str(int(c_idx))
                result_json[c_str] = grouped_ids

        # JSON 저장
        if 'mask_info_file' in row and pd.notna(row['mask_info_file']):
            save_path = row['mask_info_file']
            
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
    
    # === [Modification] Filter rows where seq_len <= 2000 ===
    if 'seq_len' in df.columns:
        original_count = len(df)
        df = df[df['seq_len'] <= 2000]
        filtered_count = len(df)
        print(f"Filtered metadata: {original_count} -> {filtered_count} (seq_len <= 2000)")
    else:
        print("Warning: 'seq_len' column not found. Skipping filter.")
    # ========================================================

    # DataFrame을 dict list로 변환 (병렬 처리를 위해)
    data_list = df.to_dict('records')
    total_files = len(data_list)

    if total_files == 0:
        print("No files to process after filtering.")
        return

    print(f"Start processing {total_files} files with {num_processes} processes...")

    # 병렬 처리 시작
    with mp.Pool(num_processes) as pool:
        # imap을 사용하여 순서대로 결과를 받으며 tqdm 업데이트
        results = list(tqdm(pool.imap(process_row, data_list), total=total_files))

    # 에러 로그 출력
    error_count = 0
    for res in results:
        if res is not None:
            print(res)
            error_count += 1
            
    print(f"Finished. Total: {total_files}, Errors: {error_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', type=str, default='/home/psh/data/general_v2/meta/metadata.csv')
    parser.add_argument('--num_processes', type=int, default=60, help='Number of parallel processes')
    args = parser.parse_args()
    
    process_metadata(args.metadata_path, args.num_processes)