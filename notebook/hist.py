import os
import numpy as np
from data.utils import read_pkl
from data import residue_constants
from data.featurizer import get_ref_basic_feature
from openfold.data.data_transforms import make_atom14_masks
from multiprocessing import Pool
from tqdm import tqdm
import torch 

def check_consistency(args):
    meta_dir, file_name = args
    filepath = os.path.join(meta_dir, file_name)
    raw_info = read_pkl(filepath)
    
    # 1. 데이터 전처리 및 Tensor 변환
    chain_info = {
        'aatype': torch.tensor(raw_info['aatype']).long(),
        'chain_index': torch.tensor(raw_info['chain_index']),
        'all_atom_positions': torch.tensor(raw_info['atom_positions']).float(),
        'all_atom_mask': torch.tensor(raw_info['atom_mask']).float(),
        'residue_index': torch.tensor(raw_info['residue_index']).long(), # 필수 포함
        'seq_mask': torch.tensor(raw_info['bb_mask']).int()
    }
    
    # OpenFold transform 적용 (atom14 관련 마스크 생성)
    chain_info = make_atom14_masks(chain_info)
    
    aatype = chain_info['aatype'][None, ...]
    atom14_atom_exists = chain_info['atom14_atom_exists'][None, ...]
    residue_index = chain_info['residue_index'][None, ...]
    all_atom_mask = chain_info['all_atom_mask']
    
    # 2. get_ref_basic_feature 실행을 통해 ref_pos 획득
    outputs = get_ref_basic_feature(aatype, atom14_atom_exists, residue_index)
    ref_pos = outputs[-1] # (B, N_ref_atoms, 3)
    
    # 3. 개수 측정
    # (1) ref_pos의 2번째 차원 (추출된 원자 총 개수)
    count_ref_pos = ref_pos.shape[1]
    
    # (2) atom14_atom_exists에서 1인 원소 개수
    count_atom14 = int(torch.sum(atom14_atom_exists == 1))
    
    # (3) all_atom_mask에서 1인 원소 개수
    count_all_atom = int(torch.sum(all_atom_mask == 1))
    
    # 4. 검증: 세 값이 모두 같은지 확인
    if not (count_ref_pos == count_atom14):
        print(f"[MISMATCH] {filepath}")
        return {
            'file': file_name,
            'ref_pos': count_ref_pos,
            'atom14': count_atom14,
            'all_atom': count_all_atom,
            'status': 'mismatch'
        }



if __name__ == "__main__":
    meta_dir = '/home/psh/data/loop_ppi_v2/meta'
    all_files = [f for f in os.listdir(meta_dir) if f.endswith('.pkl')]
    
    # 테스트를 위해 개수 조절 가능 (예: [:100])
    files_to_check = all_files
    tasks = [(meta_dir, f) for f in files_to_check]
    
    num_cores = 90
    print(f"Checking {len(tasks)} files using {num_cores} cores...")
    
    mismatches = []
    errors = []

    with Pool(processes=num_cores) as pool:
        for result in tqdm(pool.imap_unordered(check_consistency, tasks), total=len(tasks)):
            if result:
                if result['status'] == 'mismatch':
                    mismatches.append(result)
                else:
                    errors.append(result)

    # 결과 출력
    print("\n" + "="*50)
    print(f"Analysis Complete.")
    print(f"Total Mismatches: {len(mismatches)}")
    print(f"Total Errors: {len(errors)}")
    print("="*50)

    if mismatches:
        print("\n[Mismatch Details]")
        for m in mismatches[:10]: # 너무 많을 수 있으니 상위 10개만 출력
            print(f"File: {m['file']} | ref_pos: {m['ref_pos']}, atom14: {m['atom14']}, all_atom: {m['all_atom']}")