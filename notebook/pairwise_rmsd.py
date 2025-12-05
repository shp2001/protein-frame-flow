import torch
import numpy as np
from Bio import PDB
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed


def load_single_pdb(pdb_file):
    """멀티프로세스에서 안전하게 단일 PDB 로드"""
    print(pdb_file)
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_file)
    coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())

    arr = np.array(coords, dtype=np.float32)
    return arr   # numpy로 반환 (Tensor는 프로세스 간 전달 시 느림)


def calculate_prealigned_pairwise_rmsd_numpy(coords_list, loop_mask):
    """좌표 리스트를 numpy로 받고 RMSD 계산 (단일 프로세스)"""
    device = loop_mask.device
    coords_batch = torch.tensor(np.stack(coords_list), dtype=torch.float32, device=device)

    loop_mask = loop_mask.bool()
    loop_coords = coords_batch[:, loop_mask, :]   # (B, L, 3)

    num_pdbs = loop_coords.shape[0]
    if num_pdbs < 2:
        return 0.0

    # pairwise diff → RMSD
    diff = loop_coords.unsqueeze(1) - loop_coords.unsqueeze(0)
    msd_matrix = (diff ** 2).sum(dim=-1).mean(dim=-1)
    rmsd_matrix = torch.sqrt(msd_matrix)

    # 상삼각 평균
    rows, cols = torch.triu_indices(num_pdbs, num_pdbs, offset=1)
    average_rmsd = rmsd_matrix[rows, cols].mean().item()
    return average_rmsd


def compute_pairwise_rmsd_parallel(pdb_files, loop_mask, max_workers=16):
    """멀티프로세스로 PDB 읽기 + 단일 프로세스 RMSD 계산"""

    coords_list = []

    # ----------- 1) 병렬로 모든 PDB 파일 로드 -----------
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single_pdb, f): f for f in pdb_files}
        for future in as_completed(futures):
            pdb_path = futures[future]
            try:
                coords_list.append(future.result())
            except Exception as e:
                print(f"[ERROR] {pdb_path}: {e}")

    # ----------- 2) 단일 프로세스에서 RMSD 계산 -----------
    result = calculate_prealigned_pairwise_rmsd_numpy(coords_list, loop_mask)
    return result

root_dir = '/home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.3.2.1_stage_2_ori_perturb/2025-12-04_22-18-03/epoch=16-step=24361_copy/run_2025-12-05_14-16-57'

for mt_id in os.listdir(root_dir):
    if 'config' in mt_id:
        continue

    mt_dir = os.path.join(root_dir, mt_id)
    files_list = os.listdir(mt_dir)

    pdb_files = [
        os.path.join(mt_dir, sample, 'sample_1.pdb')
        for sample in files_list if 'loop_mask' not in sample
    ]

    loop_mask_path = os.path.join(mt_dir, 'loop_mask.pt')
    loop_mask = torch.load(loop_mask_path, map_location='cpu')[0]

    # ---------- 병렬 RMSD 계산 ----------
    result = compute_pairwise_rmsd_parallel(
        pdb_files,
        loop_mask,
        max_workers=16   # CPU 개수에 맞게 조절
    )

    result_save_path = os.path.join(mt_dir, 'pairwise_rmsd.json')
    with open(result_save_path, 'w') as f:
        json.dump({"pairwise_rmsd": result}, f, indent=4)
