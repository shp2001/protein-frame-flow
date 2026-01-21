import os
import gzip
from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import MMCIFParser
import io
import mdtraj as md 
import json 

from collections import defaultdict

import argparse
import dataclasses
import functools as fn
import time
import numpy as np
import yaml

from data import utils as du
from data import parsers
from data import errors

import tempfile 
import copy 

parser = argparse.ArgumentParser(
    description='PDB processing script.')
parser.add_argument(
    '--pdb_ids_path',
    help='Path to directory with PDB files.',
    type=str,
    default='/home/psh/protein-frame-flow/parse_general/meta/only_protein_filtered.txt')
parser.add_argument(
    '--num_processes',
    help='Number of processes.',
    type=int,
    default=12)
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true')
parser.add_argument(
    '--config',
    help='config file path',
    type=str,
    default='/home/psh/protein-frame-flow/parse_general/configs/datasets.yaml'
)

<<<<<<< HEAD
<<<<<<< HEAD
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
=======
 
def get_cif_path(cif_ID):
    cif_path = (
        "/public_data/BioMolDB_2024Oct21/cif/"
        + cif_ID[1:3]
        + "/"
        + cif_ID
        + ".cif.gz"
    )
    return cif_path

def mmcif_to_cif_text(gz_path: str) -> io.StringIO:
    with gzip.open(gz_path, 'rt') as f_in:
        cif_content = f_in.read()

    mmcif_dict = MMCIF2Dict(io.StringIO(cif_content))

    # 필요한 필드 추출
    group_PDBs  = mmcif_dict.get("_atom_site.group_PDB", [])
    atom_ids    = mmcif_dict.get("_atom_site.label_atom_id", [])
    comp_ids    = mmcif_dict.get("_atom_site.label_comp_id", [])
    asym_ids    = mmcif_dict.get("_atom_site.label_asym_id", [])
    seq_ids     = mmcif_dict.get("_atom_site.label_seq_id", [])
    alt_ids     = mmcif_dict.get("_atom_site.label_alt_id", [])
    entity_ids  = mmcif_dict.get("_atom_site.label_entity_id", [])
    x_coords    = mmcif_dict.get("_atom_site.Cartn_x", [])
    y_coords    = mmcif_dict.get("_atom_site.Cartn_y", [])
    z_coords    = mmcif_dict.get("_atom_site.Cartn_z", [])
    occupancies = mmcif_dict.get("_atom_site.occupancy", [])
    b_factors   = mmcif_dict.get("_atom_site.B_iso_or_equiv", [])
    elements    = mmcif_dict.get("_atom_site.type_symbol", [])
    model_nums  = mmcif_dict.get("_atom_site.pdbx_PDB_model_num", [])

    # auth 필드
    auth_seq_ids   = mmcif_dict.get("_atom_site.auth_seq_id", [])
    auth_comp_ids  = mmcif_dict.get("_atom_site.auth_comp_id", [])
    auth_asym_ids  = mmcif_dict.get("_atom_site.auth_asym_id", [])
    auth_atom_ids  = mmcif_dict.get("_atom_site.auth_atom_id", [])
    formal_charges = mmcif_dict.get("_atom_site.pdbx_formal_charge", [])
    ins_codes      = mmcif_dict.get("_atom_site.pdbx_PDB_ins_code", [])

    n_atoms = len(atom_ids)
    residue_dict = defaultdict(lambda: defaultdict(list))

    # 1. residue_dict 생성
    for i in range(n_atoms):
        if group_PDBs[i] != "ATOM":
            continue
        if comp_ids[i] in ("HOH", "WAT"):
            continue

        alt_id = alt_ids[i] if alt_ids[i] not in ('?', '.', '') else '.'
        occ = float(occupancies[i]) if occupancies[i] != '?' else 1.0
        residue_key = (asym_ids[i], seq_ids[i])  # (chain_id, residue_id)
        residue_dict[residue_key][alt_id].append((occ, i))

    # 2. alt_id 선택
    selected_indices = []

    for alt_groups in residue_dict.values():
        # 1. '.' altLoc인 애들 먼저 무조건 선택
        if '.' in alt_groups:
            selected_indices.extend([i for occ, i in alt_groups['.']])
        
        # 2. '.' 외의 altLoc 후보들 처리
        non_dot_alts = {k: v for k, v in alt_groups.items() if k != '.'}
        if non_dot_alts:
            best_alt_id = max(
                non_dot_alts.items(),
                key=lambda kv: sum(occ for occ, _ in kv[1]) / len(kv[1])
            )[0]
            selected_indices.extend([i for occ, i in non_dot_alts[best_alt_id]])
    # 출력 필드 정의
    atom_site_keys = [
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num"
>>>>>>> parent of dbd5ad1 (change the code of parsing general protein)
    ]

    output_lines = []
    output_lines.append("data_filtered_structure\n")
    output_lines.append("loop_")
    for k in atom_site_keys:
        output_lines.append(k)

    for new_id, i in enumerate(selected_indices, 1):
        def safe_get(lst, default="."):
            return lst[i] if i < len(lst) else default

        row = [
            "ATOM",                              # group_PDB
            str(new_id),                         # id (serial)
            safe_get(elements),                  # type_symbol
            safe_get(atom_ids),                  # label_atom_id
            safe_get(alt_ids),                   # alt_loc
            safe_get(comp_ids),                  # label_comp_id
            safe_get(asym_ids),                  # label_asym_id
            safe_get(entity_ids),                # label_entity_id
            safe_get(seq_ids),                   # label_seq_id
            safe_get(ins_codes),                 # ins_code
            safe_get(x_coords),                  # x
            safe_get(y_coords),                  # y
            safe_get(z_coords),                  # z
            safe_get(occupancies, "1.00"),       # occupancy
            safe_get(b_factors, "0.00"),         # B-factor
            safe_get(formal_charges, "."),       # formal charge
            safe_get(auth_seq_ids),              # auth_seq_id
            safe_get(auth_comp_ids),             # auth_comp_id
            safe_get(auth_asym_ids),             # auth_asym_id
            safe_get(auth_atom_ids),             # auth_atom_id
            safe_get(model_nums, "1")            # model number
        ]
        # 공백 포함되면 따옴표 처리
        row = [f"'{v}'" if ' ' in str(v) else v for v in row]
        output_lines.append(" ".join(row))

    output_lines.append("")  # 파일 종료

    cif_text = "\n".join(output_lines)
    return io.StringIO(cif_text)

def find_coil_regions_with_tolerance(ss_list, tolerance=4):
    coil_groups = []
    current_group = []
    gap_buffer = []

    for i, ss in enumerate(ss_list):
        if ss == 'C':
            if gap_buffer:
                current_group.extend(gap_buffer)
                gap_buffer = []
            current_group.append(i)
        elif ss in {'H', 'E'}:
            if current_group:
                gap_buffer.append(i)
                if len(gap_buffer) >= tolerance:
                    coil_groups.append(current_group)
                    current_group = []
                    gap_buffer = []
        else:
            # 다른 문자 또는 이상값이 오면 group 종료
            if current_group:
                coil_groups.append(current_group)
                current_group = []
                gap_buffer = []

<<<<<<< HEAD
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
=======
    if current_group:
        coil_groups.append(current_group)

    return coil_groups

def get_interface_residues_ca(traj, cutoff_angstrom=6.0):
    cutoff_nm = cutoff_angstrom / 10.0
    topology = traj.topology
    chains = list(topology.chains)
    interface_dict = defaultdict(set)

    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            chain_i = chains[i]
            chain_j = chains[j]

            ca_i = [atom.index for atom in chain_i.atoms if atom.name == 'CA']
            ca_j = [atom.index for atom in chain_j.atoms if atom.name == 'CA']

            if not ca_i or not ca_j:
                continue

            pairs = np.array([[a, b] for a in ca_i for b in ca_j])
            dists = md.compute_distances(traj, pairs)[0]
            close_pairs = pairs[dists < cutoff_nm]

            for a_idx, b_idx in close_pairs:
                res_i = topology.atom(a_idx).residue
                res_j = topology.atom(b_idx).residue

                interface_dict[res_i.chain.index].add(res_i.index)
                interface_dict[res_j.chain.index].add(res_j.index)

    return {k: sorted(list(v)) for k, v in interface_dict.items()}

def make_mask(file_path: str, cfg: str):
    '''
    file_path: .cif.gz format
    '''
    metadata = {}
    pdb_name = os.path.basename(file_path).replace('.cif.gz', '')
    if os.path.exists(os.path.join('/home/psh/data/general/masking_index', pdb_name + '.json')):
        print(f'{pdb_name} already exists')
        return None
    with open(cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # 1. CIF 필터링 수행
    cif_text_io = mmcif_to_cif_text(file_path)

    # 2. 임시 파일로 저장 (MMCIFParser는 파일 경로만 받음)
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".cif", delete=False) as tmp_cif:
        tmp_cif.write(cif_text_io.getvalue())
        tmp_cif_path = tmp_cif.name

    try:
        # 3. 마스킹 정보
        mask_json_dir = cfg['shared']['masking_dir']
        if not os.path.exists(mask_json_dir):
            os.makedirs(mask_json_dir, exist_ok=True)
            
        pdb_name = os.path.basename(file_path).replace('.cif.gz', '')

=======
 
def get_cif_path(cif_ID):
    cif_path = (
        "/public_data/BioMolDB_2024Oct21/cif/"
        + cif_ID[1:3]
        + "/"
        + cif_ID
        + ".cif.gz"
    )
    return cif_path

def mmcif_to_cif_text(gz_path: str) -> io.StringIO:
    with gzip.open(gz_path, 'rt') as f_in:
        cif_content = f_in.read()

    mmcif_dict = MMCIF2Dict(io.StringIO(cif_content))

    # 필요한 필드 추출
    group_PDBs  = mmcif_dict.get("_atom_site.group_PDB", [])
    atom_ids    = mmcif_dict.get("_atom_site.label_atom_id", [])
    comp_ids    = mmcif_dict.get("_atom_site.label_comp_id", [])
    asym_ids    = mmcif_dict.get("_atom_site.label_asym_id", [])
    seq_ids     = mmcif_dict.get("_atom_site.label_seq_id", [])
    alt_ids     = mmcif_dict.get("_atom_site.label_alt_id", [])
    entity_ids  = mmcif_dict.get("_atom_site.label_entity_id", [])
    x_coords    = mmcif_dict.get("_atom_site.Cartn_x", [])
    y_coords    = mmcif_dict.get("_atom_site.Cartn_y", [])
    z_coords    = mmcif_dict.get("_atom_site.Cartn_z", [])
    occupancies = mmcif_dict.get("_atom_site.occupancy", [])
    b_factors   = mmcif_dict.get("_atom_site.B_iso_or_equiv", [])
    elements    = mmcif_dict.get("_atom_site.type_symbol", [])
    model_nums  = mmcif_dict.get("_atom_site.pdbx_PDB_model_num", [])

    # auth 필드
    auth_seq_ids   = mmcif_dict.get("_atom_site.auth_seq_id", [])
    auth_comp_ids  = mmcif_dict.get("_atom_site.auth_comp_id", [])
    auth_asym_ids  = mmcif_dict.get("_atom_site.auth_asym_id", [])
    auth_atom_ids  = mmcif_dict.get("_atom_site.auth_atom_id", [])
    formal_charges = mmcif_dict.get("_atom_site.pdbx_formal_charge", [])
    ins_codes      = mmcif_dict.get("_atom_site.pdbx_PDB_ins_code", [])

    n_atoms = len(atom_ids)
    residue_dict = defaultdict(lambda: defaultdict(list))

    # 1. residue_dict 생성
    for i in range(n_atoms):
        if group_PDBs[i] != "ATOM":
            continue
        if comp_ids[i] in ("HOH", "WAT"):
            continue

        alt_id = alt_ids[i] if alt_ids[i] not in ('?', '.', '') else '.'
        occ = float(occupancies[i]) if occupancies[i] != '?' else 1.0
        residue_key = (asym_ids[i], seq_ids[i])  # (chain_id, residue_id)
        residue_dict[residue_key][alt_id].append((occ, i))

    # 2. alt_id 선택
    selected_indices = []

    for alt_groups in residue_dict.values():
        # 1. '.' altLoc인 애들 먼저 무조건 선택
        if '.' in alt_groups:
            selected_indices.extend([i for occ, i in alt_groups['.']])
        
        # 2. '.' 외의 altLoc 후보들 처리
        non_dot_alts = {k: v for k, v in alt_groups.items() if k != '.'}
        if non_dot_alts:
            best_alt_id = max(
                non_dot_alts.items(),
                key=lambda kv: sum(occ for occ, _ in kv[1]) / len(kv[1])
            )[0]
            selected_indices.extend([i for occ, i in non_dot_alts[best_alt_id]])
    # 출력 필드 정의
    atom_site_keys = [
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num"
    ]

    output_lines = []
    output_lines.append("data_filtered_structure\n")
    output_lines.append("loop_")
    for k in atom_site_keys:
        output_lines.append(k)

    for new_id, i in enumerate(selected_indices, 1):
        def safe_get(lst, default="."):
            return lst[i] if i < len(lst) else default

        row = [
            "ATOM",                              # group_PDB
            str(new_id),                         # id (serial)
            safe_get(elements),                  # type_symbol
            safe_get(atom_ids),                  # label_atom_id
            safe_get(alt_ids),                   # alt_loc
            safe_get(comp_ids),                  # label_comp_id
            safe_get(asym_ids),                  # label_asym_id
            safe_get(entity_ids),                # label_entity_id
            safe_get(seq_ids),                   # label_seq_id
            safe_get(ins_codes),                 # ins_code
            safe_get(x_coords),                  # x
            safe_get(y_coords),                  # y
            safe_get(z_coords),                  # z
            safe_get(occupancies, "1.00"),       # occupancy
            safe_get(b_factors, "0.00"),         # B-factor
            safe_get(formal_charges, "."),       # formal charge
            safe_get(auth_seq_ids),              # auth_seq_id
            safe_get(auth_comp_ids),             # auth_comp_id
            safe_get(auth_asym_ids),             # auth_asym_id
            safe_get(auth_atom_ids),             # auth_atom_id
            safe_get(model_nums, "1")            # model number
        ]
        # 공백 포함되면 따옴표 처리
        row = [f"'{v}'" if ' ' in str(v) else v for v in row]
        output_lines.append(" ".join(row))

    output_lines.append("")  # 파일 종료

    cif_text = "\n".join(output_lines)
    return io.StringIO(cif_text)

def find_coil_regions_with_tolerance(ss_list, tolerance=4):
    coil_groups = []
    current_group = []
    gap_buffer = []

    for i, ss in enumerate(ss_list):
        if ss == 'C':
            if gap_buffer:
                current_group.extend(gap_buffer)
                gap_buffer = []
            current_group.append(i)
        elif ss in {'H', 'E'}:
            if current_group:
                gap_buffer.append(i)
                if len(gap_buffer) >= tolerance:
                    coil_groups.append(current_group)
                    current_group = []
                    gap_buffer = []
        else:
            # 다른 문자 또는 이상값이 오면 group 종료
            if current_group:
                coil_groups.append(current_group)
                current_group = []
                gap_buffer = []

    if current_group:
        coil_groups.append(current_group)

    return coil_groups

def get_interface_residues_ca(traj, cutoff_angstrom=6.0):
    cutoff_nm = cutoff_angstrom / 10.0
    topology = traj.topology
    chains = list(topology.chains)
    interface_dict = defaultdict(set)

    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            chain_i = chains[i]
            chain_j = chains[j]

            ca_i = [atom.index for atom in chain_i.atoms if atom.name == 'CA']
            ca_j = [atom.index for atom in chain_j.atoms if atom.name == 'CA']

            if not ca_i or not ca_j:
                continue

            pairs = np.array([[a, b] for a in ca_i for b in ca_j])
            dists = md.compute_distances(traj, pairs)[0]
            close_pairs = pairs[dists < cutoff_nm]

            for a_idx, b_idx in close_pairs:
                res_i = topology.atom(a_idx).residue
                res_j = topology.atom(b_idx).residue

                interface_dict[res_i.chain.index].add(res_i.index)
                interface_dict[res_j.chain.index].add(res_j.index)

    return {k: sorted(list(v)) for k, v in interface_dict.items()}

def make_mask(file_path: str, cfg: str):
    '''
    file_path: .cif.gz format
    '''
    metadata = {}
    pdb_name = os.path.basename(file_path).replace('.cif.gz', '')
    if os.path.exists(os.path.join('/home/psh/data/general/masking_index', pdb_name + '.json')):
        print(f'{pdb_name} already exists')
        return None
    with open(cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # 1. CIF 필터링 수행
    cif_text_io = mmcif_to_cif_text(file_path)

    # 2. 임시 파일로 저장 (MMCIFParser는 파일 경로만 받음)
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".cif", delete=False) as tmp_cif:
        tmp_cif.write(cif_text_io.getvalue())
        tmp_cif_path = tmp_cif.name

    try:
        # 3. 마스킹 정보
        mask_json_dir = cfg['shared']['masking_dir']
        if not os.path.exists(mask_json_dir):
            os.makedirs(mask_json_dir, exist_ok=True)
            
        pdb_name = os.path.basename(file_path).replace('.cif.gz', '')

>>>>>>> parent of dbd5ad1 (change the code of parsing general protein)
        # 4. 구조 파싱 (mmCIF용 파서)
        parser = MMCIFParser(QUIET=False)
        structure = parser.get_structure(pdb_name, tmp_cif_path)
        models = [model for model in structure]
        if len(models) == 1:
            model = structure[0]
            struct_chains = {chain.id: chain for chain in model}
<<<<<<< HEAD
>>>>>>> parent of dbd5ad1 (change the code of parsing general protein)
=======
>>>>>>> parent of dbd5ad1 (change the code of parsing general protein)
        else:
            return None 

        # 5. 체인 정보 추출
        struct_chains = {chain.id: chain for chain in structure.get_chains()}
        metadata['num_chains'] = len(struct_chains)
        metadata['mode'] = 'monomer' if metadata['num_chains'] == 1 else 'polymer'
        struct_feats = []
        all_seqs = set()
        chain_lengths = dict()
        for chain_id, chain in struct_chains.items():
            chain_id_int = du.chain_str_to_int(chain_id)
            chain_prot = parsers.process_chain(chain, chain_id_int)
            chain_dict = dataclasses.asdict(chain_prot)
            chain_dict = du.parse_chain_feats(chain_dict)
            all_seqs.add(tuple(chain_dict['aatype']))
            chain_lengths[chain_id] = len(chain_dict['aatype'])
            struct_feats.append(chain_dict)

<<<<<<< HEAD
<<<<<<< HEAD
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
=======
        metadata['quaternary_category'] = 'homomer' if len(all_seqs) == 1 else 'heteromer'

        complex_feats = du.concat_np_features(struct_feats, False)
        complex_aatype = complex_feats['aatype']
        metadata['seq_len'] = len(complex_aatype)
>>>>>>> parent of dbd5ad1 (change the code of parsing general protein)

        if metadata['mode'] == 'monomer':
            # 1. MDTraj로 구조 불러오기
            traj = md.load(tmp_cif_path)
            # 2. DSSP 계산
            ss_array = md.compute_dssp(traj, simplified=True)  # simplified=True면 H, E, C만 나옴
            # 3. 체인 하나이므로 첫 프레임만 사용
            ss_list = ss_array[0].tolist()
            coil_groups = find_coil_regions_with_tolerance(ss_list)

<<<<<<< HEAD
    print(f"Start processing {total_files} files with {num_processes} processes...")

    with mp.Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_row, data_list), total=total_files))

    error_count = 0
    for res in results:
        if res is not None:
            print(res)
            error_count += 1
=======
=======
        metadata['quaternary_category'] = 'homomer' if len(all_seqs) == 1 else 'heteromer'

        complex_feats = du.concat_np_features(struct_feats, False)
        complex_aatype = complex_feats['aatype']
        metadata['seq_len'] = len(complex_aatype)

        if metadata['mode'] == 'monomer':
            # 1. MDTraj로 구조 불러오기
            traj = md.load(tmp_cif_path)
            # 2. DSSP 계산
            ss_array = md.compute_dssp(traj, simplified=True)  # simplified=True면 H, E, C만 나옴
            # 3. 체인 하나이므로 첫 프레임만 사용
            ss_list = ss_array[0].tolist()
            coil_groups = find_coil_regions_with_tolerance(ss_list)

>>>>>>> parent of dbd5ad1 (change the code of parsing general protein)
            filtered_coil_groups = [
                group for group in coil_groups
                if 0 not in group
                and (metadata['seq_len'] - 1) not in group
                and 5 <= len(group) <= 28
            ]
<<<<<<< HEAD
>>>>>>> parent of dbd5ad1 (change the code of parsing general protein)
=======
>>>>>>> parent of dbd5ad1 (change the code of parsing general protein)
            
            output_path = os.path.join(mask_json_dir, pdb_name + '.json')
            with open(output_path, 'w') as f:
                json.dump(filtered_coil_groups, f, indent=4)

        if metadata['mode'] == 'polymer':
            traj = md.load(tmp_cif_path)
            interface_residues = get_interface_residues_ca(traj)
            output_dict = {
                "interface_residues": interface_residues,
                "chain_lengths": chain_lengths
            }

            output_path = os.path.join(mask_json_dir, pdb_name + '.json')
            with open(output_path, "w") as f:
                json.dump(output_dict, f, indent=4)
    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_cif_path):
            os.remove(tmp_cif_path)

def process_fn(
        file_path,
        verbose=True,
        cfg=None):
    
    try:
        start_time = time.time()
        metadata = make_mask(
            file_path,
            cfg)
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
        return metadata
    except errors.DataError as e:
        if verbose:
            print(f'Failed {file_path}: {e}')

def main(args):
    pdb_ids_path = args.pdb_ids_path
    cfg = args.config
    debug = False  # 예: argparse.ArgumentParser에서 추가했다고 가정

    with open(pdb_ids_path, 'r') as f:
        pdb_ids = [line.strip() for line in f if line.strip()]

    all_file_paths = [get_cif_path(pdb_id) for pdb_id in pdb_ids]

    _process_fn = fn.partial(
        process_fn,
        verbose=args.verbose,
        cfg=cfg
    )

    if debug:
        # 디버그 모드: 순차 실행
        for file_path in all_file_paths:
            _process_fn(file_path)
    else:
        # 일반 모드: 멀티프로세싱 실행
        import multiprocessing as mp
        with mp.Pool(processes=args.num_processes) as pool:
            pool.map(_process_fn, all_file_paths)


if __name__ == "__main__":
<<<<<<< HEAD
<<<<<<< HEAD
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', type=str, default='/home/psh/data/loop_ppi/meta/metadata.csv')
    parser.add_argument('--num_processes', type=int, default=60, help='Number of parallel processes')
=======
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
>>>>>>> parent of dbd5ad1 (change the code of parsing general protein)
=======
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
>>>>>>> parent of dbd5ad1 (change the code of parsing general protein)
    args = parser.parse_args()
    main(args)
