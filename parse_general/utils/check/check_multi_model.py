import os
import gzip
from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import MMCIFParser
import io

import pandas as pd
from collections import defaultdict

import argparse
import dataclasses
import functools as fn
import pandas as pd
import os
import multiprocessing as mp
import time
import numpy as np
import yaml

from data import utils as du
from data import parsers
from data import errors

import string 
import tempfile 
import copy 


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
    ins_codes       = mmcif_dict.get("_atom_site.pdbx_PDB_ins_code", [])

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

def process_cif_file(file_path: str, cfg: str):
    '''
    file_path: .cif.gz format
    '''
    metadata = {}

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
        mask_filename = os.path.basename(file_path).replace('.cif.gz', '.json')
        mask_info_file = os.path.join(mask_json_dir, mask_filename)
        metadata['mask_info_file'] = mask_info_file

        pdb_name = os.path.basename(file_path).replace('.cif.gz', '')
        metadata['pdb_name'] = pdb_name
        metadata['raw_path'] = file_path

        # 4. 구조 파싱 (mmCIF용 파서)
        parser = MMCIFParser(QUIET=False)
        structure = parser.get_structure(pdb_name, tmp_cif_path)
        models = [model for model in structure]

        if len(models) > 1:
            print(pdb_name)
        else:
            print("pass")
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
        process_cif_file(
            file_path,
            cfg)
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
    except errors.DataError as e:
        if verbose:
            print(f'Failed {file_path}: {e}')


pdb_ids_path = '/home/psh/protein-frame-flow/parse_general/meta/only_protein_filtered.txt'
cfg = '/home/psh/protein-frame-flow/parse_general/configs/datasets.yaml'

with open(pdb_ids_path, 'r') as f:
    pdb_ids = [line.strip() for line in f if line.strip()]

all_file_paths = [get_cif_path(pdb_id) for pdb_id in pdb_ids]

_process_fn = fn.partial(
    process_fn,
    verbose=False,
    cfg=cfg)
with mp.Pool(processes=12) as pool:
    pool.map(_process_fn, all_file_paths)


