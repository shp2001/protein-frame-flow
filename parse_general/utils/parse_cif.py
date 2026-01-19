import os
import io
import gzip
import re
import time
import json
import yaml
import string
import shutil
import logging
import argparse
import tempfile
import traceback
import itertools
import functools as fn
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import MMCIFParser

# 사용자 정의 라이브러리 (환경에 맞게 경로 확인 필요)
from data import utils as du
from data import parsers
from data import errors

# =============================================================================
# Constants & Configuration
# =============================================================================

ALPHANUMERIC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

parser = argparse.ArgumentParser(description='PDB processing script (Full Pipeline).')
parser.add_argument('--pdb_ids_path', type=str, 
                    default='/home/psh/protein-frame-flow/parse_general/meta/only_protein_1.txt',
                    help='Path to text file containing list of PDB IDs.')
parser.add_argument('--num_processes', type=int, default=72,
                    help='Number of parallel processes.')
parser.add_argument('--write_dir', type=str, 
                    default='/home/psh/data/general_v2/meta',
                    help='Directory to save results (pkl, json, csv).')
parser.add_argument('--config', type=str, 
                    default='/home/psh/protein-frame-flow/parse_general/configs/datasets.yaml',
                    help='Path to config yaml file.')
parser.add_argument('--debug', action='store_true', help='Run in debug mode (serial processing).')
parser.add_argument('--verbose', action='store_true', help='Print detailed logs.')

# =============================================================================
# Helper Functions: CIF & Assembly Parsing
# =============================================================================

def get_cif_path(cif_ID):
    """PDB ID를 기반으로 CIF 파일 경로 생성 (서버 환경에 맞게 수정 필요)"""
    # 예시 경로 포맷
    cif_path = (
        "/public_data/BioMolDB_2024Oct21/cif/cif_raw/"
        + cif_ID[1:3]
        + "/"
        + cif_ID
        + ".cif.gz"
    )
    return cif_path

def parse_pdbx_struct_oper_list(mmcif_dict):
    """회전/이동 행렬 파싱"""
    ids = mmcif_dict.get("_pdbx_struct_oper_list.id", [])
    if not ids: return {}

    def get_float_list(key):
        return [float(x) for x in mmcif_dict.get(key, [])]

    try:
        m11 = get_float_list("_pdbx_struct_oper_list.matrix[1][1]")
        m12 = get_float_list("_pdbx_struct_oper_list.matrix[1][2]")
        m13 = get_float_list("_pdbx_struct_oper_list.matrix[1][3]")
        m21 = get_float_list("_pdbx_struct_oper_list.matrix[2][1]")
        m22 = get_float_list("_pdbx_struct_oper_list.matrix[2][2]")
        m23 = get_float_list("_pdbx_struct_oper_list.matrix[2][3]")
        m31 = get_float_list("_pdbx_struct_oper_list.matrix[3][1]")
        m32 = get_float_list("_pdbx_struct_oper_list.matrix[3][2]")
        m33 = get_float_list("_pdbx_struct_oper_list.matrix[3][3]")
        v1 = get_float_list("_pdbx_struct_oper_list.vector[1]")
        v2 = get_float_list("_pdbx_struct_oper_list.vector[2]")
        v3 = get_float_list("_pdbx_struct_oper_list.vector[3]")
    except Exception:
        return {}

    oper_dict = {}
    for i, oper_id in enumerate(ids):
        if i >= len(m11): break
        rotation = np.array([
            [m11[i], m12[i], m13[i]],
            [m21[i], m22[i], m23[i]],
            [m31[i], m32[i], m33[i]]
        ])
        translation = np.array([v1[i], v2[i], v3[i]])
        oper_dict[oper_id] = (rotation, translation)
    return oper_dict

def _expand_expression(expression):
    """(1-3,5) 형태의 표현식을 리스트로 확장"""
    groups = re.findall(r'\((.*?)\)', expression)
    if not groups: groups = [expression]

    parsed_groups = []
    for group in groups:
        tokens = group.split(',')
        group_ids = []
        for token in tokens:
            if '-' in token:
                try:
                    start, end = token.split('-')
                    group_ids.extend([str(x) for x in range(int(start), int(end) + 1)])
                except ValueError:
                    group_ids.append(token)
            else:
                group_ids.append(token.strip())
        parsed_groups.append(group_ids)
    
    all_ids = []
    for g in parsed_groups:
        all_ids.extend(g)
    return all_ids

def parse_assembly_gen(mmcif_dict):
    """Assembly 생성 명령어 파싱"""
    assembly_ids = mmcif_dict.get("_pdbx_struct_assembly_gen.assembly_id", [])
    oper_expressions = mmcif_dict.get("_pdbx_struct_assembly_gen.oper_expression", [])
    asym_id_lists = mmcif_dict.get("_pdbx_struct_assembly_gen.asym_id_list", [])

    instructions = []
    # assembly_id가 '1'인 것을 우선 찾음, 없으면 첫 번째 것 사용
    target_indices = [i for i, aid in enumerate(assembly_ids) if aid == '1']
    if not target_indices and assembly_ids:
        target_indices = [0]
    
    for i in target_indices:
        if i >= len(oper_expressions) or i >= len(asym_id_lists): continue
        op_expr = oper_expressions[i]
        chain_list = [c.strip() for c in asym_id_lists[i].split(',')]
        ops = _expand_expression(op_expr)
        instructions.append({'chains': chain_list, 'opers': ops})
    return instructions

def chain_id_generator():
    """A-Z, a-z, 0-9, AA, AB... 순서로 ID 생성"""
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    for size in itertools.count(1):
        for s in itertools.product(chars, repeat=size):
            yield "".join(s)

# =============================================================================
# Core Logic: mmCIF to Clean CIF Text (with Chain Remapping)
# =============================================================================

def mmcif_to_cif_text(gz_path: str) -> io.StringIO:
    try:
        with gzip.open(gz_path, 'rt') as f_in:
            cif_content = f_in.read()

        mmcif_dict = MMCIF2Dict(io.StringIO(cif_content))

        # --- Data Extraction ---
        group_PDBs = mmcif_dict.get("_atom_site.group_PDB", [])
        atom_ids = mmcif_dict.get("_atom_site.label_atom_id", [])
        comp_ids = mmcif_dict.get("_atom_site.label_comp_id", [])
        asym_ids = mmcif_dict.get("_atom_site.label_asym_id", [])
        seq_ids = mmcif_dict.get("_atom_site.label_seq_id", [])
        alt_ids = mmcif_dict.get("_atom_site.label_alt_id", [])
        entity_ids = mmcif_dict.get("_atom_site.label_entity_id", [])
        x_coords = mmcif_dict.get("_atom_site.Cartn_x", [])
        y_coords = mmcif_dict.get("_atom_site.Cartn_y", [])
        z_coords = mmcif_dict.get("_atom_site.Cartn_z", [])
        occupancies = mmcif_dict.get("_atom_site.occupancy", [])
        b_factors = mmcif_dict.get("_atom_site.B_iso_or_equiv", [])
        elements = mmcif_dict.get("_atom_site.type_symbol", [])
        formal_charges = mmcif_dict.get("_atom_site.pdbx_formal_charge", [])
        ins_codes = mmcif_dict.get("_atom_site.pdbx_PDB_ins_code", [])

        n_atoms = len(atom_ids)
        if n_atoms == 0: raise ValueError("No atoms found")

        # --- AltLoc Filtering ---
        residue_dict = defaultdict(lambda: defaultdict(list))
        for i in range(n_atoms):
            if group_PDBs[i] != "ATOM": continue
            if comp_ids[i] in ("HOH", "WAT"): continue
            
            alt = alt_ids[i] if alt_ids[i] not in ('?', '.', '') else '.'
            occ = float(occupancies[i]) if occupancies[i] != '?' else 1.0
            # (Chain, ResidueSeq)를 키로 사용
            residue_dict[(asym_ids[i], seq_ids[i])][alt].append((occ, i))

        selected_indices_set = set()
        for alt_groups in residue_dict.values():
            if '.' in alt_groups:
                selected_indices_set.update([i for occ, i in alt_groups['.']])
            
            # '.'이 아닌 altloc 중 가장 occupancy가 높은 것 선택
            non_dot = {k: v for k, v in alt_groups.items() if k != '.'}
            if non_dot:
                best = max(non_dot.items(), key=lambda kv: sum(o for o, _ in kv[1])/len(kv[1]))[0]
                selected_indices_set.update([i for occ, i in non_dot[best]])

        selected_indices = sorted(list(selected_indices_set))

        # --- Assembly Expansion ---
        oper_dict = parse_pdbx_struct_oper_list(mmcif_dict)
        instructions = parse_assembly_gen(mmcif_dict)
        
        # Assembly 정보가 없으면 Identity(1)로 처리
        if not instructions or not oper_dict:
            unique_chains = set([asym_ids[i] for i in selected_indices])
            instructions = [{'chains': list(unique_chains), 'opers': ['1']}]
            if '1' not in oper_dict: oper_dict['1'] = (np.eye(3), np.zeros(3))

        # --- Chain ID Remapping Strategy ---
        chain_mapper = {}
        used_new_ids = set()
        id_gen = chain_id_generator()

        # 1. Chain Mapping 테이블 생성
        # 순서를 보장하여 Op '1'(Identity)이나 앞쪽 체인이 우선권을 갖도록 함
        for instr in instructions:
            target_chains = sorted(list(set(instr['chains'])))
            target_opers = sorted(instr['opers'], key=lambda x: (len(x), x))

            for op_id in target_opers:
                if op_id not in oper_dict: continue
                
                for old_chain in target_chains:
                    pair_key = (old_chain, op_id)
                    if pair_key in chain_mapper:
                        continue
                    
                    # (규칙 1) 원본 ID가 1글자이고 사용 가능하면 -> 유지
                    if (len(old_chain) == 1 and 
                        old_chain in ALPHANUMERIC and 
                        old_chain not in used_new_ids):
                        
                        new_safe_chain_id = old_chain
                    
                    # (규칙 2) 아니면(2글자 이상 or 이미 선점) -> 새 ID 생성
                    else:
                        while True:
                            candidate = next(id_gen)
                            if candidate not in used_new_ids:
                                new_safe_chain_id = candidate
                                break
                    
                    chain_mapper[pair_key] = new_safe_chain_id
                    used_new_ids.add(new_safe_chain_id)

        # --- Generate New Atoms ---
        final_atoms = []
        new_serial_id = 1

        for instr in instructions:
            target_chains = set(instr['chains'])
            target_opers = instr['opers']

            for op_id in target_opers:
                if op_id not in oper_dict: continue
                rot, trans = oper_dict[op_id]
                
                # 해당 Operator에 해당하는 Atom들만 순회 (효율성을 위해 전체 Loop 안에서 필터링)
                # 실제로는 selected_indices 전체를 돌면서 필터링하는 구조
                for i in selected_indices:
                    chain_id = asym_ids[i]
                    if chain_id not in target_chains: continue

                    original_coord = np.array([float(x_coords[i]), float(y_coords[i]), float(z_coords[i])])
                    new_coord = np.dot(rot, original_coord) + trans
                    
                    # 위에서 확정한 Mapping ID 사용
                    new_safe_chain_id = chain_mapper[(chain_id, op_id)]

                    def safe_get(lst, idx, default="."):
                        return lst[idx] if idx < len(lst) else default

                    atom_data = {
                        "group_PDB": "ATOM",
                        "id": str(new_serial_id),
                        "type_symbol": safe_get(elements, i),
                        "label_atom_id": safe_get(atom_ids, i),
                        "label_alt_id": ".",
                        "label_comp_id": safe_get(comp_ids, i),
                        "label_asym_id": new_safe_chain_id, # Remapped Chain ID
                        "label_entity_id": safe_get(entity_ids, i),
                        "label_seq_id": safe_get(seq_ids, i),
                        "pdbx_PDB_ins_code": safe_get(ins_codes, i),
                        "Cartn_x": f"{new_coord[0]:.3f}",
                        "Cartn_y": f"{new_coord[1]:.3f}",
                        "Cartn_z": f"{new_coord[2]:.3f}",
                        "occupancy": "1.00",
                        "B_iso_or_equiv": safe_get(b_factors, i, "0.00"),
                        "pdbx_formal_charge": safe_get(formal_charges, i, "."),
                        "auth_seq_id": safe_get(seq_ids, i),
                        "auth_comp_id": safe_get(comp_ids, i),
                        "auth_asym_id": new_safe_chain_id, # Remapped Chain ID
                        "auth_atom_id": safe_get(atom_ids, i),
                        "pdbx_PDB_model_num": "1"
                    }
                    final_atoms.append(atom_data)
                    new_serial_id += 1

        # --- Output String Construction ---
        atom_site_keys = [
            "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
            "_atom_site.label_atom_id", "_atom_site.label_alt_id", "_atom_site.label_comp_id",
            "_atom_site.label_asym_id", "_atom_site.label_entity_id", "_atom_site.label_seq_id",
            "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x", "_atom_site.Cartn_y",
            "_atom_site.Cartn_z", "_atom_site.occupancy", "_atom_site.B_iso_or_equiv",
            "_atom_site.pdbx_formal_charge", "_atom_site.auth_seq_id", "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id", "_atom_site.auth_atom_id", "_atom_site.pdbx_PDB_model_num"
        ]

        output_lines = ["data_filtered_structure", "loop_"]
        output_lines.extend(atom_site_keys)
        for atom in final_atoms:
            row = []
            for k in atom_site_keys:
                short_key = k.replace("_atom_site.", "")
                val = str(atom.get(short_key, "."))
                # CIF 포맷: 공백이 포함된 값은 따옴표 처리
                row.append(f"'{val}'" if ' ' in val else val)
            output_lines.append(" ".join(row))
        output_lines.append("#")
        
        return io.StringIO("\n".join(output_lines))

    except Exception as e:
        # 에러 발생 시 로그 출력 후 re-raise
        print(f"\n[ERROR] Failed to process file in mmcif_to_cif_text: {gz_path}")
        print(traceback.format_exc())
        raise e

# =============================================================================
# Main Processing Function
# =============================================================================

def process_cif_file(file_path: str, write_dir: str, cfg: dict):
    metadata = {}

    # 1. Clean & Expand CIF Text 생성
    cif_text_io = mmcif_to_cif_text(file_path)

    # 2. Temp File 생성 (MMCIFParser는 파일 경로 필요)
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".cif", delete=False) as tmp_cif:
        tmp_cif.write(cif_text_io.getvalue())
        tmp_cif_path = tmp_cif.name

    try:
        pdb_name = os.path.basename(file_path).replace('.cif.gz', '')
        
        # 경로 설정
        mask_json_dir = cfg['shared']['masking_dir']
        mask_filename = f"{pdb_name}.json"
        mask_info_file = os.path.join(mask_json_dir, mask_filename)
        
        processed_path = os.path.join(write_dir, f'{pdb_name}.pkl')
        
        # 메타데이터 기본 정보
        metadata['pdb_name'] = pdb_name
        metadata['raw_path'] = file_path
        metadata['processed_path'] = os.path.abspath(processed_path)
        metadata['mask_info_file'] = os.path.abspath(mask_info_file)

        # 3. Biopython 파싱
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_name, tmp_cif_path)
        
        # 모델 개수 확인 (보통 1개여야 함)
        models = list(structure)
        if len(models) != 1:
            # 여러 모델이 있어도 첫 번째만 사용하거나 에러 처리 (여기선 첫 번째 사용)
            pass
        model = models[0]

        # 4. Feature Extraction (data.parsers 활용)
        struct_chains = {chain.id: chain for chain in model}
        metadata['num_chains'] = len(struct_chains)
        metadata['mode'] = 'monomer' if metadata['num_chains'] == 1 else 'polymer'

        struct_feats = []
        all_seqs = set()
        
        for chain_id, chain in struct_chains.items():
            # data.utils의 chain 변환 함수 사용 가정
            chain_id_int = du.chain_str_to_int(chain_id)
            
            # data.parsers의 체인 처리 함수
            chain_prot = parsers.process_chain(chain, chain_id_int)
            chain_dict = dataclasses.asdict(chain_prot)
            chain_dict = du.parse_chain_feats(chain_dict)
            
            all_seqs.add(tuple(chain_dict['aatype']))
            struct_feats.append(chain_dict)

        metadata['quaternary_category'] = 'homomer' if len(all_seqs) == 1 else 'heteromer'

        # Feature 병합
        complex_feats = du.concat_np_features(struct_feats, False)
        complex_aatype = complex_feats['aatype']
        metadata['seq_len'] = len(complex_aatype)
        
        modeled_idx = np.where(complex_aatype != 20)[0] # 20: Usually gap or unknown
        if len(modeled_idx) == 0:
            raise errors.LengthError('No modeled residues')

        metadata['modeled_seq_len'] = int(np.max(modeled_idx) - np.min(modeled_idx) + 1)
        complex_feats['modeled_idx'] = modeled_idx

        # 5. 결과 저장 (Pickle)
        du.write_pkl(processed_path, complex_feats)
        
        # 6. Mask Info 저장 (JSON) - 빈 딕셔너리로 초기화하거나 필요한 정보 채움
        # (기존 코드 흐름상 여기선 파일을 생성해두는 것이 중요해 보임)
        os.makedirs(os.path.dirname(mask_info_file), exist_ok=True)
        if not os.path.exists(mask_info_file):
            with open(mask_info_file, 'w') as f:
                json.dump({}, f)

        return metadata

    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_cif_path):
            os.remove(tmp_cif_path)

# =============================================================================
# Multiprocessing Wrappers
# =============================================================================

def process_wrapper(file_path, write_dir, cfg, verbose):
    try:
        start_time = time.time()
        metadata = process_cif_file(file_path, write_dir, cfg)
        elapsed = time.time() - start_time
        if verbose:
            print(f"Finished {os.path.basename(file_path)} in {elapsed:.2f}s")
        return metadata
    except Exception as e:
        if verbose:
            print(f"Failed {os.path.basename(file_path)}: {e}")
        # 에러 발생 시 None 반환하여 메타데이터 집계에서 제외
        return None

def main(args):
    # Config 로드
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 출력 디렉토리 생성
    if not os.path.exists(args.write_dir):
        os.makedirs(args.write_dir)

    # PDB ID 리스트 로드
    with open(args.pdb_ids_path, 'r') as f:
        pdb_ids = [line.strip() for line in f if line.strip()]

    all_file_paths = [get_cif_path(pid) for pid in pdb_ids]
    total_files = len(all_file_paths)
    
    print(f"Start processing {total_files} files using {args.num_processes} processes.")
    print(f"Output Directory: {args.write_dir}")

    # Partial Function 생성 (고정 인자 설정)
    _process_fn = fn.partial(
        process_wrapper,
        write_dir=args.write_dir,
        cfg=cfg,
        verbose=args.verbose
    )

    all_metadata = []

    if args.debug or args.num_processes == 1:
        # Serial Execution
        for path in all_file_paths:
            res = _process_fn(path)
            if res: all_metadata.append(res)
    else:
        # Parallel Execution
        with mp.Pool(processes=args.num_processes) as pool:
            # tqdm 없이 기본적인 진행상황만 보고 싶다면 map 사용
            # 결과 순서 보장은 map이 하지만, 완료되는 대로 받고 싶다면 imap_unordered 추천
            results = pool.map(_process_fn, all_file_paths)
            all_metadata = [r for r in results if r is not None]

    # CSV 저장
    csv_name = 'metadata_debug.csv' if args.debug else 'metadata.csv'
    csv_path = os.path.join(args.write_dir, csv_name)
    
    df = pd.DataFrame(all_metadata)
    df.to_csv(csv_path, index=False)
    
    print(f"Done. Successfully processed {len(all_metadata)}/{total_files} files.")
    print(f"Metadata saved to {csv_path}")

if __name__ == "__main__":
    # GPU 비활성화
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # dataclasses가 import 되어야 함 (parsers 내부에서 사용됨)
    import dataclasses 
    
    args = parser.parse_args()
    main(args)