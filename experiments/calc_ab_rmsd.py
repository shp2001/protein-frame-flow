import os
import json
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from multiprocessing import Pool
from tqdm import tqdm

from Bio.PDB import PDBParser, Superimposer
from Bio.Data.IUPACData import protein_letters_3to1
"""
Antibody-Antigen Structure RMSD Analysis Script (ID-Based Matching)
==================================================================

이 스크립트는 항체(Antibody)와 항원(Antigen) 복합체 구조를 비교하여 RMSD를 계산합니다.
단순한 인덱스 슬라이싱 대신 Residue ID(Chain ID, ResSeq, ICode)를 사용하여
Inference 결과물이 크롭(Cropping)되거나 서열 길이가 달라도 정확한 부위를 찾아 정렬합니다.


입력 데이터:
- label_dir: 기준이 되는 정답 PDB (unnumberred) 파일들이 위치한 디렉토리
- cdrflow_inf_dir: 모델이 생성한 Inference PDB 파일들이 위치한 디렉토리
- metadata_path: 각 PDB별 CDR 시작/끝 인덱스 정보가 담긴 CSV 파일 (training 혹은 inference에서 썼던 metadata)

주요 분석 시나리오:
1. Scenario 1 (Antigen-based Alignment):
   - 두 구조의 공통 항원(Antigen) CA 원자를 기준으로 정렬(Superimpose)합니다.
   - 정렬된 상태에서 항체 프레임워크(Framework)의 RMSD를 계산하여 상대적 위치 변화를 측정합니다.

2. Scenario 2 (Framework-based Alignment):
   - 두 구조의 공통 항체 프레임워크(Framework) CA 원자를 기준으로 정렬합니다.
   - 정렬된 상태에서 각 CDR(H1, H2, H3, L1, L2, L3)별 RMSD를 계산하여 루프 구조의 차이를 측정합니다.

핵심 매칭 로직:
- Label PDB의 인덱스 기반으로 CDR 범위를 먼저 정의합니다.
- 정의된 CDR의 Residue ID를 추출하여 Inference PDB에서 동일한 ID를 가진 residue를 검색합니다.
- Framework는 양쪽 구조에 공통으로 존재하는 Antibody residue 중 CDR이 아닌 것들로 자동 정의됩니다.
"""

# =========================================================
# Basic helpers
# =========================================================

def get_clean_residues(chain):
    """표준 아미노산만 추출 (HETATM 포함)"""
    clean = []
    for res in chain:
        resname = res.get_resname().capitalize()
        if resname in protein_letters_3to1:
            clean.append(res)
    return clean

def get_res_key(residue):
    """Residue 식별자: (Chain_ID, ResSeq, ICode) - ICode 공백 제거"""
    chain_id = residue.get_parent().id
    resseq = residue.id[1]
    icode = residue.id[2].strip()
    return (chain_id, resseq, icode)

def residue_to_aa(res):
    try:
        return protein_letters_3to1[res.get_resname().capitalize()]
    except KeyError:
        return "X"

def residues_to_sequence(residues: List):
    return "".join(residue_to_aa(r) for r in residues)

# =========================================================
# Sequence-based Matching Logic
# =========================================================

def get_matched_atoms_seq_based(label_cdr_res: List, inf_ab_res: List, atom_name: str = "CA"):
    """
    서열 기반 매칭 로직:
    1. Label CDR 서열을 Inference Antibody 전체 서열에서 검색
    2. 중복 발생 시 Residue ID가 가장 많이 겹치는 구간 선택
    """
    label_seq = residues_to_sequence(label_cdr_res)
    inf_seq = residues_to_sequence(inf_ab_res)
    
    # 모든 일치 위치 탐색 (overlapping matches 포함)
    match_starts = [m.start() for m in re.finditer(f'(?={label_seq})', inf_seq)]
    
    if not match_starts:
        return [], []

    best_start = match_starts[0]
    
    # 중복 서열 존재 시 ID 대조로 결정
    if len(match_starts) > 1:
        label_ids = [get_res_key(r) for r in label_cdr_res]
        max_overlap = -1
        for start in match_starts:
            candidate_res = inf_ab_res[start : start + len(label_cdr_res)]
            candidate_ids = [get_res_key(r) for r in candidate_res]
            overlap = len(set(label_ids) & set(candidate_ids))
            if overlap > max_overlap:
                max_overlap = overlap
                best_start = start

    matched_inf_res = inf_ab_res[best_start : best_start + len(label_cdr_res)]
    
    atoms1, atoms2 = [], []
    for r1, r2 in zip(label_cdr_res, matched_inf_res):
        if atom_name in r1 and atom_name in r2:
            atoms1.append(r1[atom_name])
            atoms2.append(r2[atom_name])
            
    return atoms1, atoms2

def match_atoms_by_id(res_list1: List, res_list2: List, atom_name: str = "CA"):
    """Antigen 매칭용: ID가 동일한 것들끼리 atom 추출"""
    map1 = {get_res_key(r): r[atom_name] for r in res_list1 if atom_name in r}
    map2 = {get_res_key(r): r[atom_name] for r in res_list2 if atom_name in r}
    common_keys = sorted(set(map1.keys()) & set(map2.keys()))
    return [map1[k] for k in common_keys], [map2[k] for k in common_keys]

# =========================================================
# Core analysis
# =========================================================

def analyze_double_alignment(pdb1_path, pdb2_path, cdr_ranges_global):
    parser = PDBParser(QUIET=True)
    s1 = parser.get_structure("Label", pdb1_path)
    s2 = parser.get_structure("Inf", pdb2_path)

    # 1. 데이터 로드 (ab_global: H+L 리스트)
    model1 = next(s1.get_models()); chains1 = list(model1.get_chains())
    model2 = next(s2.get_models()); chains2 = list(model2.get_chains())
    
    ab1 = get_clean_residues(chains1[0]) + get_clean_residues(chains1[1])
    ab2 = get_clean_residues(chains2[0]) + get_clean_residues(chains2[1])
    
    ag1 = []; [ag1.extend(get_clean_residues(c)) for c in chains1[2:]]
    ag2 = []; [ag2.extend(get_clean_residues(c)) for c in chains2[2:]]

    # 2. Framework Identification (ID match)
    all_cdr_ids = set()
    for start, end in cdr_ranges_global.values():
        for i in range(start, end + 1): # end 포함
            if i < len(ab1): all_cdr_ids.add(get_res_key(ab1[i]))

    ab1_map = {get_res_key(r): r for r in ab1}
    ab2_map = {get_res_key(r): r for r in ab2}
    common_ids = set(ab1_map.keys()) & set(ab2_map.keys())
    fr_ids = [rid for rid in common_ids if rid not in all_cdr_ids]
    
    fr_atoms1 = [ab1_map[rid]["CA"] for rid in fr_ids if "CA" in ab1_map[rid]]
    fr_atoms2 = [ab2_map[rid]["CA"] for rid in fr_ids if "CA" in ab2_map[rid]]

    output = {}

    # Scenario 1: Antigen-based alignment -> FR RMSD
    ag_atoms1, ag_atoms2 = match_atoms_by_id(ag1, ag2)
    if len(ag_atoms1) >= 3:
        sup_ag = Superimposer()
        sup_ag.set_atoms(ag_atoms1, ag_atoms2)
        sup_ag.apply(list(model2.get_atoms()))
        output["fr_rms"] = calculate_rmsd(fr_atoms1, fr_atoms2)
    else:
        output["fr_rms"] = None

    # Scenario 2: Framework-based alignment -> CDR RMSD
    if len(fr_atoms1) >= 3:
        sup_fr = Superimposer()
        sup_fr.set_atoms(fr_atoms1, fr_atoms2)
        sup_fr.apply(list(model2.get_atoms()))

        for cdr, (start, end) in cdr_ranges_global.items():
            label_cdr_res = ab1[start : end + 1] # end 포함 슬라이싱
            c1, c2 = get_matched_atoms_seq_based(label_cdr_res, ab2)
            output[f"{cdr}_rms"] = calculate_rmsd(c1, c2)
    else:
        for cdr in cdr_ranges_global: output[f"{cdr}_rms"] = None

    return output

def calculate_rmsd(atoms1, atoms2):
    if not atoms1 or not atoms2 or len(atoms1) != len(atoms2): return None
    c1 = np.array([a.coord for a in atoms1])
    c2 = np.array([a.coord for a in atoms2])
    return np.sqrt(np.mean(np.sum((c1 - c2)**2, axis=1)))

# =========================================================
# Runner
# =========================================================

def process_one_sample(args):
    label_path, inf_path, write_path, cdr_ranges = args
    try:
        output = analyze_double_alignment(label_path, inf_path, cdr_ranges)
        clean_output = {k: float(v) if v is not None else None for k, v in output.items()}
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        with open(write_path, "w") as f: json.dump(clean_output, f, indent=4)
        return True
    except Exception as e:
        return {"label": label_path, "inf": inf_path, "error": str(e)}

if __name__ == "__main__":
    label_dir = "/home/psh/benchmark_after210930/pdb"
    inf_base_dir = "/home/psh/af3_output_filtered"
    metadata_path = "/home/psh/benchmark_after210930/meta/metadata.csv"
    num_workers = 50
    
    df = pd.read_csv(metadata_path)
    tasks = []

    for pdb_id_ext in os.listdir(label_dir):
        if not pdb_id_ext.endswith(".pdb"): continue
        pdb_id = pdb_id_ext.replace(".pdb", "")

        inf_dir = os.path.join(inf_base_dir, pdb_id)
        if not os.path.exists(inf_dir): continue

        row = df[df["pdb_name"] == pdb_id]
        if row.empty: continue

        cdr_ranges = {
            t: (int(row[f"{t}_start"].iloc[0]), int(row[f"{t}_end"].iloc[0]))
            for t in ["h1", "h2", "h3", "l1", "l2", "l3"]
        }

        for sample in os.listdir(inf_dir):
            if not sample.endswith(".pdb") or any(x in sample for x in ["plddt", "trans"]):
                continue
            
            tasks.append((
                os.path.join(label_dir, pdb_id_ext),
                os.path.join(inf_dir, sample),
                os.path.join(inf_dir, "rmsd", sample.replace(".pdb", ".json")),
                cdr_ranges
            ))

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_one_sample, tasks), total=len(tasks)))

    errors = [r for r in results if r is not True]
    print(f"Done. Errors: {len(errors)}")
    if errors: print(f"Example error: {errors[0]}")