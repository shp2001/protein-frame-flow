
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from multiprocessing import Pool
from tqdm import tqdm

from Bio.PDB import PDBParser, Superimposer

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
    """표준 아미노산(HETATM 제외)만 반환"""
    return [res for res in chain if res.id[0] == " "]

def get_res_key(residue):
    """Residue 식별자 반환: (Chain_ID, ResSeq, ICode)"""
    return (residue.get_parent().id, residue.id[1], residue.id[2])

# =========================================================
# Atom Matching Logic
# =========================================================

def match_atoms_by_id(res_list1: List, res_list2: List, atom_name: str = "CA"):
    """
    두 residue 리스트에서 ID가 동일한 것들끼리 atom 추출
    """
    map1 = {get_res_key(r): r[atom_name] for r in res_list1 if atom_name in r}
    map2 = {get_res_key(r): r[atom_name] for r in res_list2 if atom_name in r}
    
    common_keys = sorted(set(map1.keys()) & set(map2.keys()))
    
    atoms1 = [map1[k] for k in common_keys]
    atoms2 = [map2[k] for k in common_keys]
    
    return atoms1, atoms2

# =========================================================
# Structure parsing
# =========================================================

def get_structure_data(structure):
    """
    첫 번째 모델 사용 (H, L, Ag 순서 가정)
    """
    model = next(structure.get_models())
    chains = list(model.get_chains())

    if len(chains) < 3:
        raise ValueError("Structure must contain at least 3 chains (H, L, Ag)")

    h_res = get_clean_residues(chains[0])
    l_res = get_clean_residues(chains[1])
    ag_res = get_clean_residues(chains[2])

    ab_global = h_res + l_res
    all_atoms = list(model.get_atoms())

    return ab_global, ag_res, all_atoms

# =========================================================
# Core analysis
# =========================================================

def analyze_double_alignment(
    pdb1_path: str, # Label (Reference)
    pdb2_path: str, # Inference (Target)
    cdr_ranges_global: Dict[str, Tuple[int, int]],
):
    parser = PDBParser(QUIET=True)
    s1 = parser.get_structure("Label", pdb1_path)
    s2 = parser.get_structure("Inf", pdb2_path)

    ab1, ag1, all_atoms1 = get_structure_data(s1)
    ab2, ag2, all_atoms2 = get_structure_data(s2)

    # 1. CDR Residues 식별 (Label 기준 Index 사용)
    cdr_res_ids = {k: set() for k in cdr_ranges_global}
    all_cdr_ids = set()

    for cdr, (start, end) in cdr_ranges_global.items():
        # Label의 ab1 리스트 index 기준 추출
        for i in range(start, end):
            if i < len(ab1):
                res_id = get_res_key(ab1[i])
                cdr_res_ids[cdr].add(res_id)
                all_cdr_ids.add(res_id)

    # 2. Framework Residues 분리 (ID 매칭 기반)
    # 양쪽 구조에 공통으로 존재하는 Antibody residue 중 CDR이 아닌 것
    ab1_map = {get_res_key(r): r for r in ab1}
    ab2_map = {get_res_key(r): r for r in ab2}
    common_ab_ids = set(ab1_map.keys()) & set(ab2_map.keys())
    
    fr_ids = [rid for rid in common_ab_ids if rid not in all_cdr_ids]
    
    fr_atoms1 = [ab1_map[rid]["CA"] for rid in fr_ids if "CA" in ab1_map[rid]]
    fr_atoms2 = [ab2_map[rid]["CA"] for rid in fr_ids if "CA" in ab2_map[rid]]

    # 3. Antigen Atoms 매칭
    ag_atoms1, ag_atoms2 = match_atoms_by_id(ag1, ag2)

    output = {}

    # Scenario 1: Antigen-based alignment -> FR RMSD
    if len(ag_atoms1) >= 3:
        sup_ag = Superimposer()
        sup_ag.set_atoms(ag_atoms1, ag_atoms2)
        sup_ag.apply(all_atoms2)
        output["fr_rms"] = calculate_rmsd(fr_atoms1, fr_atoms2)
    else:
        output["fr_rms"] = None

    # Scenario 2: Framework-based alignment -> CDR RMSD
    if len(fr_atoms1) >= 3:
        sup_fr = Superimposer()
        sup_fr.set_atoms(fr_atoms1, fr_atoms2)
        sup_fr.apply(all_atoms2)

        for cdr, target_ids in cdr_res_ids.items():
            # 현재 CDR에 해당하는 ID들 중 Inference에도 존재하는 것들 매칭
            matched_ids = target_ids & set(ab2_map.keys())
            c1 = [ab1_map[rid]["CA"] for rid in matched_ids if "CA" in ab1_map[rid]]
            c2 = [ab2_map[rid]["CA"] for rid in matched_ids if "CA" in ab2_map[rid]]
            output[f"{cdr}_rms"] = calculate_rmsd(c1, c2)
    else:
        for cdr in cdr_ranges_global:
            output[f"{cdr}_rms"] = None

    return output

def calculate_rmsd(atoms1, atoms2):
    if not atoms1 or not atoms2 or len(atoms1) != len(atoms2):
        return None
    c1 = np.array([a.coord for a in atoms1])
    c2 = np.array([a.coord for a in atoms2])
    return np.sqrt(np.mean(np.sum((c1 - c2)**2, axis=1)))

# =========================================================
# Parallel worker
# =========================================================

def process_one_sample(args):
    label_sample, cdrflow_sample, write_path, cdr_ranges = args

    try:
        output = analyze_double_alignment(
            label_sample,
            cdrflow_sample,
            cdr_ranges,
        )
        clean_output = {}
        for k, v in output.items():
            if v is not None:
                clean_output[k] = float(v)  # np.float32 -> python float
            else:
                clean_output[k] = None
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        with open(write_path, "w") as f:
            json.dump(clean_output, f, indent=4)

        return True

    except Exception as e:
        return {
            "label": label_sample,
            "sample": cdrflow_sample,
            "error": str(e),
        }


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    label_dir = "/home/psh/benchmark_after210930/pdb_unnum"
    cdrflow_inf_dir = "/home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.4.0_loop_ppi/2026-01-29_00-29-37/epoch=57-step=41470/benchmark_w_perturb_w_ag_cond"
    metadata_path = "/home/psh/benchmark_after210930/meta/metadata.csv"
    num_workers = 30
    df = pd.read_csv(metadata_path)

    tasks = []

    for pdb_id in os.listdir(label_dir):
        if not pdb_id.endswith(".pdb"):
            continue

        pdb_inf_dir = os.path.join(
            cdrflow_inf_dir,
            pdb_id.replace(".pdb", "")
        )
        if not os.path.exists(pdb_inf_dir):
            continue

        filtered_row = df[df["pdb_name"] == pdb_id.replace(".pdb", "")]
        if filtered_row.empty:
            continue

        label_sample = os.path.join(label_dir, pdb_id)

        cdr_ranges = {}
        for cdr_type in ["h1", "h2", "h3", "l1", "l2", "l3"]:
            start_val = int(filtered_row[f"{cdr_type}_start"].iloc[0])
            end_val = int(filtered_row[f"{cdr_type}_end"].iloc[0])
            cdr_ranges[cdr_type] = (start_val, end_val)

        for sample in os.listdir(pdb_inf_dir):
            if "plddt" in sample or "trans" in sample:
                continue

            cdrflow_sample = os.path.join(pdb_inf_dir, sample)
            if os.path.isdir(cdrflow_sample):
                continue

            write_path = os.path.join(
                pdb_inf_dir,
                "rmsd",
                sample.replace(".pdb", ".json"),
            )

            tasks.append((
                label_sample,
                cdrflow_sample,
                write_path,
                cdr_ranges,
            ))

    print(f"Total tasks: {len(tasks)}")
    print(f"Using {num_workers} workers")

    results = []
    with Pool(num_workers) as pool:
        for res in tqdm(
            pool.imap_unordered(process_one_sample, tasks),
            total=len(tasks),
            desc="RMSD calculation",
        ):
            results.append(res)

    errors = [r for r in results if r is not True]
    print(f"Failed samples: {len(errors)}")

    if errors:
        print("Example error:")
        print(errors[0])
