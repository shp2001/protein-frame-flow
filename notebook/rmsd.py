from Bio import PDB
import numpy as np
import os
import pandas as pd 

########################################################################################################
kkh_dir = "/home/psh/protein-frame-flow/inference_outputs/kkh_20251020_chothia"
cdrflow_dir = '/home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.3.2.1_stage_2_wt_confidence_lr_1e-3/2025-10-26_10-47-26/epoch=42-step=61619/kkh'
plddt_csv = '/home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.3.2.1_stage_2_wt_confidence_lr_1e-3/2025-10-26_10-47-26/epoch=42-step=61619/kkh/get_capri/capri_info_with_plddt_merged_with_kkh.csv'
########################################################################################################
csv_info = pd.read_csv(plddt_csv)

# -------------------------------
# Chothia numbering 기준 CDR residue ranges
# -------------------------------
CDR_DEFS = {
    "h1": (26, 32),
    "h2": (52, 56),
    "h3": (95, 102),
    "l1": (24, 34),
    "l2": (50, 56),
    "l3": (89, 97),
}

# -------------------------------
# Chain ID 기준 Heavy/Light 순서 지정
# -------------------------------
def get_chain_sequences_by_id(structure):
    chains = list(structure.get_chains())
    heavy_chains = [c for c in chains if c.id.upper() == 'A']
    light_chains = [c for c in chains if c.id.upper() == 'B']
    if not heavy_chains or not light_chains:
        heavy_chains = [chains[0]]
        light_chains = [chains[1]]
    return heavy_chains + light_chains

# -------------------------------
# Residue.id[1] 기준 dict 생성
# -------------------------------
def get_residues_dict(chain):
    return {f"{res.id[1]}_{res.id[2]}": res for res in chain if PDB.is_aa(res, standard=True)}

# -------------------------------
# 공통 backbone atom 추출 (heavy+light chain)
# -------------------------------
def get_common_atoms_all(chains_ref, chains_mob, only_frame=False):
    ref_atoms_all = []
    mob_atoms_all = []
    common_residue_map = {}  # chain_idx -> res_id -> (ref_atom_indices, mob_atom_indices)

    for chain_idx, (chain_ref, chain_mob) in enumerate(zip(chains_ref, chains_mob)):
        residues_ref = get_residues_dict(chain_ref)
        residues_mob = get_residues_dict(chain_mob)
        common_res_ids = sorted(set(residues_ref.keys()) & set(residues_mob.keys()))

        chain_ref_indices = []
        chain_mob_indices = []

        # CDR 범위 정의
        if chain_idx == 0:
            cdr_ranges = [CDR_DEFS[k] for k in CDR_DEFS if k.startswith('h')]
        else:
            cdr_ranges = [CDR_DEFS[k] for k in CDR_DEFS if k.startswith('l')]

        for rid in common_res_ids:
            # only_frame=True면 CDR 영역 제외
            res_num = int(rid.split('_')[0])
            if only_frame and any(start <= res_num <= end for start, end in cdr_ranges):
                continue  # skip CDR residues if only_frame

            res_ref = residues_ref[rid]
            res_mob = residues_mob[rid]
            for atom_name in ["N", "CA", "C"]:
                if atom_name in res_ref and atom_name in res_mob:
                    ref_atoms_all.append(res_ref[atom_name])
                    mob_atoms_all.append(res_mob[atom_name])
                    chain_ref_indices.append(len(ref_atoms_all)-1)
                    chain_mob_indices.append(len(mob_atoms_all)-1)

            # chain_idx -> res_id -> (ref_indices, mob_indices)
            if chain_ref_indices and chain_mob_indices:
                common_residue_map.setdefault(chain_idx, {})[rid] = (
                    chain_ref_indices[-3:], chain_mob_indices[-3:]
                )

    return ref_atoms_all, mob_atoms_all, common_residue_map

# -------------------------------
# Superpose 구조
# -------------------------------
def superpose_structures(ref_atoms, mob_atoms, mob_structure):
    assert len(ref_atoms) == len(mob_atoms)

    sup = PDB.Superimposer()
    sup.set_atoms(ref_atoms, mob_atoms)
    sup.apply(mob_structure.get_atoms())
    return sup.rms

# -------------------------------
# 전체 계산 함수
# -------------------------------
def compute_cdr_rmsd_save_aligned(ref_pdb, mob_pdb):
    parser = PDB.PDBParser(QUIET=True)
    s_ref = parser.get_structure("ref", ref_pdb)
    s_mob = parser.get_structure("mob", mob_pdb)

    chains_ref = get_chain_sequences_by_id(s_ref)
    chains_mob = get_chain_sequences_by_id(s_mob)

    # -------------------------------
    # 공통 backbone atom 추출 (Heavy+Light)
    # -------------------------------
    ref_atoms_all, mob_atoms_all, common_residue_map = get_common_atoms_all(chains_ref, chains_mob, only_frame=False)
    ref_atoms_frame, mob_atoms_frame, common_residue_map_frame = get_common_atoms_all(chains_ref, chains_mob, only_frame=True)

    # Superpose
    framework_rms = superpose_structures(ref_atoms_frame, mob_atoms_frame, s_mob)

    # -------------------------------
    # CDR RMSD 계산
    # -------------------------------
    all_cdr_rmsds = {}
    for chain_idx, chain in enumerate(chains_ref):
        cdr_types = [k for k in CDR_DEFS if (k.startswith('h') if chain_idx==0 else k.startswith('l'))]
        for cdr_name in cdr_types:
            start, end = CDR_DEFS[cdr_name]
            residues_map = common_residue_map.get(chain_idx, {})

            # CDR 범위 안에 들어있는 residue만 선택
            cdr_ref_indices = []
            cdr_mob_indices = []
            for rid in residues_map:
                if start <= int(rid.split('_')[0]) <= end:
                    ref_idx, mob_idx = residues_map[rid]
                    cdr_ref_indices.extend(ref_idx)
                    cdr_mob_indices.extend(mob_idx)

            diffs = np.array([ref_atoms_all[i].coord - mob_atoms_all[j].coord
                              for i,j in zip(cdr_ref_indices, cdr_mob_indices)])
            rmsd = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))
            all_cdr_rmsds[cdr_name] = rmsd

    return framework_rms, all_cdr_rmsds



# ==========================================================
# 사용 예시
# ==========================================================
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # 진행률 표시


# ==========================================================
# RMSD 계산용 함수 (행 단위)
# ==========================================================
def process_rmsd_row(row):
    pdb_id = row['pdb_id']
    data_name = row['data_name']
    sample_id = row['sample_id']

    pdb1 = os.path.join(kkh_dir, pdb_id, data_name + '.pdb')
    pdb2 = os.path.join(cdrflow_dir, pdb_id, data_name, sample_id, 'sample_1_chothia.pdb')

    # PDB 존재 확인
    frame_rms, cdr_rmsds = compute_cdr_rmsd_save_aligned(pdb1, pdb2)


    # 결과 딕셔너리 반환
    result = {
        'frame_rmsd': frame_rms,
        'h1_rmsd': cdr_rmsds.get('h1'),
        'h2_rmsd': cdr_rmsds.get('h2'),
        'h3_rmsd': cdr_rmsds.get('h3'),
        'l1_rmsd': cdr_rmsds.get('l1'),
        'l2_rmsd': cdr_rmsds.get('l2'),
        'l3_rmsd': cdr_rmsds.get('l3'),
    }
    return row.name, result

# ==========================================================
# 병렬 처리 + 진행률 표시
# ==========================================================
num_workers = 80  # CPU 코어 수
results = {}

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(process_rmsd_row, row): idx for idx, row in csv_info.iterrows()}
    
    # tqdm으로 진행률 표시
    for future in tqdm(as_completed(futures), total=len(futures), desc="Computing RMSDs"):
        idx, res_dict = future.result()
        results[idx] = res_dict

# ==========================================================
# 결과 CSV에 적용
# ==========================================================
for idx, res_dict in results.items():
    for col, val in res_dict.items():
        csv_info.at[idx, col] = val

# 저장
output_csv = os.path.join(cdrflow_dir, 'get_capri', 'capri_info_with_plddt_rmsd.csv')
csv_info.to_csv(output_csv, index=False)
print(f"병렬 RMSD 계산 완료. 저장된 CSV: {output_csv}")