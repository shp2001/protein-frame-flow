import os
import gzip
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from io import StringIO
from multiprocessing import Pool, cpu_count

def has_polypeptide_chain(cif_dict):
    types = cif_dict.get("_entity_poly.type", [])
    if isinstance(types, str):
        types = [types]
    for t in types:
        if t.strip().lower() in ("polypeptide(l)", "polypeptide(d)"):
            return True
    return False

def check_polypeptide_in_pdb(args):
    pdb_id, base_dir = args
    cif_path = os.path.join(base_dir, pdb_id[1:3], pdb_id + ".cif.gz")
    try:
        with gzip.open(cif_path, 'rt') as f:
            cif_data = f.read()
        cif_dict = MMCIF2Dict(StringIO(cif_data))
        result = has_polypeptide_chain(cif_dict)
        print(f"{pdb_id} has polypeptide: {result}")
        return (pdb_id, result)
    except Exception as e:
        print(f"Error checking {pdb_id}: {e}")
        return (pdb_id, False)

def validate_pdb_list(pdb_txt_path, cif_base_dir):
    with open(pdb_txt_path, 'r') as f:
        pdb_ids = [line.strip() for line in f if line.strip()]

    print(f"총 {len(pdb_ids)}개의 PDB ID 검증 시작... (CPU 12개 사용)")

    # 멀티프로세싱 실행
    with Pool(processes=12) as pool:
        results = pool.map(check_polypeptide_in_pdb, [(pdb_id, cif_base_dir) for pdb_id in pdb_ids])

    # 결과 분리
    valid_ids = [pdb_id for pdb_id, is_valid in results if is_valid]
    invalid_ids = [pdb_id for pdb_id, is_valid in results if not is_valid]

    print(f"\n✅ 유효한 단백질 구조: {len(valid_ids)}개")
    print(f"❌ 유효하지 않은 구조 (carbohydrate, RNA, DNA 등만 포함): {len(invalid_ids)}개")

    if invalid_ids:
        print("\n유효하지 않은 구조 목록:")
        for invalid in invalid_ids:
            print(invalid)

    return valid_ids, invalid_ids

# 사용 예시
if __name__ == "__main__":
    pdb_txt_path = "/home/psh/protein-frame-flow/parse_general/meta/only_protein_filtered.txt"
    cif_base_dir = "/public_data/BioMolDB_2024Oct21/cif"

    valid_ids, invalid_ids = validate_pdb_list(pdb_txt_path, cif_base_dir)