import os
import gzip
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from io import StringIO
import pandas as pd
from multiprocessing import Pool, cpu_count

def has_polypeptide_chain(cif_dict):
    types = cif_dict.get("_entity_poly.type", [])
    if isinstance(types, str):
        types = [types]
    for t in types:
        t_clean = t.strip().lower()
        if t_clean in ("polypeptide(l)", "polypeptide(d)"):
            return True
    return False

def is_polypeptide_present(cif_gz_path):
    try:
        with gzip.open(cif_gz_path, 'rt') as f:
            cif_data = f.read()
        cif_dict = MMCIF2Dict(StringIO(cif_data))

        if has_polypeptide_chain(cif_dict):
            print(f"{os.path.basename(cif_gz_path)} is selected.")
            return os.path.basename(cif_gz_path).replace(".cif.gz", "")
        return None
    except Exception as e:
        print(f"Error parsing {cif_gz_path}: {e}")
        return None

def process_one_cif(cif_ID):
    cif_path = (
        "/public_data/BioMolDB_2024Oct21/cif/"
        + cif_ID[1:3]
        + "/"
        + cif_ID
        + ".cif.gz"
    )
    return is_polypeptide_present(cif_path)


if __name__ == "__main__":
    meta_csv = pd.read_csv('/home/psh/protein-frame-flow/parse_general/meta/resolution_2.0.csv', header=None)
    cif_list = meta_csv[0].to_list()

    # 병렬 처리 시작
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_one_cif, cif_list)

    # 결과 필터링 및 저장
    all_peptide_list = [r for r in results if r is not None]

    with open('/home/psh/protein-frame-flow/parse_general/utils/only_protein_2.txt', 'w') as f:
        for item in all_peptide_list:
            f.write(item + '\n')
