import os
import gzip
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from io import StringIO
import pandas as pd
from multiprocessing import Pool, cpu_count

def contains_nonstandard_amino_acid(cif_gz_path):
    non_standard_amino_acids = {
        "MSE", "SEP", "TPO", "PTR", "CSO", "HYP", "MLY",
        "LLP", "PCA", "KCX", "CIR", "ACE", "NH2", "PYL", "SEC"
    }
    try:
        with gzip.open(cif_gz_path, 'rt') as f:
            cif_data = f.read()
        cif_dict = MMCIF2Dict(StringIO(cif_data))

        comp_ids = cif_dict.get("_atom_site.label_comp_id", [])
        if isinstance(comp_ids, str):
            comp_ids = [comp_ids]

        unique_comp_ids = set(comp_ids)

        if unique_comp_ids.intersection(non_standard_amino_acids):
            return True
        return False
    except Exception as e:
        print(f"Error processing {cif_gz_path}: {e}")
        return False

def process_one_cif(cif_ID):
    cif_path = (
        "/public_data/BioMolDB_2024Oct21/cif/"
        + cif_ID[1:3]
        + "/"
        + cif_ID
        + ".cif.gz"
    )
    result = contains_nonstandard_amino_acid(cif_path)
    print(f"{cif_ID} has non-standard amino acid: {result}")
    return cif_ID, result

if __name__ == "__main__":
    with open('/home/psh/protein-frame-flow/parse_general/meta/only_protein_filtered.txt', 'r') as f:
        cif_list = [line.strip() for line in f if line.strip()]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_one_cif, cif_list)

    # non-standard 아미노산 포함된 pdb만 필터링
    nonstandard_pdbs = [pdb_id for pdb_id, has_nonstd in results if has_nonstd]

    # 파일로 저장
    out_path = '/home/psh/protein-frame-flow/parse_general/meta/pdbs_with_nonstandard_amino_acid.txt'
    with open(out_path, 'w') as f:
        for pdb_id in nonstandard_pdbs:
            f.write(pdb_id + '\n')

    print(f"🌟 Non-standard 아미노산 포함된 PDB ID 총 {len(nonstandard_pdbs)}개 저장 완료: {out_path}")
