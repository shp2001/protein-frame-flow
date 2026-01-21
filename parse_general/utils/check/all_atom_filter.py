import os
import gzip
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def get_protein_atom_and_residue_ratio(args):
    pdb_id, base_dir = args
    cif_path = os.path.join(base_dir, pdb_id[1:3], pdb_id + ".cif.gz")
    protein_residues = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU",
        "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
        "PRO", "SER", "THR", "TRP", "TYR", "VAL"
    }
    try:
        with gzip.open(cif_path, 'rt') as f:
            cif_data = f.read()
        cif_dict = MMCIF2Dict(StringIO(cif_data))

        atom_ids = cif_dict.get("_atom_site.label_atom_id", [])
        seq_ids = cif_dict.get("_atom_site.label_seq_id", [])
        chain_ids = cif_dict.get("_atom_site.auth_asym_id", [])
        res_names = cif_dict.get("_atom_site.label_comp_id", [])

        if isinstance(atom_ids, str): atom_ids = [atom_ids]
        if isinstance(seq_ids, str): seq_ids = [seq_ids]
        if isinstance(chain_ids, str): chain_ids = [chain_ids]
        if isinstance(res_names, str): res_names = [res_names]

        # protein residue에 해당하는 원자만 필터링
        filtered_atoms = [
            (atom, seq, chain, res)
            for atom, seq, chain, res in zip(atom_ids, seq_ids, chain_ids, res_names)
            if res in protein_residues
        ]

        if not filtered_atoms:
            return None

        atom_count = len(filtered_atoms)
        residues = set((seq, chain, res) for _, seq, chain, res in filtered_atoms)
        residue_count = len(residues)

        if residue_count == 0:
            return None

        ratio = atom_count / residue_count
        print(f"{pdb_id} protein atom/residue ratio: {ratio:.2f}")
        return ratio
    except Exception as e:
        print(f"{pdb_id} 처리 실패: {e}")
        return None

def main():
    pdb_txt_path = "/home/psh/protein-frame-flow/parse_general/meta/only_protein_filtered.txt"
    cif_base_dir = "/public_data/BioMolDB_2024Oct21/cif"

    with open(pdb_txt_path, 'r') as f:
        pdb_ids = [line.strip() for line in f if line.strip()]

    # 병렬 처리
    print(f"📦 총 {len(pdb_ids)}개 PDB ID 처리 중... (CPU 12개 사용)")
    with Pool(processes=12) as pool:
        ratios = list(tqdm(pool.imap(get_protein_atom_and_residue_ratio, [(pdb_id, cif_base_dir) for pdb_id in pdb_ids]), total=len(pdb_ids)))

    # None 제거
    ratios = [r for r in ratios if r is not None]
    result_list = [
        {"pdb": pdb_id, "atom_residue_ratio": ratio}
        for pdb_id, ratio in zip(pdb_ids, ratios)
        if ratio is not None
    ]

    # DataFrame으로 변환 후 CSV 저장
    df = pd.DataFrame(result_list)
    df.to_csv("/home/psh/protein-frame-flow/parse_general/meta/atom_residue_ratio.csv", index=False)
    print(f"✅ CSV 저장 완료: protein_atom_residue_ratio.csv")

if __name__ == "__main__":
    main()
