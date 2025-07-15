import os
import csv
import traceback
from multiprocessing import Pool

# Clash 계산 함수들
from Bio.PDB import PDBParser, kdtrees
from Bio import PDB
import numpy as np
import torch
import data.residue_constants as rc
from openfold.utils.loss import within_residue_violations

# Clash 관련 파라미터
atom_radii = {
    "C": 1.70, 
    "N": 1.55, 
    "O": 1.52,
    "S": 1.80,
    "F": 1.47, 
    "P": 1.80, 
    "CL": 1.75, 
    "MG": 1.73,
}

def count_clashes(pdb_path, clash_cutoff=0.63):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    clash_cutoffs = {
        f"{i}_{j}": (clash_cutoff * (atom_radii[i] + atom_radii[j]))
        for i in atom_radii for j in atom_radii
    }
    atoms = [x for x in structure.get_atoms() if x.element in atom_radii]
    coords = np.array([a.coord for a in atoms], dtype="d")
    kdt = kdtrees.KDTree(coords)
    clashes = []

    for atom_1 in atoms:
        kdt_search = kdt.search(np.array(atom_1.coord, dtype="d"), max(clash_cutoffs.values()))
        potential_clash = [(a.index, a.radius) for a in kdt_search]
        for ix, atom_distance in potential_clash:
            atom_2 = atoms[ix]
            if atom_1.parent.id == atom_2.parent.id:
                continue
            if (atom_2.name == "C" and atom_1.name == "N") or (atom_2.name == "N" and atom_1.name == "C"):
                continue
            if (atom_2.name == "SG" and atom_1.name == "SG") and atom_distance > 1.88:
                continue
            key = f"{atom_2.element}_{atom_1.element}"
            if key in clash_cutoffs and atom_distance < clash_cutoffs[key]:
                clashes.append((atom_1, atom_2))
    return len(clashes) // 2

def compute_within_clash_loss_from_pdb(
    pdb_path,
    clash_overlap_tolerance=1.5,
    violation_tolerance_factor=15,
):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]
    atom14_pred_positions = []
    atom14_atom_exists = []
    aatype = []
    pdb_residue_ids = []

    for res in model.get_residues():
        resname = res.get_resname()
        if resname not in rc.restype_3to1:
            continue
        one_letter = rc.restype_3to1[resname]
        if one_letter not in rc.restype_order:
            continue
        res_index = rc.restype_order[one_letter]
        residue_id = ''.join(str(x).strip() for x in res.get_id())
        pdb_residue_ids.append(residue_id)

        aatype.append(res_index)
        pos = np.zeros((14, 3), dtype=np.float32)
        exists = np.zeros(14, dtype=np.float32)
        mapping = rc.restype_name_to_atom14_names[resname]
        for atom14_idx, atom_name in enumerate(mapping):
            if atom_name in res:
                pos[atom14_idx] = res[atom_name].get_coord()
                exists[atom14_idx] = 1.0

        atom14_pred_positions.append(pos)
        atom14_atom_exists.append(exists)

    pos_tensor = torch.tensor([atom14_pred_positions])
    exist_tensor = torch.tensor([atom14_atom_exists])
    aatype_tensor = torch.tensor([aatype])
    bounds = rc.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )
    lower_bound = pos_tensor.new_tensor(bounds["lower_bound"])[aatype_tensor]
    upper_bound = pos_tensor.new_tensor(bounds["upper_bound"])[aatype_tensor]
    clash_info = within_residue_violations(
        pos_tensor, exist_tensor, lower_bound, upper_bound
    )
    per_atom_violations = clash_info["per_atom_violations"]
    per_residue_violation = torch.any(per_atom_violations[0] > 0, dim=1)
    violated_residue_indices = torch.nonzero(per_residue_violation).squeeze().tolist()
    if isinstance(violated_residue_indices, int):
        violated_residue_indices = [violated_residue_indices]
    violated_residue_pdb_ids = [pdb_residue_ids[i] for i in violated_residue_indices]
    num_violations = torch.sum(per_atom_violations > 0).item() // 2
    return num_violations, violated_residue_pdb_ids

# 개별 샘플 처리 함수 (멀티프로세싱 대상)
def process_sample(args):
    pdb_id, sample, sample_path = args
    print(f"{pdb_id}_{sample}")
    try:
        num_inter_clash = count_clashes(sample_path)
        num_intra_clash, intra_clash_resids = compute_within_clash_loss_from_pdb(sample_path)
        return {
            'pdb': pdb_id,
            'sample': sample,
            'num_inter_clash': num_inter_clash,
            'num_intra_clash': num_intra_clash,
            'intra_clash_resids': ";".join(str(r) for r in intra_clash_resids)
        }
    except Exception as e:
        print(f"[ERROR] {pdb_id}/{sample}: {e}")
        traceback.print_exc()
        return None

# Main 함수
def main():
    inf_dir = '/home/psh/protein-frame-flow/inference_outputs/fm_aa_ref_pos/2025-05-26_23-05-11/epoch=99-step=141600/run_2025-05-29_12-12-24'
    output_csv = '/home/psh/protein-frame-flow/experiments/clash_info/fm_aa_ref_pos.csv'

    task_list = []
    for pdb_id in os.listdir(inf_dir):
        pdb_path = os.path.join(inf_dir, pdb_id)
        if not os.path.isdir(pdb_path):
            continue
        for sample in os.listdir(pdb_path):
            sample_pdb_path = os.path.join(pdb_path, sample, 'sample_1.pdb')
            if os.path.exists(sample_pdb_path):
                task_list.append((pdb_id, sample, sample_pdb_path))

    print(f"총 처리할 샘플 수: {len(task_list)}")

    with Pool(processes=8) as pool:
        results = pool.map(process_sample, task_list)

    # 결과 필터링 및 저장
    valid_rows = [r for r in results if r is not None]

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pdb', 'sample', 'num_inter_clash', 'num_intra_clash', 'intra_clash_resids'])
        writer.writeheader()
        writer.writerows(valid_rows)

    print(f"✅ CSV 저장 완료: {output_csv}")

if __name__ == "__main__":
    main()
