import warnings
from abnumber import Chain
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1
import pyrosetta
import rosetta

import os
import json 

def clean_pdb(pdb_file):
    with open(pdb_file, "r") as f:
        lines = f.readlines()

    with open(pdb_file, "w") as f:
        for l in lines:
            if "ATOM" in l:
                f.write(l)

def is_heavy(seq):
    chain = Chain(seq, scheme='chothia')

    return chain.is_heavy_chain()


def rechain_pdb(pdb_file):
    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure("_", pdb_file)

    for i, chain in enumerate(structure.get_chains()):
        if i == 2:
            break
        seq = seq1(''.join([residue.resname for residue in chain]))
        abnum_chain = Chain(seq, scheme='chothia')
        chain_id = "H" if abnum_chain.is_heavy_chain() else "L"
        try:
            chain.id = chain_id
        except ValueError:
            chain.id = chain_id + "_"
    for chain in structure.get_chains():
        if "_" in chain.id:
            chain.id = chain.id.replace("_", "")

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_file)


def renumber_pdb(
    in_pdb_file,
    out_pdb_file=None,
    scheme="chothia",
):
    """
    Renumber the pdb file.
    """
    if out_pdb_file is None:
        out_pdb_file = in_pdb_file

    clean_pdb(in_pdb_file)

    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure(
            "_",
            in_pdb_file,
        )

    for i, chain in enumerate(structure.get_chains()):
        if i == 2:
            break
        seq = seq1(''.join([residue.resname for residue in chain]))
        abnum_chain = Chain(seq, scheme=scheme)
        numbering = abnum_chain.positions.items()

        chain_res = list(chain.get_residues())
        if len(chain_res) != len(numbering):
            return False

        for pdb_r, (pos, aa) in zip(chain_res, numbering):
            if aa != seq1(pdb_r.get_resname()):
                raise Exception(f"Failed to renumber PDB file {in_pdb_file}")
            pos = str(pos)[1:]
            if not pos[-1].isnumeric():
                ins = pos[-1]
                pos = int(pos[:-1])
            else:
                pos = int(pos)
                ins = ' '

            pdb_r._id = (' ', pos, ins)

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb_file)

    return True

def truncate_seq(seq, scheme="chothia"):
    abnum_chain = Chain(seq, scheme=scheme)
    numbering = abnum_chain.positions.items()
    seq = "".join([r[1] for r in list(numbering)])

    return seq

def get_ab_metrics(pdb_file_1, pdb_file_2, output_file):
    # pdb_file_1: predicted
    # pdb_file_2: label

    # PyRosetta 초기화
    pyrosetta.init("-ignore_zero_occupancy false -check_cdr_chainbreaks false")

    # PDB 파일로부터 pose 객체 생성
    pose_1 = pyrosetta.pose_from_file(pdb_file_1)
    pose_2 = pyrosetta.pose_from_file(pdb_file_2)

    # AntibodyInfo 객체 생성
    try:
        pose_i1 = rosetta.protocols.antibody.AntibodyInfo(pose_1)
        pose_i2 = rosetta.protocols.antibody.AntibodyInfo(pose_2)
    except RuntimeError as e:
        print(f"Error creating AntibodyInfo: {e}")
        return False

    # CDR 백본 RMSD 계산
    results = rosetta.protocols.antibody.cdr_backbone_rmsds(
        pose_1,
        pose_2,
        pose_i1,
        pose_i2,
    )

    # 결과 레이블과 결과 딕셔너리 준비
    # results_labels = [
    #     'ocd', 'frh_rms', 'h1_rms', 'h2_rms', 'h3_rms', 'frl_rms', 'l1_rms',
    #     'l2_rms', 'l3_rms', 'interface_energy', 'label_interface_energy', 'sasa', 'label_sasa', 'total_Hbond_E', 'label_total_Hbond_E'
    # ]
    results_labels = [
        'ocd', 'frh_rms', 'h1_rms', 'h2_rms', 'h3_rms', 'frl_rms', 'l1_rms', 'l2_rms', 'l3_rms'
    ]
    results_dict = {}
    for i in range(9):
        results_dict[results_labels[i]] = results[i + 1]
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=4)

    print(f"Results saved to {output_file}")

    return results_dict