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


def renumber_pdb_wt_constant(
    in_pdb_file,
    out_pdb_file=None,
    scheme="chothia",
):
    """
    PDB 파일의 항체 가변 영역(V-domain)을 renumbering하고
    불변 영역(C-domain)은 기존 번호를 유지합니다.
    """
    if out_pdb_file is None:
        out_pdb_file = in_pdb_file

    # clean_pdb(in_pdb_file) # 해당 함수가 정의되어 있지 않아 주석 처리
    
    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure(
            "_",
            in_pdb_file,
        )

    for i, chain in enumerate(structure.get_chains()):
        if i == 2:  # Heavy/Light chain (최대 2개)만 처리
            break

        # 1. 비표준 잔기(HOH, 리간드 등)를 제외한 표준 아미노산 잔기 리스트와 서열 생성
        std_residues = []
        std_resnames = []
        for r in chain.get_residues():
            try:
                # seq1 변환이 가능한 표준 아미노산인지 확인
                aa = seq1(r.get_resname())
                std_residues.append(r)
                std_resnames.append(r.get_resname())
            except KeyError:
                continue  # HOH, 리간드 등 비표준 잔기 건너뛰기
        
            
        seq = seq1(''.join(std_resnames))

        # 2. abnumber를 사용해 V-domain 서열 번호 매기기
        abnum_chain = Chain(seq, scheme=scheme)

        # 3. V-domain 번호 리스트와 V-domain 시작 인덱스 가져오기
        numbering_list = list(abnum_chain.positions.items())  # 순서가 보장된 리스트
        vd_len = len(numbering_list)

        # 4. PDB 잔기 리스트(std_residues)에서 V-domain에 해당하는 부분만 슬라이싱
        vd_residues_pdb = std_residues[:vd_len]

        # 5. 길이 재확인 (이제 길이가 같아야 함)
        if len(vd_residues_pdb) != len(numbering_list):
            print(f"오류: Chain {chain.id} 슬라이싱 후에도 길이가 불일치합니다. 로직 확인 필요.")
            continue

        print(f"Chain {chain.id}: V-domain {len(vd_residues_pdb)}개 잔기를 renumbering합니다...")
        # 6. V-domain 부분만 PDB ID (번호) 변경
        for pdb_r, (pos, aa) in zip(vd_residues_pdb, numbering_list):
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
    print(f"Renumbering 완료. 파일 저장: {out_pdb_file}")

    return True

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
            print("chain_res", chain_res)
            print("numbering", numbering)
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