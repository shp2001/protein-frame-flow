import warnings
from abnumber import Chain
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1
# import pyrosetta
# import rosetta

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


from Bio.PDB import PDBParser, PDBIO, Chain as PDBChain
from Bio.SeqUtils import seq1
from abnumber import Chain, ChainParseError  # 에러 처리를 위해 import
import warnings

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

    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure("_", in_pdb_file)

    for i, chain in enumerate(structure.get_chains()):
        if i == 2:  # Heavy/Light chain (최대 2개)만 처리
            break

        # 1. 표준 아미노산 잔기 리스트와 서열 생성
        std_residues = []
        std_resnames = []
        for r in chain.get_residues():
            try:
                # seq1 변환이 가능한 표준 아미노산인지 확인
                aa = seq1(r.get_resname())
                std_residues.append(r)
                std_resnames.append(r.get_resname())
            except KeyError:
                continue
        
        full_seq = seq1(''.join(std_resnames))

        # [수정 1] 가변 영역 인식을 위해 서열 앞부분(150aa)만 잘라서 사용
        # C-domain이 너무 길게 붙어있으면 abnumber가 인식을 실패함
        seq_for_numbering = full_seq[:150] if len(full_seq) > 150 else full_seq

        try:
            # 2. abnumber를 사용해 V-domain 서열 번호 매기기
            abnum_chain = Chain(seq_for_numbering, scheme=scheme)
        except ChainParseError:
            print(f"Warning: Chain {chain.id}에서 V-domain을 인식하지 못했습니다. (Skip)")
            continue

        # 3. V-domain 번호 리스트 가져오기
        numbering_list = list(abnum_chain.positions.items())
        vd_len = len(numbering_list)

        # 4. PDB 잔기 리스트에서 V-domain 부분 슬라이싱
        vd_residues_pdb = std_residues[:vd_len]

        if len(vd_residues_pdb) != len(numbering_list):
            print(f"오류: Chain {chain.id} 길이 불일치.")
            continue

        print(f"Chain {chain.id}: V-domain {len(vd_residues_pdb)}개 잔기를 renumbering합니다...")
        
        # 5. Renumbering 수행
        for pdb_r, (pos, aa_abnum) in zip(vd_residues_pdb, numbering_list):
            # [수정 2] 안전 장치: PDB의 아미노산과 abnumber가 인식한 아미노산이 같은지 확인
            pdb_aa = seq1(pdb_r.get_resname())
            if pdb_aa != aa_abnum:
                print(f"Warning: 서열 불일치 감지 (PDB: {pdb_aa} vs AbNum: {aa_abnum}). 정렬 확인이 필요합니다.")
                # 필요시 여기서 break 하거나 continue 할 수 있음

            pos_str = str(pos)[1:] # H100 -> 100
            
            # Insertion code 처리 (예: 100A)
            if not pos_str[-1].isnumeric():
                ins = pos_str[-1]
                res_num = int(pos_str[:-1])
            else:
                res_num = int(pos_str)
                ins = ' '

            # PDB ID 변경 (Hetero flag, Sequence identifier, Insertion code)
            pdb_r._id = (' ', res_num, ins)

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