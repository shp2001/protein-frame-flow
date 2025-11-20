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
    PDB 파일의 항체 가변 영역(V-domain, VHH 포함)을 renumbering하고
    불변 영역(C-domain)은 기존 번호를 유지합니다.
    Antigen 등 비-항체 체인은 건너뜁니다.
    """
    if out_pdb_file is None:
        out_pdb_file = in_pdb_file

    parser = PDBParser(QUIET=True) # 경고 메시지 최소화
    structure = parser.get_structure("_", in_pdb_file)

    # 모든 체인에 대해 순회 (기존의 i==2 break 제거)
    for chain in structure.get_chains():
        
        # 1. 비표준 잔기(HOH, 리간드 등)를 제외한 표준 아미노산 잔기 리스트와 서열 생성
        std_residues = []
        std_resnames = []
        
        for r in chain.get_residues():
            # PDB 표준 아미노산인지 확인 (Bio.PDB.Polypeptide.is_aa 등을 쓸 수도 있으나 seq1 활용)
            resname = r.get_resname()
            # seq1은 표준 아미노산이 아니면 '' 혹은 에러가 날 수 있음, 보통 3글자->1글자 변환
            # 안전하게 처리하기 위해 try-except 혹은 dict check 사용 권장되나,
            # 기존 로직을 존중하여 KeyError/ValueError 처리
            try:
                # 리간드나 물(HOH) 등은 seq1 변환 시 예외가 발생하거나 비표준 문자가 될 수 있음
                if r.id[0] != ' ': # HETATM 제외 (물, 리간드 등)
                    continue
                
                aa = seq1(resname)
                # seq1('UNK') 등은 'X'를 반환할 수 있음. 필요한 경우 필터링
                std_residues.append(r)
                std_resnames.append(resname)
            except (KeyError, ValueError):
                continue
        
        # 서열이 너무 짧으면 항체가 아닐 확률 높음
        if not std_resnames:
            continue

        seq = seq1(''.join(std_resnames))

        # 2. abnumber를 사용해 V-domain 서열 번호 매기기
        # Nanobody(VHH)도 Heavy chain으로 인식되어 처리됨
        # Antigen이나 비-항체 서열은 ChainParseError 발생하므로 예외 처리
        try:
            abnum_chain = AbChain(seq, scheme=scheme)
        except Exception:
            # abnumber가 항체 서열로 인식하지 못함 -> renumbering 건너뛰기 (Antigen 등)
            # print(f"Info: Chain {chain.id} is not an antibody variable domain. Skipping.")
            continue

        # 3. V-domain 번호 리스트 가져오기
        # abnumber는 가변 영역만 인식하므로, 전체 서열 중 앞부분(V-domain)에 해당하는 번호만 반환
        numbering_list = list(abnum_chain.positions.items())
        vd_len = len(numbering_list)

        if vd_len == 0:
            continue

        # 4. PDB 잔기 리스트에서 V-domain에 해당하는 부분 매핑
        # 주의: abnumber는 입력 서열의 '앞부분'에서 V-domain을 찾았다고 가정합니다.
        # (일반적인 항체 PDB 구조상 V-domain이 N-term에 위치함)
        vd_residues_pdb = std_residues[:vd_len]

        # 길이 검증
        if len(vd_residues_pdb) != len(numbering_list):
            print(f"[Warning] Chain {chain.id}: Length mismatch between PDB residues ({len(vd_residues_pdb)}) and abnumber result ({len(numbering_list)}). Skipping.")
            continue

        # 5. V-domain 부분만 PDB ID (Residue Number) 변경
        for pdb_r, (pos, aa) in zip(vd_residues_pdb, numbering_list):
            # pos 예시: 'H100A', 'L24' 등 (Chain type 포함될 수 있음 -> abnumber 버전에 따라 다름)
            # abnumber의 positions 키값은 보통 string이거나 별도 객체임. 
            # 문자열로 변환 후 파싱
            
            pos_str = str(pos) # 예: "100", "100A", "H100A" (설정에 따라 다름)
            
            # abnumber Chain 객체의 positions 키는 보통 숫자+삽입코드 형태지만,
            # Chain type(H/L)이 붙어있을 수 있으니 제거 (맨 앞 글자가 알파벳이고 뒤가 숫자면)
            if pos_str[0].isalpha() and len(pos_str) > 1 and pos_str[1].isdigit():
                 pos_str = pos_str[1:] # H100 -> 100
            
            # 숫자와 삽입코드(Insertion Code) 분리
            if not pos_str[-1].isnumeric():
                ins = pos_str[-1]       # 삽입 코드 (예: 'A')
                res_num = int(pos_str[:-1]) # 잔기 번호 (예: 100)
            else:
                ins = ' '               # 삽입 코드 없음
                res_num = int(pos_str)
            
            # PDB 잔기 ID 업데이트: (Hetero flag, Sequence identifier, Insertion code)
            # Hetero flag는 표준 아미노산이므로 ' '
            pdb_r.id = (' ', res_num, ins)

    # 저장
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb_file)

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
    pyrosetta.init(
        "-ignore_unrecognized_res "
        "-ignore_waters "
        "-ignore_zero_occupancy true "
        "-mute all"
        "-check_cdr_chainbreaks false"
    )
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