
import argparse
import sys
import os

import warnings
warnings.filterwarnings("ignore")

# 현재 경로 및 부모 디렉터리 경로 설정
sys.path.append('/home/psh/protein-frame-flow')

# from local_igfold.test.test import *
from data.ab_metrics import *
from Bio import PDB

parser = argparse.ArgumentParser(
    description='Inference processing script.')
parser.add_argument(
    '--pred_dir',
    help='Path to directory with predicted PDB files.',
    type=str)
parser.add_argument(
    '--label_dir',
    help='Path to directory with label PDB files.',
    type=str,
    default='preprocessed')
parser.add_argument(
    '--debug',
    help='Turn on for debugging.',
    action='store_true')
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true')
parser.add_argument(
    '--config',
    help='config file path',
    type=str,
    default='/home/psh/protein-frame-flow/configs/datasets.yaml'
)
parser.add_argument(
    '--num_processes',
    help='Number of processes.',
    type=int,
    default=50)

def extract_first_two_chains(input_pdb, output_pdb):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)

    chains = list(structure.get_chains())
    
    if len(chains) == 1:
        # Nanobody
        chains[0].id = 'H'
    elif len(chains) >= 2:
        # 첫 두 체인만 선택
        first_chain, second_chain = chains[:2]

        # 이미 H/L이면 그대로
        if first_chain.id != 'H':
            first_chain.id = 'H'
        if second_chain.id != 'L':
            second_chain.id = 'L'

        # 기존 모델에서 chain 제거 후 다시 추가
        for model in structure:
            existing_chains = list(model)
            for c in existing_chains:
                model.detach_child(c.id)
            model.add(first_chain)
            model.add(second_chain)
    else:
        raise ValueError(f"PDB 파일에 적절한 체인이 없습니다: {input_pdb}")

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)


def count_chains_in_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    chains = set()
    for model in structure:
        for chain in model:
            chains.add(chain.id)
    
    return len(chains)

def main(args):
    predicted_dir = args.pred_dir
    label_dir = args.label_dir

    # 예측 파일에 대해 반복 처리
    for i, filename in enumerate(os.listdir(predicted_dir)):
        pred_subdir = os.path.join(predicted_dir, filename)
        if not os.path.isdir(pred_subdir):
            continue

        try:
            print(f'start to measure {filename}')
            
            for sample in os.listdir(pred_subdir):
                if 'sample' not in sample:
                    continue                
                predicted_filepath = os.path.join(pred_subdir, sample, 'sample_1.pdb')
                only_ab_filepath = os.path.join(pred_subdir, sample, 'only_ab.pdb')
                output_file = os.path.join(pred_subdir, sample, 'cdr_rmsd.json')

                if os.path.exists(output_file):
                    continue
                a = renumber_pdb(predicted_filepath)
                extract_first_two_chains(predicted_filepath, only_ab_filepath)

                if '.pkl' in filename:
                    filename = filename.replace('.pkl', '')
                if '#' in filename:
                    filename = filename.replace('#', "NA")
                label_filepath = os.path.join(label_dir, filename + '.pdb')
            
                # PDB 파일 재체인 및 리넘버링
                # rechain_pdb(predicted_filepath)

                # rechain_pdb(label_filepath)
                # b = renumber_pdb(label_filepath)

                # 리넘버링 실패 시 건너뛰기
                if a == False:
                    print(f"Skipping {filename} due to renumbering issue.")
                    continue
                
                # 메트릭 계산 및 결과 저장
                results = get_ab_metrics(only_ab_filepath, label_filepath, output_file)

        except Exception as e:
            # 오류가 발생하면 무시하고 다음 파일로 진행
            print(f"Error processing {filename}: {e}")
            continue


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)