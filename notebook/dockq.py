import os
import subprocess
import re
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_chain_order(pdb_path):
    """PDB 파일에서 Chain ID가 등장하는 순서대로 리스트를 추출합니다."""
    chains = []
    seen = set()
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                # ATOM 또는 HETATM 레코드의 21번째 인덱스(컬럼 22)에 Chain ID가 있습니다.
                if line.startswith(('ATOM', 'HETATM')):
                    chain_id = line[21]
                    if chain_id not in seen:
                        seen.add(chain_id)
                        chains.append(chain_id)
    except Exception as e:
        print(f"Error reading chains from {pdb_path}: {e}")
    return chains

def create_remapped_model(native_pdb, model_pdb):
    """
    Model PDB의 Chain ID를 Native PDB의 Chain ID 순서에 맞춰 변경한
    임시 파일의 경로를 반환합니다.
    """
    native_chains = get_chain_order(native_pdb)
    model_chains = get_chain_order(model_pdb)

    # Chain 개수가 다를 경우 경고 (하지만 가능한 만큼 매핑 진행)
    if len(native_chains) != len(model_chains):
        # 로깅이 필요하면 여기에 추가
        pass

    # 매핑 딕셔너리 생성 (Model Chain -> Native Chain)
    # 예: Model ['H', 'L'] -> Native ['A', 'B']  => {'H': 'A', 'L': 'B'}
    chain_map = {}
    for i, m_chain in enumerate(model_chains):
        if i < len(native_chains):
            chain_map[m_chain] = native_chains[i]
        else:
            # Native보다 Model 체인이 더 많을 경우 그대로 유지
            chain_map[m_chain] = m_chain

    # 임시 파일 생성 (delete=False로 닫은 후 subprocess에서 사용)
    temp_fd, temp_path = tempfile.mkstemp(suffix='.pdb', text=True)
    
    with os.fdopen(temp_fd, 'w') as fout, open(model_pdb, 'r') as fin:
        for line in fin:
            if line.startswith(('ATOM', 'HETATM', 'TER')) and len(line) > 21:
                old_chain = line[21]
                # 매핑된 Chain ID가 있으면 교체
                if old_chain in chain_map:
                    new_chain = chain_map[old_chain]
                    # 문자열은 불변이므로 슬라이싱으로 교체
                    new_line = line[:21] + new_chain + line[22:]
                    fout.write(new_line)
                else:
                    fout.write(line)
            else:
                fout.write(line)
                
    return temp_path

def run_dockq(native_pdb, model_pdb, dockq_path="./DockQ.py"):
    # 1. Chain ID 매핑을 위해 임시 Model 파일 생성
    #    (Chain ID가 이미 같더라도 로직 통일성을 위해 수행하거나, 
    #     최적화를 위해 chain 검사 후 다를 때만 수행할 수도 있습니다. 
    #     여기서는 안전하게 항상 매핑을 시도합니다.)
    temp_model_path = create_remapped_model(native_pdb, model_pdb)

    try:
        cmd = ["python", dockq_path, temp_model_path, native_pdb]
        
        # DockQ 실행
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return {"error": result.stderr.strip()}

        output = result.stdout
        scores = {}
        # 정규표현식으로 점수 파싱
        for key in ["fnat", "iRMSD", "LRMSD", "DockQ"]:
            match = re.search(rf"{key}:\s*([\d\.]+)", output)
            if match:
                scores[key] = float(match.group(1))
        return scores

    finally:
        # 2. 작업이 끝나면 임시 파일 삭제
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

def process_one(native_pdb, model_sample_path, dockq_path, pdb_id, sample):
    # 출력 파일 경로 미리 계산
    output_json = model_sample_path.replace('.pdb', '.json') # 파일명만 추출하여 경로 결합

    # 이미 처리된 파일이 있다면 건너뛰기 (선택 사항)
    # if os.path.exists(output_json):
    #     print(f"Skipping {pdb_id}_{sample}, already exists.")
    #     return (pdb_id, sample, {"status": "skipped"})

    print(f"Processing: {pdb_id}_{sample}")
    
    score = run_dockq(native_pdb, model_sample_path, dockq_path)
    
    # 디버깅: JSON 경로 확인
    # print("output_json", output_json)
    
    with open(output_json, "w") as f:
        json.dump(score, f, indent=2)
        
    return (pdb_id, sample, score)

def batch_run(native_pdb_dir, pred_dir, dockq_path, max_workers=30):
    futures = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for pdb_sample_id in os.listdir(pred_dir):
            if 'config' in pdb_sample_id:
                continue
            pdb_id = "_".join(pdb_sample_id.split('_')[:4])
            native_pdb = os.path.join(native_pdb_dir, pdb_id + '.pdb')

            pred_sample_dir = os.path.join(pred_dir, pdb_sample_id)
            
            for sample in os.listdir(pred_sample_dir):
                model_sample_path = os.path.join(pred_sample_dir, sample, 'sample_1.pdb')
                
                futures.append(
                    executor.submit(process_one, native_pdb, model_sample_path, dockq_path, pdb_id, sample)
                )

        for future in as_completed(futures):
            try:
                pdb_id, sample, score = future.result()
                # print(f"Finished {pdb_id}_{sample}: {score}") # 결과가 많으면 출력 줄이기
            except Exception as exc:
                print(f"Generated an exception: {exc}")

if __name__ == "__main__":
    native_pdb_dir = '/home/psh/benchmark_after210930/pdb'             
    pred_dir = '/home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.3.2.2_stage_2_ori_perturb_stage3/2025-12-06_22-56-57/epoch=52-step=75949_copy/kkh_wo_perturb'
    dockq_path = "/home/psh/DockQ/src/DockQ/DockQ.py"             

    batch_run(native_pdb_dir, pred_dir, dockq_path, max_workers=20)