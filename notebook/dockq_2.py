import os
import subprocess
import concurrent.futures
import multiprocessing
import glob
from tqdm import tqdm # 진행 상황을 보기 위해 tqdm 라이브러리를 사용합니다 (선택 사항)

# --- 설정 입력 ---
# 모델들이 들어있는 최상위 폴더
models_folder = "/home/psh/BioBetter/kkh_benchmark/chothia"
# 정답(Native) 파일 경로 폴더
native_folder = "/home/psh/benchmark_after210930/pdb" 

# --- 병렬 처리 설정 ---
# 사용할 CPU 코어 수 설정. 
# None으로 설정하면 사용 가능한 모든 코어를 사용합니다. (최대 성능)
# 만약 코어 수를 제한하고 싶다면 숫자를 입력하세요 (예: max_workers = 8)
MAX_WORKERS = 20
print(f"INFO: Using {MAX_WORKERS} CPU cores for parallel processing.")

# ----------------------------------------------------

def run_dockq_task(task_args):
    """
    하나의 DockQ 작업을 실행하는 함수 (각 CPU 코어에서 독립적으로 실행됨)
    """
    model_path, native_file, json_output_path = task_args
    
    # Native 파일 존재 여부 확인 (병렬 처리 중 에러 방지)
    if not os.path.exists(native_file):
        return f"WARNING: Native file not found: {native_file}. Skipping {model_path}."

    # Model 파일 존재 여부 확인
    if not os.path.exists(model_path):
        return f"WARNING: Model file not found: {model_path}. Skipping."
        
    cmd = [
        "DockQ",
        model_path,
        native_file,
        "--json",
        json_output_path
    ]
    
    try:
        # 명령어 실행 (출력을 캡처하여 터미널을 깔끔하게 유지)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return f"SUCCESS: Processed {model_path}"
    except subprocess.CalledProcessError as e:
        # DockQ 실행 중 오류 발생 시
        return f"ERROR: DockQ failed for {model_path}. Stderr: {e.stderr.strip()}"
    except FileNotFoundError:
        # 'DockQ' 명령어를 찾지 못했거나 경로 문제 발생 시
        return f"FATAL ERROR: 'DockQ' command not found or path error for {model_path}"
    except Exception as e:
        return f"UNEXPECTED ERROR: Failed to process {model_path}. Error: {e}"

def collect_tasks():
    """
    모든 DockQ 작업을 리스트로 수집하는 함수
    """
    tasks = []
    
    # models_folder 안의 모든 항목 순회
    for pdb_id in os.listdir(models_folder):
        
        # 'config' 포함 항목 건너뛰기
        if 'config' in pdb_id:
            continue
        
        pdb_dir = os.path.join(models_folder, pdb_id)
        native_file = os.path.join(native_folder, pdb_id + '.pdb')

        for sample_id in os.listdir(pdb_dir):
            model_path = os.path.join(pdb_dir, sample_id)
            json_output_path = os.path.join(pdb_dir, f"{sample_id}_dockq.json")
            tasks.append((model_path, native_file, json_output_path))

    return tasks

if __name__ == '__main__':
    all_tasks = collect_tasks()
    total_tasks = len(all_tasks)
    
    print(f"INFO: Collected {total_tasks} DockQ tasks to run.")

    if total_tasks == 0:
        print("모든 작업이 완료되었습니다. (실행할 작업이 없습니다.)")
    else:
        # ProcessPoolExecutor를 사용하여 병렬 실행
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            
            # map 함수를 사용하여 모든 작업을 executor에 할당하고 결과를 받음
            # tqdm을 사용하여 진행 상황 막대 출력
            results = list(tqdm(executor.map(run_dockq_task, all_tasks), 
                                total=total_tasks, 
                                desc="Running DockQ Tasks"))
        
        # 결과 출력 및 오류 보고
        print("\n--- Summary of Results ---")
        error_count = 0
        for result in results:
            if result.startswith("ERROR") or result.startswith("WARNING"):
                print(result)
                error_count += 1

        print(f"\nCompleted {total_tasks} tasks. ({error_count} errors/warnings reported)")
        print("모든 작업이 완료되었습니다.")