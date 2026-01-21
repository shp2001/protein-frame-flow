import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
# import orjson  # pip install orjson (필수 권장)

# orjson이 없다면 아래 주석 해제하여 표준 json 사용 (속도는 떨어짐)
import json

MASK_INDEX_DIR = "/home/psh/data/general_v2/mask_index"
METADATA_PATH = "/home/psh/data/general_v2/meta/metadata.csv"

# 전역 변수로 선언 (각 프로세스에서 읽기 전용으로 사용)
shared_metadata = None

def load_metadata(metadata_path):
    """
    metadata.csv를 pdb_name -> num_chains dict로 로드
    """
    # engine='c'와 필요한 컬럼만 로드하여 속도 향상
    df = pd.read_csv(metadata_path, usecols=["pdb_name", "num_chains"], engine='c')
    
    if "pdb_name" not in df.columns or "num_chains" not in df.columns:
        raise ValueError("metadata.csv에 pdb_name 또는 num_chains 컬럼이 없습니다.")

    return dict(zip(df["pdb_name"], df["num_chains"]))

def init_worker(metadata_dict):
    """
    워커 프로세스 초기화: 메타데이터를 전역 변수에 할당
    """
    global shared_metadata
    shared_metadata = metadata_dict

def check_single_file(json_path):
    """
    단일 json 파일 검증
    """
    # 전역 변수 사용 (인자로 받지 않음 -> 오버헤드 제거)
    global shared_metadata
    
    fname = os.path.basename(json_path)
    pdb_name = fname[:-5]  # .json 제거

    # 1. Memory Lookup First (Disk I/O 전에 검사)
    # metadata에 pdb_name이 없으면 즉시 반환
    if pdb_name not in shared_metadata:
        return json_path

    expected_num_chains = shared_metadata[pdb_name]


    # 2. Fast I/O & Parsing
    # with open(json_path, "rb") as f: # orjson은 binary 모드로 읽어야 함
    #     # orjson.loads는 bytes를 입력받아 매우 빠르게 파싱
    #     data = orjson.loads(f.read())
        
        # 표준 json 사용 시:
    with open(json_path, "r") as f:
        data = json.load(f)
            


    # 3. Validation
    if not isinstance(data, dict):
        return json_path

    # 키 개수 비교
    if len(data) != expected_num_chains:
        return json_path

    return None

def collect_json_files(root_dir):
    """
    os.scandir을 사용하여 파일 탐색 속도 소폭 개선
    """
    json_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))
    return json_files

def main():
    print("메타데이터 로딩 중...")
    metadata_dict = load_metadata(METADATA_PATH)
    
    print("파일 목록 수집 중...")
    json_files = collect_json_files(MASK_INDEX_DIR)
    total_files = len(json_files)
    print(f"총 json 파일 수: {total_files}")

    bad_files = []

    # 프로세스 개수 설정 (CPU 코어 수 or 30)
    num_procs = min(30, cpu_count())

    # chunksize 계산: 전체 작업 수를 프로세스 수로 나눈 것의 일부 (통신 빈도 줄임)
    chunk_size = max(1, total_files // (num_procs * 4))

    print(f"검증 시작 (Processes: {num_procs}, Chunksize: {chunk_size})...")
    
    # initializer를 통해 metadata_dict를 각 프로세스에 미리 배포
    with Pool(processes=num_procs, initializer=init_worker, initargs=(metadata_dict,)) as pool:
        # args에는 json_path만 전달 (데이터 전송량 최소화)
        for result in tqdm(
            pool.imap_unordered(check_single_file, json_files, chunksize=chunk_size),
            total=total_files,
            desc="Checking json files",
        ):
            if result is not None:
                bad_files.append(result)

    print("\n조건을 만족하지 않는 파일들:")
    for f in bad_files:
        print(f)

    print(f"\n총 출력된 파일 개수: {len(bad_files)}")

if __name__ == "__main__":
    main()