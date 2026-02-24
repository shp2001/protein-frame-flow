# CDRFlow

CDRFlow는 항원-항체 복합체의 서열과 epitope 정보를 활용해 정밀하게 CDR conformation 및 binding orientation을 예측할 수 있도록 한다.

## Input: 
- 항원-항체 복합체의 서열
- 항원-항체 복합체의 backbone 구조
  - 정답 구조가 아닌 예측 구조, 혹은 MT를 예측할 때는 WT의 구조 활용
  - 모델이 대략적인 binding orientation과 epitope 정보를 알 수 있도록 한다. 

## Output:
- 항원-항체 복합체의 all-atom 구조 

## Inference 방법
### Data Processing 
1. /home/psh/data/{target_name}/pdb을 만들고 분석하고자 하는 protein의 PDB 파일을 저장. 
    - 저장하는 PDB file은 chothia numbering이 되지 않은, 기본 PDB 파일이어야 한다.
    - Heavy chain id는 A, Light chain id는 B로 맞춰야 한다.
    - 아직 코드 상의 이유로 하나의 PDB 파일만 처리한다. 

2. /home/psh/protein-frame-flow/notebook/process_data.ipynb 주피터 노트북에서 INPUT_PDB_DIR와 CDR_SEQS를 지정하고 '모두 실행' 클릭
    - INPUT_PDB_DIR: 사용자가 PDB 파일을 저장한 디렉토리 
    - CDR_SEQS: 사용자가 지정한 CDR 영역의 서열

### Utilize DMS Data (dms point mutation을 예측하고 싶지 않다면 이 단계는 스킵)
1. /home/psh/data/{target_name}/dms에 Proteina에서 제공한 dms excel 파일 저장
2. /home/psh/protein-frame-flow/notebook/utilize_dms.ipynb 주피터 노트북에서 INPUT_META_DIR, XLSX_FILES, REL_AFFINITY_THRESHOLD을 '모두 실행' 클릭 
    - INPUT_META_DIR: /home/psh/data/{target_name}/meta
    - XLSX_FILES: dms excel 파일 경로들을 리스트 형태로 지정
    - REL_AFFINITY_THRESHOLD: DMS 데이터 상에서 relativate affinity가 REL_AFFINITY_THRESHOLD 이상인 point mutation만 선별

### Prediction 
1. /home/psh/protein-frame-flow/configs/_inference.yaml의 csv_path키 값을 /home/psh/data/{target_name}/meta/metadata_dms.csv으로 변경
2. terminal에서 다음을 실행
    conda activate fm
    cd /home/psh/protein-frame-flow
    sbatch 
