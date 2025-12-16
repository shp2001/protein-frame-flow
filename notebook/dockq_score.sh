# 결과를 저장할 폴더 생성
mkdir -p results

# models_dir 안의 모든 pdb 파일에 대해 반복
for model in models_dir/*.pdb; do
    # 파일 이름만 추출 (예: 1A2K_model.pdb -> 1A2K_model)
    filename=$(basename "$model" .pdb)
    
    # DockQ 실행
    # 결과 JSON을 results 폴더에 파일명과 동일하게 저장
    DockQ "$model" path/to/native.pdb --json "results/${filename}.json"
    
    echo "Processed: $filename"
done