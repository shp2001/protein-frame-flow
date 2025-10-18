import glob
import os 

import pymol 
pymol.finish_launching()

from pymol import cmd

root_dir = "/home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.3.2.1_stage_2/2025-10-14_00-20-37/epoch=63-step=91712_copy/run_2025-10-16_12-51-03"
files = []
for mutation in os.listdir(root_dir):
     if 'config' in mutation:
         continue
     mut_dir = os.path.join(root_dir, mutation)

     for sample in os.listdir(mut_dir):
        if 'sample' not in sample:
            continue
        sample_path = os.path.join(mut_dir, sample, "sample_1.pdb")
        files.append(sample_path)
        break

    

# 기준 구조 (첫 번째 파일)
ref = files[0].split('/')[-1].replace(".pdb", "")
cmd.load(files[0], ref)

# 나머지 파일 로드 및 정렬
for f in files[1:]:
    name = f.split('/')[-1].replace(".pdb", "")
    cmd.load(f, name)

# 색상 랜덤 지정
cmd.util.cbc()

# H3 루프 선택 및 강조 (필요 시 수정)
cmd.select("cdrh3", "chain H and resi 95-102")
cmd.show("cartoon")
cmd.show("sticks", "cdrh3")
cmd.set("cartoon_transparency", 0.4)
cmd.zoom("cdrh3")