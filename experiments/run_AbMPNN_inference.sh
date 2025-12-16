#!/bin/bash
#SBATCH -J OptimAb_avelumab
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem=32g
#SBATCH -o out.%j
#SBATCH -e err.%j
#SBATCH --gres=gpu:A5000:1
#SBATCH -w gpu02

source ~/.bashrc
conda activate ligandmpnn_env

export PYTHONPATH=/home/fullmoon/projects/ProteinMPNN_CSSB/fullmoon_initial_package:$PYTHONPATH
python -u /home/psh/protein-frame-flow/experiments/run_AbMPNN_inference.py \
        --pdb_path "/home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.3.2.1_stage_2_ori_perturb_stage3/2025-12-06_22-56-57/epoch=52-step=75949_copy/5grj_high_var/5grj_H_L_A_WT__WT__K100R_G102S/optimab_input/sample_0_upper.pdb" \
        --out_folder "/home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.3.2.1_stage_2_ori_perturb_stage3/2025-12-06_22-56-57/epoch=52-step=75949_copy/5grj_high_var/5grj_H_L_A_WT__WT__K100R_G102S/optimab_output" \
        --number_of_batches 50 \
        --batch_size 20 \
        --temperature 0.5 \
        --fasta_seq_separation ":" \
        --model "/home/fullmoon/projects/Ab_MPNN/EF_MPNN/train/BUG_FIX_Ab_ft_PatchDecoding_chunk_8_epoch_5_rigid/weights/Ab_ft_PatchDecoding_chunk_8_epoch_5_rigid_best.pt" \
        --redesigned_residues "H99 H100 H101 H102 H103 H104 H105 H106 H107 H108" \
        --parse_atoms_with_zero_occupancy 1 \
        --omit_AA "C" \
        --decoding_type "PatchDecoding_chunk" \
        --N_patch 8 