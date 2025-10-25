#!/bin/bash
#SBATCH -J v2.3.2.1_stage2_wt_confidence
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=200g
#SBATCH -c 64
#SBATCH -w gpu05
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm.err


echo "Running train_abb3.py"
echo $CUDA_VISIBLE_DEVICES  # 여러 개 나와야 함
torchrun --standalone --nnodes=1 --nproc_per_node=2 /home/psh/protein-frame-flow/experiments/train_se3_flows.py \
    > /home/psh/protein-frame-flow/experiments/logs/v2.3.2.1_wt_confidence.log 