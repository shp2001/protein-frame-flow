#!/bin/bash
#SBATCH -J v2.3.2.2_affinity
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=120g
#SBATCH -c 32
#SBATCH -w gpu04
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm.err

export PYTHONWARNINGS="ignore"
echo "Running train_abb3.py"
echo $CUDA_VISIBLE_DEVICES  # 여러 개 나와야 함
torchrun --standalone --nnodes=1 --nproc_per_node=4 /home/psh/protein-frame-flow/experiments/scripts/affinity/train_affinity.py \
    > /home/psh/protein-frame-flow/experiments/logs/v2.3.2.2_affinity.log 