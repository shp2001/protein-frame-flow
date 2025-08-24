#!/bin/bash
#SBATCH -J CDRFlow_v2.0.0_debug
#SBATCH -p gpu
#SBATCH -w gpu05
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=128g
#SBATCH -c 24
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm_debug.err


echo "Running train_abb3.py"
echo $CUDA_VISIBLE_DEVICES  # 여러 개 나와야 함
CUDA_LAUNCH_BLOCKING=1 
torchrun --standalone --nnodes=1 --nproc_per_node=2 /home/psh/protein-frame-flow/experiments/train_se3_flows.py \
    > /home/psh/protein-frame-flow/experiments/logs/CDRFlow_v2.0.0_debug.log 