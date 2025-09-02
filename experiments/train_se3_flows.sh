#!/bin/bash
#SBATCH -J CDRFlow_v2.0.1_sc_loss_schedule
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=100g
#SBATCH -c 12
#SBATCH -w gpu05
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm.err


echo "Running train_abb3.py"
echo $CUDA_VISIBLE_DEVICES  # 여러 개 나와야 함
torchrun --standalone --nnodes=1 --nproc_per_node=1 /home/psh/protein-frame-flow/experiments/train_se3_flows.py \
    > /home/psh/protein-frame-flow/experiments/logs/CDRFlow_v2.0.2_debug_2.log 