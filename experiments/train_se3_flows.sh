#!/bin/bash
#SBATCH -J CDRFlow_v2.3.1.1
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=200g
#SBATCH -c 32
#SBATCH -w gpu05
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm.err


echo "Running train_abb3.py"
echo $CUDA_VISIBLE_DEVICES  # 여러 개 나와야 함
torchrun --standalone --nnodes=1 --nproc_per_node=2 /home/psh/protein-frame-flow/experiments/train_se3_flows.py \
    > /home/psh/protein-frame-flow/experiments/logs/CDRFlow_v2.3.1.1.log 