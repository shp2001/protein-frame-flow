#!/bin/bash
#SBATCH -J CDRFlow_v1.2.2_ddp
#SBATCH -p gpu
#SBATCH -w gpu04
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=128g
#SBATCH -c 24
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm.err


echo "Running train_abb3.py"
echo $CUDA_VISIBLE_DEVICES  # 여러 개 나와야 함
torchrun --nproc_per_node=2 /home/psh/protein-frame-flow/experiments/train_se3_flows.py > /home/psh/protein-frame-flow/experiments/logs/CDRFlow_v1.2.2.log