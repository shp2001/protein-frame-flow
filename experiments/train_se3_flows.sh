#!/bin/bash
#SBATCH -J CDRFlow_v1.4.2
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=128g
#SBATCH -c 24
#SBATCH -w gpu05
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm.err


echo "Running train_abb3.py"
torchrun --nproc_per_node=4 /home/psh/protein-frame-flow/experiments/train_se3_flows.py > /home/psh/protein-frame-flow/experiments/logs/CDRFlow_v1.4.2.log