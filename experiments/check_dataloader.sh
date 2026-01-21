#!/bin/bash
#SBATCH -J check_dataloader
#SBATCH -p gpu
#SBATCH -w gpu05
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=128g
#SBATCH -c 24
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm2.err


echo "Running train_abb3.py"
torchrun --standalone --nnodes=1 --nproc_per_node=4 /home/psh/protein-frame-flow/experiments/check_dataloader.py > /home/psh/protein-frame-flow/experiments/logs/check_dataloader.log