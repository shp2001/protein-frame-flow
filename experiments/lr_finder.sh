#!/bin/bash
#SBATCH -J CDRFlow_lrfinder
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=48g
#SBATCH -c 12
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm.err


echo "Running train_abb3.py"
torchrun --standalone --nnodes=1 --nproc_per_node=2 /home/psh/protein-frame-flow/experiments/lr_finder.py > /home/psh/protein-frame-flow/experiments/logs/lr_finder.log