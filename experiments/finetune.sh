#!/bin/bash
#SBATCH -J ebm_bb_rmsd_1e-3
#SBATCH -p gpu
#SBATCH -w gpu05
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=128g
#SBATCH -c 24
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm2.err


echo "Running train_abb3.py"
torchrun --standalone --nnodes=1 --nproc_per_node=2 /home/psh/protein-frame-flow/experiments/finetune.py > /home/psh/protein-frame-flow/experiments/logs/debug2.log