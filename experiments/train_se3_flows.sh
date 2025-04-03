#!/bin/bash
#SBATCH -J fm_tri_partial_mask
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=30g
#SBATCH -c 8
#SBATCH -w gpu03
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/train_se3_flows.py > /home/psh/protein-frame-flow/experiments/logs/fm_tri_partial_mask.log