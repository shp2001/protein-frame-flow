#!/bin/bash
#SBATCH -J fm_tri_fape_layer6
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48g
#SBATCH -c 12
#SBATCH -w gpu01
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/train_se3_flows.py > /home/psh/protein-frame-flow/experiments/logs/fm_tri_fape_layer6.log