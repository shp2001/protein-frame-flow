#!/bin/bash
#SBATCH -J CDRFlow_v1_lrfinder
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=48g
#SBATCH -c 12
#SBATCH -w gpu04
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/lr_finder.py > /home/psh/protein-frame-flow/experiments/logs/lr_finder.log