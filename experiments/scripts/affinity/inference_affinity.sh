#!/bin/bash
#SBATCH -J v2.3.2.2_affinity
#SBATCH -p gpu
#SBATCH -w gpu04
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48g
#SBATCH -c 12
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/inf.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/scripts/affinity/inference_affinity.py > /home/psh/protein-frame-flow/experiments/logs/CDRFlow_1.2.4_aug_1.log
