#!/bin/bash
#SBATCH -J v2.3.2.2_affinity
#SBATCH -p gpu
#SBATCH -w gpu01
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=36g
#SBATCH -c 12
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/inf.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/inference_se3_flows.py > /home/psh/protein-frame-flow/experiments/logs/CDRFlow_1.2.4_aug_2.log
