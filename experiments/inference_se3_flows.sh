#!/bin/bash
#SBATCH -J v2.3.2.1_sc17_h3_single
#SBATCH -p gpu
#SBATCH -w gpu02
#SBATCH --gres=gpu:A5000:1
#SBATCH --mem=80g
#SBATCH -c 12
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/inf.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/inference_se3_flows.py > /home/psh/protein-frame-flow/experiments/logs/CDRFlow_1.2.4_aug_1.log
