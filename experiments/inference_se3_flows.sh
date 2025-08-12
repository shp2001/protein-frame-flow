#!/bin/bash
#SBATCH -J CDRFlow_1.2.4_aug
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=38g
#SBATCH -c 12
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/inf.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/inference_se3_flows.py > /home/psh/protein-frame-flow/experiments/logs/CDRFlow_1.2.4_aug.log