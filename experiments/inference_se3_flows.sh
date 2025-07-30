#!/bin/bash
#SBATCH -J CDRFlow_inf
#SBATCH -p gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH --mem=70g
#SBATCH -c 16
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/inf.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/inference_se3_flows.py > /home/psh/protein-frame-flow/experiments/logs/guidance_5.0.log