#!/bin/bash
#SBATCH -J CDRFlow_guidance_5.0
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=15g
#SBATCH -c 4
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/inf.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/inference_se3_flows.py > /home/psh/protein-frame-flow/experiments/logs/guidance_5.0.log