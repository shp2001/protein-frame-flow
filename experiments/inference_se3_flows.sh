#!/bin/bash
#SBATCH -J fm_default_framepred_inf
#SBATCH -p gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH --mem=60g
#SBATCH -c 12
#SBATCH -w gpu02
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/inf.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/inference_se3_flows.py > /home/psh/protein-frame-flow/experiments/logs/inf_default.log