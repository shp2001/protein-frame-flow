#!/bin/bash
#SBATCH -J v2.3.2.2_sbe
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --gres=gpu:v100:1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/inference_se3_flows.py > %x_%j.log
