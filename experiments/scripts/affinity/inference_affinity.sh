#!/bin/bash
#SBATCH -J v2.3.2.2_affinity
#SBATCH -p h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=200G 
#SBATCH --qos=cssb_h100
#SBATCH -c 16
#SBATCH -w node01
#SBATCH --error=/home/psh/logs/inf.err
#SBATCH --out=/home/psh/logs/inf.out


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/scripts/affinity/inference_affinity.py > /home/psh/protein-frame-flow/experiments/logs/inf.log
