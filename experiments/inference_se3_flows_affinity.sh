#!/bin/bash
#SBATCH -J v2.3.2.2_affinity
#SBATCH -p h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=320G 
#SBATCH --qos=cssb_h100
#SBATCH -c 32
#SBATCH -w node02
#SBATCH --error=/home/psh/logs/v2.4.0_inf.err
#SBATCH --out=/home/psh/logs/v2.4.0_inf.out


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/inference_se3_flows_affinity.py > /home/psh/logs/v2.4.0_inf.log
