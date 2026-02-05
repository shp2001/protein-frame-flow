#!/bin/bash
#SBATCH -J CDRFlow_v2.4.0
#SBATCH -p h100
#SBATCH -c 10
#SBATCH -w node02
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G 
#SBATCH --error=/home/psh/logs/v2.4.0_inf.err
#SBATCH --out=/home/psh/logs/v2.4.0_inf.out


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/inference_se3_flows.py > /home/psh/logs/v2.4.0_inf.log
