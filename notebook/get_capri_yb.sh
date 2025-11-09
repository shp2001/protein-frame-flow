#!/bin/bash
#SBATCH -J v2.3.2.1_capri
#SBATCH -p gpu
#SBATCH -w gpu02
#SBATCH --gres=gpu:A5000:1
#SBATCH --mem=90g
#SBATCH -c 15
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/inf.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/notebook/get_capri_yb.py > /home/psh/protein-frame-flow/experiments/logs/v2.3.2.1_capri.log
