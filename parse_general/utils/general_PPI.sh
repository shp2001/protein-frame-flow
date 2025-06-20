#!/bin/bash
#SBATCH -J only_pp
#SBATCH -p cpu
#SBATCH --mem=48g
#SBATCH -c 12
#SBATCH -w node01
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/parse_general/utils/general_PPI.py > /home/psh/protein-frame-flow/experiments/logs/only_PPI.log