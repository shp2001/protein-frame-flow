#!/bin/bash
#SBATCH -J gen_diff_mask
#SBATCH -p cpu
#SBATCH --mem=20g
#SBATCH -c 15
#SBATCH -w node02
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/make_mask.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/parse_general/utils/make_mask.py > /home/psh/protein-frame-flow/experiments/logs/make_mask.log