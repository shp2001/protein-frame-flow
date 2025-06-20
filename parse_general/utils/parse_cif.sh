#!/bin/bash
#SBATCH -J cif_2_pkl
#SBATCH -p cpu
#SBATCH --mem=30g
#SBATCH -c 30
#SBATCH -w node02
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/parse_cif.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/parse_general/utils/parse_cif.py > /home/psh/protein-frame-flow/experiments/logs/parse_cif.log