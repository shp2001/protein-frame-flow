#!/bin/bash
#SBATCH -J check_nonstandard
#SBATCH -p cpu
#SBATCH --mem=20g
#SBATCH -c 15
#SBATCH -w node01
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm_cif.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/parse_general/utils/check_nonstandard.py > /home/psh/protein-frame-flow/experiments/logs/parse_cif.log