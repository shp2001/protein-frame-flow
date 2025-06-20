#!/bin/bash
#SBATCH -J check_multi_model
#SBATCH -p cpu
#SBATCH --mem=20g
#SBATCH -c 15
#SBATCH -w node02
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm_cif.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/parse_general/utils/check_multi_model.py > /home/psh/protein-frame-flow/experiments/logs/check_multi_model_2.log