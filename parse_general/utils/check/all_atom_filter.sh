#!/bin/bash
#SBATCH -J filter_all_atom_cif
#SBATCH -p cpu
#SBATCH --mem=20g
#SBATCH -c 15
#SBATCH -w node01
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/all_atom.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/parse_general/utils/all_atom_filter.py > /home/psh/protein-frame-flow/experiments/logs/all_atom.log