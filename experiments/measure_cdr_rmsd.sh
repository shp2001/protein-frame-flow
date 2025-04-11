#!/bin/bash
#SBATCH -J measure_cdr_metric
#SBATCH -p gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH --mem=10g
#SBATCH -c 6
#SBATCH -w gpu02
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/measure_cdr_rmsd.py --pred_dir /home/psh/protein-frame-flow/inference_outputs/fm_tri_partial_mask/2025-04-01_23-02-35/epoch=131-step=149424/total --label_dir /home/psh/benchmark_after210930/pdb_only_ab > /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd2.log