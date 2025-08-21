#!/bin/bash
#SBATCH -J cdr_pred
#SBATCH -p gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH --mem=5g
#SBATCH -c 3
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd.err

echo "Measure CDR Metric.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/measure_cdr_rmsd.py\
 --pred_dir /home/psh/protein-frame-flow/inference_outputs/ebm_margin_10_over14/2025-08-20_16-58-28/epoch=49-step=35850_copy/run_2025-08-21_13-02-09\
 --label_dir /home/psh/benchmark_after210930/pdb_only_ab > /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd2.log