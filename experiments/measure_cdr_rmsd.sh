#!/bin/bash
#SBATCH -J cdr_pred
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=5g
#SBATCH -c 3
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd.err

echo "Measure CDR Metric.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/measure_cdr_rmsd.py --pred_dir /home/psh/protein-frame-flow/inference_outputs/CDRFlow_v1.2.4_ebm_exp/2025-08-15_15-29-32/epoch=7-step=11472/run_2025-08-16_15-29-19 --label_dir /home/psh/benchmark_after210930/pdb_only_ab > /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd2.log