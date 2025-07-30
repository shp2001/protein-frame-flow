#!/bin/bash
#SBATCH -J cdr_pred
#SBATCH -p gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH --mem=5g
#SBATCH -c 3
#SBATCH -w gpu02
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd.err

echo "Measure CDR Metric.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/measure_cdr_rmsd.py --pred_dir /home/psh/protein-frame-flow/inference_outputs/CDRFlow_v1.4.3/2025-07-27_14-27-06/epoch=89-step=129060_copy/run_2025-07-30_10-14-39 --label_dir /home/psh/benchmark_after210930/pdb_only_ab > /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd2.log