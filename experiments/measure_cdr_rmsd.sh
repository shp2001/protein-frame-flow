#!/bin/bash
#SBATCH -J cdr_pred
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=5g
#SBATCH -c 3
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd.err

echo "Measure CDR Metric.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/measure_cdr_rmsd.py\
 --pred_dir /home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.3.1.1/2025-09-10_17-24-40/epoch=106-step=153331/run_2025-09-11_21-15-16\
 --label_dir /home/psh/benchmark_after210930/pdb_only_ab > /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd2.log
