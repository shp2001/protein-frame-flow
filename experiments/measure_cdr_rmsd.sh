#!/bin/bash
#SBATCH -J measure_cdr_metric
#SBATCH -p gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH --mem=10g
#SBATCH -c 6
#SBATCH -w gpu02
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd.err


echo "Measure CDR Metric.py"
sleep 43200
python -u -W ignore /home/psh/protein-frame-flow/experiments/measure_cdr_rmsd.py --pred_dir /home/psh/protein-frame-flow/inference_outputs/fm_tri_pair_loss_partial_mask/2025-04-21_20-47-54/epoch=89-step=101880_copy/run_2025-04-24_16-49-50 --label_dir /home/psh/benchmark_after210930/pdb_only_ab > /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd2.log