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
python -u -W ignore /home/psh/protein-frame-flow/experiments/measure_cdr_rmsd.py --pred_dir /home/psh/protein-frame-flow/inference_outputs/fm_tri_pair_loss_nano_crop300/2025-05-02_12-34-59/epoch=107-step=148068/run_2025-05-03_16-27-47 --label_dir /home/psh/benchmark_after210930/pdb_only_ab > /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd2.log