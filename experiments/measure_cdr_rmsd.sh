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
python -u -W ignore /home/psh/protein-frame-flow/experiments/measure_cdr_rmsd.py --pred_dir /home/psh/protein-frame-flow/inference_outputs/fm_tri_aa_nb_sample_x2_crop450/2025-05-08_19-45-45/epoch=93-step=133104/run_2025-05-11_17-44-03 --label_dir /home/psh/benchmark_after210930/pdb_only_ab > /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd2.log