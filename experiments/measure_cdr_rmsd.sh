#!/bin/bash
#SBATCH -J fm_default_framepred_inf
#SBATCH -p gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH --mem=60g
#SBATCH -c 12
#SBATCH -w gpu02
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd.err


echo "Running train_abb3.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/measure_cdr_rmsd.py --pred_dir /home/psh/protein-frame-flow/inference_outputs/default_frame_pred/2025-03-06_23-24-22/last/scaffolding/run_2025-03-09_23-18-37 --label_dir /home/psh/benchmark_after210930/pdb_only_ab > /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd2.log