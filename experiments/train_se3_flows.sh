#!/bin/bash
#SBATCH -J fm_tri_aa_nb_sample_x2
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48g
#SBATCH -c 12
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/train_fm.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/train_fm_4.err


echo "Running train_abb3.py"
export CUDA_LAUNCH_BLOCKING=1
python -u -W ignore /home/psh/protein-frame-flow/experiments/train_se3_flows.py > /home/psh/protein-frame-flow/experiments/logs/fm_tri_aanb_4.log