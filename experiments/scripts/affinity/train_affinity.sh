#!/bin/bash
#SBATCH -J v2.4.0_affinity_mut_mask_ag_mask_label_coords
#SBATCH -p h100
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=320G 
#SBATCH --qos=cssb_h100
#SBATCH -c 16
#SBATCH -w node01
#SBATCH --error=/home/psh/logs/v2.4.0.err
#SBATCH --out=/home/psh/logs/v2.4.0.out

echo "Running train_abb3.py"
echo $CUDA_VISIBLE_DEVICES  # 여러 개 나와야 함
torchrun --standalone --nnodes=1 --nproc_per_node=4 /home/psh/protein-frame-flow/experiments/scripts/affinity/train_affinity.py \
    > /home/psh/logs/ag_mask.log 