#!/bin/bash
#SBATCH -J CDRFlow_v2.4.0_general_stage1_hotspot_mask_2
#SBATCH -p h100
#SBATCH -c 40
#SBATCH -w node02
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=320G 
#SBATCH --error=/home/psh/logs/v2.4.0.err
#SBATCH --out=/home/psh/logs/v2.4.0.out

echo "Running train_abb3.py"
echo $CUDA_VISIBLE_DEVICES  # 여러 개 나와야 함
torchrun --standalone --nnodes=1 --nproc_per_node=4 /home/psh/protein-frame-flow/experiments/train_se3_flows.py \
    > /home/psh/logs/v2.4.0.log 