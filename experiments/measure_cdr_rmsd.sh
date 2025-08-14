#!/bin/bash
#SBATCH -J cdr_pred
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:1
<<<<<<< HEAD
#SBATCH --mem=5g
#SBATCH -c 3
=======
#SBATCH --mem=10g
#SBATCH -c 6
>>>>>>> inference_batch_aug_2
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/inf.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd.err

echo "Measure CDR Metric.py"
<<<<<<< HEAD
python -u -W ignore /home/psh/protein-frame-flow/experiments/measure_cdr_rmsd.py --pred_dir /home/psh/protein-frame-flow/inference_outputs/CDRFlow_v1.2.4_biomol_stage2/2025-08-11_21-58-59/epoch=21-step=31548_copy/run_2025-08-12_19-43-33 --label_dir /home/psh/benchmark_after210930/pdb_only_ab > /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd2.log
=======
sleep 21600
python -u -W ignore /home/psh/protein-frame-flow/experiments/measure_cdr_rmsd.py --pred_dir /home/psh/protein-frame-flow/inference_outputs/CDRFlow_v1.2.4_biomol_stage2_no_conf/2025-08-12_08-52-08/epoch=40-step=58794_copy/run_2025-08-12_22-48-20 --label_dir /home/psh/benchmark_after210930/pdb_only_ab > /home/psh/protein-frame-flow/experiments/logs/cdr_rmsd2.log
>>>>>>> inference_batch_aug_2
