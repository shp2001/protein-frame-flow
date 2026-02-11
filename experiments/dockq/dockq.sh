#!/bin/bash
#SBATCH -J dockq_wo_ag_cond
#SBATCH -p cpu
#SBATCH --mem=30g
#SBATCH -c 30
#SBATCH -o /home/psh/protein-frame-flow/experiments/logs/dockq_wo_ag_cond.log
#SBATCH -e /home/psh/protein-frame-flow/experiments/logs/dockq_wo_ag_cond.err

echo "Measure CDR Metric.py"
python -u -W ignore /home/psh/protein-frame-flow/experiments/dockq/dockq_benchmark.py
