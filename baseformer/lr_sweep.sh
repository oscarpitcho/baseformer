#!/bin/bash
#SBATCH --job-name=lr_sweep_small
#SBATCH --array=0-2
#SBATCH --output=slurm-%A_%a.out
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=opitcho@mit.edu

# Learning rates and experiment names
LRS=(1e-2 7e-3 3e-3)
NAMES=(small_lr_1e-2 small_lr_7e-3 small_lr_3e-3)

# Get values for this task
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_TASK_ID: lr=$LR, name=$NAME"

# Run with Hydra overrides
uv run python main.py \
    experiment.lr=$LR \
    experiment_name=$NAME \
    checkpoint.resume=null
