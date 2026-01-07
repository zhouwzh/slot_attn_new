#!/bin/bash

#SBATCH --output=/scratch/wz3008/SlotAttn/slurm_outputs/%x_%j.out
#SBATCH --error=/scratch/wz3008/SlotAttn/slurm_outputs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wz3008@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:3
#SBATCH --job-name=steve_movi_a
#SBATCH --account torch_pr_36_mren

singularity exec --nv \
    --overlay /scratch/wz3008/data/movi_a.sqf:ro \
    --overlay /scratch/wz3008/overlay-50G-10M-slotattn-test.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \
    -c "source /ext3/env.sh;conda activate steve; cd /scratch/wz3008/new_SlotAttn/steve; \
    python train.py \
        --data_path '/movi_a/*' --use_dp --batch_size 24"