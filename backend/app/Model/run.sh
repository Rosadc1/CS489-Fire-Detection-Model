#! /bin/bash

#SBATCH --job-name=A3
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu001
#SBATCH --time=1-12:00:00

source /home/username/miniconda3/bin/activate
conda activate a3

srun --unbuffered python -u main.py