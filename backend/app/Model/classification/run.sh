#! /bin/bash

#SBATCH --job-name=s25chrisrFireClassifcation
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu001
#SBATCH --time=1-12:00:00

source /home/s25chrisr/miniconda3/bin/activate
conda activate a3

srun --unbuffered python -u fire_classification_main.py