#!/bin/bash
#SBATCH --account=project_xxxxxxx
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm-%j.out

# Load the module
module purge
module use /appl/local/csc/modulefiles
module load pytorch

python3 bitsandbytesquantization.py
