#!/bin/bash -l
#SBATCH --job-name=opt125m
#SBATCH --account=project_xxxxxxx 
#SBATCH --partition=dev-g 
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%j.out

# Load the module
module purge
module use /appl/local/csc/modulefiles
module load pytorch

python3 bitsandbytesquantization.py
