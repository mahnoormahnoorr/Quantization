#!/bin/bash -l
#SBATCH --account=project_xxxxxxx 
#SBATCH --partition=dev-g 
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm-%j.out

# Load the module
module purge
module use /appl/local/csc/modulefiles
module load pytorch

python3 bitsandbytesquantization.py
