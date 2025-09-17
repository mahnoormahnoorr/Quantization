#!/bin/bash -l
#SBATCH --account=project_xxxxxxx 
#SBATCH --partition=dev-g 
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

# Load the module
module purge
module use /appl/local/csc/modulefiles
module load pytorch

python3 bitsandbytesquantization.py
