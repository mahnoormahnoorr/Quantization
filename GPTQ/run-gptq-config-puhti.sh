#!/bin/bash                                                                                                                   
#SBATCH --account=xxxxxxxx                                                                                                         
#SBATCH --partition=gputest                                                                                                   
#SBATCH --ntasks=1                                                                                                            
#SBATCH --cpus-per-task=6                                                                                                     
#SBATCH --nodes=1                                                                                                             
#SBATCH --mem=32G                                                                                                             
#SBATCH --time=0:15:00                                                                                                        
#SBATCH --gres=gpu:v100:1                                                                                                     
#SBATCH --output=slurm-%j.out                                                                                                 
#SBATCH --error=slurm-%j.err                                                                                                  

# Load the module                                                                                                             
module purge
module use /appl/local/csc/modulefiles
module load pytorch

# This will store all the Hugging Face cache such as downloaded models                                                        
# and datasets in the project's scratch folder                                                                                
export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache
mkdir -p $HF_HOME

srun python3 gptq-config.py
