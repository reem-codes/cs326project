#!/bin/bash --login
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=128G 
#SBATCH --partition=batch 
#SBATCH --mail-type=ALL
#SBATCH --output=results/%x/%j-slurm.out
#SBATCH --error=results/%x/%j-slurm.err

# entire script fails if single command fails
set -e

module purge
# activate the conda environment
conda activate cs326project
module load gcc/6.4.0
# module load cuda/10.2.89

export PROJECT_DIR="$PWD"

# launch the training script
PYTHONPATH="$PROJECT_DIR"  python -u "$@"

