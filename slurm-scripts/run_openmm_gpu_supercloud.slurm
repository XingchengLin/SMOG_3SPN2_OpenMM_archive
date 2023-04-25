#!/bin/bash
#SBATCH --job-name=openmm
#SBATCH -o slurm-output/test_openmm.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=10GB
#SBATCH --time=12:00:00

module load cuda/10.0
python OpenMM_SMOG_3SPN2C_nx12mers.py

echo "job done"

