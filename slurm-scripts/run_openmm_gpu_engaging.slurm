#!/bin/bash
#SBATCH --job-name=openmm
#SBATCH -o slurm-output/slurm.%j.out
#SBATCH -e slurm-output/slurm.%j.err
#SBATCH --partition=sched_mit_binz_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --time=12:00:00
#SBATCH --constraint=centos7

module purge
module load gcc engaging/openmpi/2.0.3 cuda/10.0
python test_xclin_dimer_code.py

echo "job done"

