#!/bin/bash 
#SBATCH --job-name=check
#SBATCH --output=slurm-output/collect_analyze_pairs.out 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=sched_mit_binz,newnodes,sched_mit_hill
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --exclude=node[445,455,457,462]
#SBATCH --export=ALL
#SBATCH --constraint=centos7

python collect_pairs.py
python analyze_pairs.py

echo "job done"
