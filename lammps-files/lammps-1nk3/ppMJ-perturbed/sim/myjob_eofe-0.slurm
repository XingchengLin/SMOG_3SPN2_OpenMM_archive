#!/bin/bash 
#SBATCH --job-name=free 
#SBATCH --output=slurm-%j-curr-0.out 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=sched_mit_binz,newnodes,sched_mit_hill
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --exclude=node[444,447,448,451,453,457,459,462,464]
#SBATCH --export=ALL
#SBATCH --constraint=centos7

module purge
module load gcc
module load engaging/openmpi/1.8.8
export pyPath=/home/xclin/bin/anaconda2/bin

# print the node name that the job uses
echo "job uses node $SLURMD_NODENAME"

# For plumed 2.4.6 MATHEVAL package
export PLUMED_USE_LEPTON=yes

lammpsdir="/home/xclin/bin/awsem/bin"

curr=0 
max_id=0 
prev=$(($curr - 1))
next=$(($curr + 1))
curr=$(printf "%'03d" ${curr}) 
prev=$(printf "%'03d" ${prev})

cd run${curr} # run the following commands in run${curr}

sed "s/RANDOMSEED/$RANDOM/g" ../../template_template.lammps > ../../template.lammps 
$pyPath/python  ../../get_lammps_input.py ../run${prev} ./ ${curr}

#mpirun -np 1 $lammpsdir/lmp_openmpi_plumed246_3spn_2016_sbm_rigidtemper_tanh -partition 1x1 -in proteinDna_sim.in
mpirun -np 1 /nfs/pool002/users/smliu/tools/lammps-sbm3spn/lammps-sbm3spn-test/src/lmp_openmpi -partition 1x1 -in proteinDna_sim.in

cd ../

echo "simulation finished"

if [ "${next}" -le "${max_id}" ]
then
    sbatch myjob_eofe-${next}.slurm
fi

echo "job done"
