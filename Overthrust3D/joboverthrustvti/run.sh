#!/bin/bash

#SBATCH -o joblogs/log

#SBATCH -e joblogs/err

#SBATCH -J SIM

#SBATCH -t 1000:0:00

#SBATCH -p normal

#SBATCH -N 250

#SBATCH --ntasks-per-node=4

#SBATCH --ntasks-per-socket=1

#SBATCH --mem=90000MB

#SBATCH --gres=dcu:4

#SBATCH --cpus-per-task=1


#SBATCH --exclude=d03r4n[00-19],d03r3n[00-19],b10r2n[00-19],c10r1n[00-19],c09r1n[00-19],b10r1n[00-19],b09r2n[00-19],c09r2n[00-19],b09r3n[00-19],c09r2n[00-19],a09r1n[00-19],d09r1n[00-19],a09r2n[00-19],c12r4n[00-19],d16r2n[00-19],d04r4n[00-19]


export OMP_NUM_THREADS=1

#module load compiler/devtoolset/7.3.1
#module load compiler/rocm/2.9
#module load mpi/hpcx/2.4.1/gcc-7.3.1
#  4x32, about   60 sec
ulimit -a
ulimit -s unlimited
ulimit -l unlimited
export MPITYPE=pmix_v3
#export MPITYPE=pmi2
echo "use srun" 
#time srun --mpi=$MPITYPE   ../../bin/forward3d  par/parameter.dat > logout
time srun --mpi=$MPITYPE   ../../bin/rtm3d  par/parameter.dat > logout
