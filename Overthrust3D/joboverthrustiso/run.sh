#!/bin/bash

#SBATCH -o joblogs/log

#SBATCH -e joblogs/err

#SBATCH -J RTM

#SBATCH -t 1000:0:00

#SBATCH -p normal

#SBATCH -N 5

#SBATCH --ntasks-per-node=4

#SBATCH --ntasks-per-socket=1

#SBATCH --mem=90000MB

#SBATCH --gres=dcu:4

#SBATCH --cpus-per-task=1

#SBATCH --exclusive

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
time srun --mpi=$MPITYPE   ../../bin/forward3d  par/parameter.dat > logout
