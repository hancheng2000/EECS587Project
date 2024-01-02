#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 4
#SBATCH -p RM-small
#SBATCH --mem-per-cpu=2000M
#SBATCH -e job.err
#SBATCH -o job.out
#SBATCH -t 00:05:00

conda activate deepmd-gpu
module load openmpi
echo $(which mpiexec)
echo $(which mpirun)
mpirun -n 4 python MDargon_spatial.py