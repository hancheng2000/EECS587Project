#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 4
#SBATCH -p RM-small
#SBATCH --mem-per-cpu=2000M
#SBATCH -e job.err
#SBATCH -o job.out
#SBATCH -t 00:02:00

conda activate mpi
echo $(which mpirun)
mpiexec -np $SLURM_NTASKS python MDargon_spatial.py