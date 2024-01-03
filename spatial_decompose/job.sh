#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 4
#SBATCH -p RM-small
#SBATCH --mem-per-cpu=2000M
#SBATCH -e job.err
#SBATCH -o job.out
#SBATCH -t 02:00:00

conda activate deepmd-gpu
echo $(which mpiexec)
mpiexec -n 4 python MDargon_spatial.py