#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 4
#SBATCH -p highmem
#SBATCH --mem-per-cpu=2000M
#SBATCH -e job.err
#SBATCH -o job.out
#SBATCH -t 5:00:00

module load openmpi/4.1.3
conda activate deepmd-cpu
srun -n 4 python MDargon_spatial.py