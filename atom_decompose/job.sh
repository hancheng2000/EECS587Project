#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 16
#SBATCH -p RM-small
#SBATCH --mem-per-cpu=2000M
#SBATCH -e job.err
#SBATCH -o job.out
#SBATCH -t 01:00:00

conda activate dpgpu
# module load openmpi/4.0.5-gcc10.2.0
echo $(which mpirun)
~/.conda/envs/dpgpu/bin/mpirun -np $SLURM_NTASKS python MDargon_atomic.py
# ~/.conda/envs/dpgpu/bin/mpirun -np $SLURM_NTASKS python MDargon_serial.py
# python MDargon_serial.py