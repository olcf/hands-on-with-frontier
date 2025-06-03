#!/bin/bash
#SBATCH -A TRN039
#SBATCH -J cupy_xfer
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1

unset SLURM_EXPORT_ENV

date

module load PrgEnv-gnu/8.6.0 
module load rocm/5.7.1
module load craype-accel-amd-gfx90a
module load miniforge3

source activate /gpfs/wolf2/olcf/stf007/world-shared/9b8/crashcourse_envs/cupy-odo

srun -n1 --gpus-per-task=3 python3 data_transfer.py
