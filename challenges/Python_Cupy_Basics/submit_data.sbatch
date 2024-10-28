#!/bin/bash
#SBATCH -A TRN001
#SBATCH -J cupy_xfer
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1

unset SLURM_EXPORT_ENV

date

module load PrgEnv-gnu/8.5.0 
module load rocm/6.1.3
module load craype-accel-amd-gfx90a
module load miniforge3

source activate /lustre/orion/world-shared/stf007/msandov1/crash_course_envs/cupy-frontier

srun -n1 --gpus-per-task=3 --gpu-bind=closest python3 data_transfer.py
