#!/bin/bash
#SBATCH -A TRN025
#SBATCH -J cupy_xfer
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1

unset SLURM_EXPORT_ENV

date

module load PrgEnv-gnu
module load amd-mixed/5.3.0
module load craype-accel-amd-gfx90a
module load miniforge3

source activate ~/.conda/envs/cupy-frontier

srun -n1 --gpus-per-task=3 --gpu-bind=closest python3 data_transfer.py
