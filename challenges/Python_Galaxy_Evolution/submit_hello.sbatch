#!/bin/bash
#SBATCH -A TRN025
#SBATCH -J mpi4py
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1

unset SLURM_EXPORT_ENV

date

module load PrgEnv-gnu
module load cray-hdf5-parallel
module load miniforge3

source activate ~/.conda/envs/h5pympi-frontier

srun -n42 python3 -u hello_mpi.py
