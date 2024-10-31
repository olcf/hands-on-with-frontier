#!/bin/bash
#SBATCH -A TRN001
#SBATCH -J galaxy
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1

unset SLURM_EXPORT_ENV

date

module load PrgEnv-gnu/8.5.0 
module load cray-hdf5-parallel/1.12.2.9
module load miniforge3

source activate /lustre/orion/world-shared/stf007/msandov1/crash_course_envs/h5pympi-frontier

srun -n8 python3 -u galaxy.py
