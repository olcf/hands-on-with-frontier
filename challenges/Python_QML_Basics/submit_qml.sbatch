#!/bin/bash
#SBATCH -A TRN001
#SBATCH -J qml_basics
#SBATCH -o %x-%j.out
#SBATCH -N 1
#SBATCH -t 0:10:00
#SBATCH -p batch

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
unset SLURM_EXPORT_ENV

cd $SLURM_SUBMIT_DIR
date

# Load modules
module load PrgEnv-gnu/8.5.0 
module load rocm/6.1.3
module load craype-accel-amd-gfx90a
module load kokkos/4.3.00-omp
module load miniforge3/23.11.0-0

source activate /lustre/orion/world-shared/stf007/msandov1/crash_course_envs/qml-frontier

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache" 
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH} 
rm -rf ${MIOPEN_USER_DB_PATH} 
mkdir -p ${MIOPEN_USER_DB_PATH}

# Get address of head node
export MASTER_ADDR=$(hostname -i)

export OMP_NUM_THREADS=1

srun -n3 -c7 --gpus-per-task=1 --gpu-bind=closest python3 -W ignore -u qml.py
