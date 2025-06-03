#!/bin/bash
#SBATCH -A TRN039
#SBATCH -J pytorch_cnn
#SBATCH -o %x-%j.out
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1

unset SLURM_EXPORT_ENV

date

module load PrgEnv-gnu/8.6.0 
module load rocm/6.1.3
module load craype-accel-amd-gfx90a
module load miniforge3

export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

source activate /gpfs/wolf2/olcf/stf007/world-shared/9b8/crashcourse_envs/torch-odo

python3 -u cnn.py
