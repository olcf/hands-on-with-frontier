#!/bin/bash
#SBATCH -A TRN039
#SBATCH -J qml_basics
#SBATCH -o %x-%j.out
#SBATCH -N 1
#SBATCH -t 0:20:00
#SBATCH -p batch
#SBATCH -C nvme

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
unset SLURM_EXPORT_ENV

cd $SLURM_SUBMIT_DIR
date

# Load modules
module load PrgEnv-gnu/8.6.0 
module load rocm/6.1.3
module load craype-accel-amd-gfx90a
module load miniforge3

source activate /gpfs/wolf2/olcf/stf007/world-shared/9b8/crashcourse_envs/qml-odo

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/mnt/bb/${USER}/my-miopen-cache" 
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH} 
rm -rf ${MIOPEN_USER_DB_PATH} 
mkdir -p ${MIOPEN_USER_DB_PATH}

# Get address of head node
export MASTER_ADDR=$(hostname -i)
export NCCL_SOCKET_IFNAME=hsn0 # necessary NCCL socket for Frontier

export OMP_NUM_THREADS=1

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

srun -n1 -c7 --gpus-per-task=1 --gpu-bind=closest python3 -W ignore -u qml.py
