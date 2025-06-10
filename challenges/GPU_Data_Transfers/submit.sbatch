#!/bin/bash 
#SBATCH -A TRN039
#SBATCH -J add_vec_cpu
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1

module load PrgEnv-amd

srun -n1 --gpu-bind=closest --gpus=1 ./run 
