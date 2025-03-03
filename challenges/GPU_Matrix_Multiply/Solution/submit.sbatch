#!/bin/bash
#SBATCH -J dgemm
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -N 1


srun -n1 -c1 --gpus-per-task=1  ./cpu_gpu_dgemm
#srun -n1 -c1  ./cpu_gpu_dgemm
