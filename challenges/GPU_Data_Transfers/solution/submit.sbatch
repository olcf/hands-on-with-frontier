#!/bin/bash 
#SBATCH -J add_vec_cpu
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -N 1


srun -n1 -c1 --gpus-per-task=1 ./run 
