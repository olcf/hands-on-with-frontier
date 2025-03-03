#!/bin/bash
#SBATCH -J vec_add
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -N 1


date
echo " " 


export OMP_NUM_THREADS=4

srun -n1 -c4 --threads-per-core=1 ./vec_add

