#!/bin/bash
#SBATCH -J srun_subil
#SBATCH -o %x-%j.out
#SBATCH -t 10:00
#SBATCH -N 1

# number of OpenMP threads
export OMP_NUM_THREADS=2

# jsrun command to modify 
srun -N 1 -n 1 -c 1 ./hello_mpi_omp | sort
