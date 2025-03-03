#!/bin/bash
#SBATCH -t 0:10:00
#SBATCH -N 1
#SBATCH -c 4


srun -n4 -c1 ./mpi_bcast
