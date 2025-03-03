#!/bin/bash
#SBATCH -J hello
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -N 1
#SBATCH -n 4


srun -N 1 -n4 -c1 ./run
