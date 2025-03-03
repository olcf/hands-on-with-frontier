#!/bin/bash
#SBATCH -J mat_mul
#SBATCH -o %x-%j.out
#SBATCH -t 0:30:00
#SBATCH -N 1



srun -n1 -c1 --gpus-per-task=1 ./matrix_multiply
