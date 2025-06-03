#!/bin/bash
#SBATCH -A trn037
#SBATCH -J qlsa
#SBATCH -o "%x_%j"
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 00:10:00

# Odo is a smaller HPC system at Oak Ridge. The following project paths are examples.

# Only necessary if submitting this job script like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
unset SLURM_EXPORT_ENV

# Set proxy settings so compute nodes can reach internet (required when using a real device)
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

module load miniforge3

# HHL circuit generator
source activate /gpfs/wolf2/olcf/trn037/proj-shared/81a/software/miniconda3-odo/envs/qlsa-circuit
mkdir models
srun -N1 -n1 -c1 python circuit_HHL.py -case sample-tridiag -casefile input_vars.yaml --savedata

# Run circuit
srun -N1 -n1 -c2 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 1000

# Run on real device
source keys.sh 
srun -N1 -n1 -c2 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 1000 -backtyp real-iqm -backmet garnet:mock --savedata

# Plot results
srun -N1 -n1 -c1 python plot_fidelity_vs_shots.py

# Run as simultaneous job steps (https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#multiple-independent-job-steps)
# srun -N1 -n1 -c2 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 100 &
# srun -N1 -n1 -c2 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 1000 &
# srun -N1 -n1 -c2 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 10000 &
# wait
