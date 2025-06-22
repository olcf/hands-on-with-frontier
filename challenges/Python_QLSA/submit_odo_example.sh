#!/bin/bash
#SBATCH -A TRN039
#SBATCH -J qlsa
#SBATCH -o "%x_%j".out
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 00:10:00

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
source activate /gpfs/wolf2/olcf/stf007/world-shared/9b8/crashcourse_envs/qlsa-solver 

# HHL circuit generator
srun -N1 -n1 -c1 python circuit_HHL.py -case sample-tridiag -casefile input_vars.yaml --savedata

# Run on simulator
srun -N1 -n1 -c2 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 1000 -backtyp ideal --savedata

# Run on emulator
#srun -N1 -n1 -c2 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 1000 -backtyp real-iqm -backmet fake_garnet --savedata

# Run on real device
#source keys.sh 
#srun -N1 -n1 -c2 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 1000 -backtyp real-iqm -backmet garnet --savedata

# Plot results
srun -N1 -n1 -c1 python plot_fidelity_vs_shots.py

# Run as simultaneous job steps (https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#multiple-independent-job-steps)
# srun -N1 -n1 -c2 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 100 &
# srun -N1 -n1 -c2 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 1000 &
# srun -N1 -n1 -c2 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 10000 &
# wait
