#!/bin/bash
#SBATCH -A m1503
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -c 8
#SBATCH --gpus-per-task=1

# Use srun to call the wrapper script. 
# We don't activate Spack here, so srun stays clean.
#srun --cpu-bind=cores ./wrap.sh test_case.py

srun --gpu-bind=single:1 ./wrap.sh test_case.py
#srun -n 4 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 ./wrap.sh test_case.py
#srun -n 4 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 nsys profile -o $SCRATCH/my_profile_%q{SLURM_PROCID} ./wrap.sh test_case.py

