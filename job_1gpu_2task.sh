#!/bin/bash
#SBATCH --job-name=trial
#SBATCH --gres=gpu
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=512G
#SBATCH --error=%x.err
#SBATCH --output=%x.out

module load orion/gpu
spack env activate linalg-gpu
source ~/venvs/edrix_spack/bin/activate


hostname
free -h
nvidia-smi

ompi_info | grep -i "cuda"

#srun ncu --target-processes all -o combined_report python test_case.py
#srun nsys profile -o timeline_%q{SLURM_PROCID} python test_case.py
#srun python -m cProfile -o profile.out test_case.py
#srun python test_case.py
#CUDA_VISIBLE_DEVICES=0,1 srun python test_case.py

# cancel cuda aware mpi
export OMPI_MCA_btl_smcuda_use_cuda_ipc=0

#srun env OMP_NUM_THREADS=1 python test_case.py
srun python test_case.py


#     -c 1 env OMP_NUM_THREADS=1 \
#srun -n 2 \
#     -c 1 env OMP_NUM_THREADS=1 \
#	python test_case.py
