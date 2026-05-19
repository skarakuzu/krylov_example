#!/bin/bash

# 1. Load system-level requirements
module load spack PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 craype-network-ofi

# 2. Setup and Activate Spack
. /global/common/software/nersc9/spack/1.1.0/share/spack/setup-env.sh
spack env activate edrixs_gpu

# 3. Load all Python packages
spack load --first python py-numba py-mpi4py py-slepc4py py-petsc4py arpack-ng

#export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# B. Critical MPI/GPU variables for NERSC
#export MPICH_GPU_SUPPORT_ENABLED=0

# C. Fix the GTL path and Preload
# Perlmutter's GTL is usually in this specific path (verify with 'module show cray-mpich')
export CRAY_GTL_PATH=/opt/cray/pe/mpich/9.0.1/gtl/lib
export LD_PRELOAD=$CRAY_GTL_PATH/libmpi_gtl_cuda.so.0:$LD_PRELOAD


export MPICH_GPU_IPC_ENABLED=0
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_PXN_DISABLE=1
export NCCL_P2P_DISABLE=1          # Also disable NCCL Peer-to-Peer
#export MPICH_GPU_IPC_DISABLE=1 

# D. Disable IPC and use "Device to Device" mode (safest for PETSc hangs)
#export MPICH_GPU_IPC_DISABLE=1
#export MPICH_GPU_IPC_CPP_OFF=1
#export CUDA_IPC_DISABLE=1
#export MPICH_GTL_DEBUG=2


# Force the use of GTL for collective operations
#export MPICH_GTL_VT_DEVICE_ID=$SLURM_LOCALID

# E. Debug line to confirm GPU visibility
echo "Rank $SLURM_PROCID (Local $SLURM_LOCALID) using GPU: $(nvidia-smi --query-gpu=uuid --format=csv,noheader)"

# 5. Run the Python command
#python "$@" -use_gpu_aware_mpi 0 -vec_type standard
python "$@" -use_gpu_aware_mpi 1 -vec_type cuda -mat_type aijcusparse

