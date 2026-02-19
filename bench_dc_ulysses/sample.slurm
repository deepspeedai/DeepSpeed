#!/bin/bash
# SLURM Job Submission OR Interactive Environment Setup
# For batch submission: sbatch sample.slurm
# For interactive:   
# srun -A bcjw-delta-gpu --time=1:00:00 --nodes=1 --mem=100G --gpus=2 --partition=gpuA100x4-interactive --pty /bin/bash
#
# SBATCH directives (ignored when sourced, used for sbatch submission):
#SBATCH -A bcjw-delta-gpu
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --gpus=2
#SBATCH --gpu-bind=closest
#SBATCH --mem=100G
#SBATCH --output=slurm_out/out_%j.log

source ~/.bashrc
conda activate /u/ndani/autosp-env

# # NCCL configuration
export NCCL_INCLUDE_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nccl-2.25/include
export NCCL_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nccl-2.25
export CPATH=$NCCL_HOME/include:$CPATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
# export CPATH=$NCCL_INCLUDE_DIR:$CPATH

export TRITON_CACHE_DIR="/tmp/triton_$USER"
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.8/bin:$PATH

export HF_DATASETS_CACHE="/u/$USER/.cache"
export HF_HOME=$HF_DATASETS_CACHE
export HF_HUB_CACHE=$HF_DATASETS_CACHE
export HF_ASSETS_CACHE=$HF_DATASETS_CACHE
export TRANSFORMERS_CACHE=$HF_DATASETS_CACHE

