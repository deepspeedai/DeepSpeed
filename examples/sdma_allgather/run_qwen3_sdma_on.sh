#!/bin/bash
# Qwen3-32B + DeepSpeed ZeRO-3 with the SDMA fast-path opted in.
#
# DS_SDMA_ALLGATHER=1 is the single opt-in switch.  When set,
# deepspeed.comm's TorchBackend tries to bring up the mori SDMA backend
# at init time and routes WORLD-group all_gather_into_tensor through it.
# Mori's MORI_ENABLE_SDMA=1 is auto-exported on the user's behalf when
# DS_SDMA_ALLGATHER=1 is set, so users normally don't need to touch it.
# This script otherwise uses the same ds_config as run_qwen3_sdma_off.sh;
# the only difference is this env var.
set -eu

export DS_SDMA_ALLGATHER=1

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
deepspeed --num_gpus "${NUM_GPUS:-8}" "${SCRIPT_DIR}/train_qwen3_zero3.py" \
    --model_name "${MODEL:-Qwen/Qwen3-32B}" \
    --num_layers "${NUM_LAYERS:-0}" \
    --seq_length "${SEQ_LEN:-1024}" \
    --batch_size "${BATCH_SIZE:-1}" \
    --num_steps "${NUM_STEPS:-100}" \
    --warmup_steps "${WARMUP_STEPS:-10}" \
    --ds_config "${DS_CONFIG:-${SCRIPT_DIR}/ds_config_zero3.json}"
