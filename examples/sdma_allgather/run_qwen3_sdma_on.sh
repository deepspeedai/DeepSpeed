#!/bin/bash
# Qwen3-32B + DeepSpeed ZeRO-3 with the transparent SDMA fast-path.
#
# On AMD MI300 with mori installed, deepspeed.comm auto-detects the SDMA
# backend at init time and routes WORLD-group all_gather_into_tensor calls
# through it.  No ds_config flag is required — this script uses the same
# config as run_qwen3_sdma_off.sh; the only difference is the env vars.
set -eu

# REQUIRED for the SDMA path: tells mori to use hipExtMallocWithFlags +
# hipDeviceMallocUncached for transit buffers.  Without this the SDMA
# kernel reads cached memory and faults at NULL on every rank.
export MORI_ENABLE_SDMA=1

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
