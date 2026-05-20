#!/bin/bash
# Qwen3-32B + DeepSpeed ZeRO-3 baseline (RCCL allgather).
#
# Default: deepspeed.comm's SDMA fast-path stays off unless the user
# explicitly sets DS_SDMA_ALLGATHER=1, so this script simply doesn't
# export it and pairs cleanly with run_qwen3_sdma_on.sh (same ds_config;
# only env vars differ) for the A/B.
#
#   model      : Qwen/Qwen3-32B (full 64 layers, BF16, eager attention)
#   data       : wikitext-103-raw-v1, 10% split, model's own tokenizer
#   parallel   : ZeRO-3, DP=8 (single node, MI300X x 8)
#   bucket     : DeepSpeed defaults (stage3_prefetch_bucket_size = 5e7)
#   seq/bs     : seq_length=1024, micro_batch=1
#   steps      : 100 measured + 10 warmup
#
# Override via env vars:  SEQ_LEN, BATCH_SIZE, NUM_STEPS, WARMUP_STEPS,
# NUM_GPUS, MODEL, DS_CONFIG.
set -eu

# Reduce HIP allocator fragmentation — the 32B model has long-lived tensors
# that benefit from segment expansion under heavy ZeRO-3 churn.
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0  # quiets harmless TCPStore shutdown trace

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
deepspeed --num_gpus "${NUM_GPUS:-8}" "${SCRIPT_DIR}/train_qwen3_zero3.py" \
    --model_name "${MODEL:-Qwen/Qwen3-32B}" \
    --num_layers "${NUM_LAYERS:-0}" \
    --seq_length "${SEQ_LEN:-1024}" \
    --batch_size "${BATCH_SIZE:-1}" \
    --num_steps "${NUM_STEPS:-100}" \
    --warmup_steps "${WARMUP_STEPS:-10}" \
    --ds_config "${DS_CONFIG:-${SCRIPT_DIR}/ds_config_zero3.json}"
