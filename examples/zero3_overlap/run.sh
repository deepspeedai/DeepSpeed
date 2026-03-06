#!/bin/bash
# =============================================================================
# DeepSpeed ZeRO-3 training with allgather overlap
# Single node, 8x AMD GPUs (MI200/MI250/MI300)
# =============================================================================

export NCCL_DEBUG=INFO
export HSA_FORCE_FINE_GRAIN_PCIE=1

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

deepspeed \
    --num_nodes 1 \
    --num_gpus 8 \
    "${SCRIPT_DIR}/train_zero3.py" \
    --deepspeed \
    --deepspeed_config "${SCRIPT_DIR}/ds_config_zero3.json" \
    --hidden_size 2048 \
    --num_layers 24 \
    --num_heads 16 \
    --max_seq_len 1024 \
    --train_steps 200
