#!/bin/bash
# Run with SDMA allgather DISABLED (baseline RCCL allgather), default GPT shape (~7B).
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
deepspeed --num_gpus 8 "${SCRIPT_DIR}/train_zero3.py" \
    --deepspeed \
    --deepspeed_config "${SCRIPT_DIR}/ds_config_zero3_nosdma.json" \
    --data_mode wikitext2 \
    --train_steps 100
