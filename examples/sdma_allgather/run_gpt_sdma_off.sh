#!/bin/bash
# GPT-7B-ish + ZeRO-3 baseline (RCCL allgather).
# Default: the SDMA fast-path inside deepspeed.comm stays off unless the
# user explicitly sets DS_SDMA_ALLGATHER=1, so this script simply doesn't
# export it.

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
deepspeed --num_gpus 8 "${SCRIPT_DIR}/train_zero3.py" \
    --deepspeed \
    --deepspeed_config "${SCRIPT_DIR}/ds_config_zero3.json" \
    --data_mode wikitext2 \
    --train_steps 100
