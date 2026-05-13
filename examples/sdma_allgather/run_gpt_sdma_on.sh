#!/bin/bash
# GPT-7B-ish + ZeRO-3 with the transparent SDMA fast-path.
#
# On AMD MI300 with mori installed, deepspeed.comm auto-detects the SDMA
# backend at init time and routes WORLD-group all_gather_into_tensor calls
# through it.  No ds_config flag is required.
#
# MORI_ENABLE_SDMA=1 is REQUIRED for the SDMA path: it tells mori to use
# hipExtMallocWithFlags + hipDeviceMallocUncached for transit buffers.
# Without it the SDMA kernel reads cached memory and faults at NULL.
export MORI_ENABLE_SDMA=1

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
deepspeed --num_gpus 8 "${SCRIPT_DIR}/train_zero3.py" \
    --deepspeed \
    --deepspeed_config "${SCRIPT_DIR}/ds_config_zero3.json" \
    --data_mode wikitext2 \
    --train_steps 100
