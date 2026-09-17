#!/bin/bash
# GPT-7B-ish + ZeRO-3 with the SDMA fast-path opted in.
#
# DS_SDMA_ALLGATHER=1 is the single opt-in switch.  When set,
# deepspeed.comm's TorchBackend tries to bring up the mori SDMA backend
# at init time and routes WORLD-group all_gather_into_tensor through it.
# Mori's MORI_ENABLE_SDMA=1 is auto-exported on the user's behalf when
# DS_SDMA_ALLGATHER=1 is set, so users normally don't need to touch it.
# Without DS_SDMA_ALLGATHER=1, even an mori-installed run stays on RCCL.
export DS_SDMA_ALLGATHER=1

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
deepspeed --num_gpus 8 "${SCRIPT_DIR}/train_zero3.py" \
    --deepspeed \
    --deepspeed_config "${SCRIPT_DIR}/ds_config_zero3.json" \
    --data_mode wikitext2 \
    --train_steps 100
