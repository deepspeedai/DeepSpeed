#!/bin/bash
# Run with SDMA allgather ENABLED, default GPT shape (~7B).

# mori SymmMemManager only allocates uncached (hipExtMallocWithFlags +
# hipDeviceMallocUncached) transit buffers when MORI_ENABLE_SDMA is set;
# otherwise the SDMA kernel reads cached memory and faults at NULL on every
# rank.  Always export it for SDMA-on runs.
export MORI_ENABLE_SDMA=1

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
deepspeed --num_gpus 8 "${SCRIPT_DIR}/train_zero3.py" \
    --deepspeed \
    --deepspeed_config "${SCRIPT_DIR}/ds_config_zero3_sdma.json" \
    --data_mode wikitext2 \
    --train_steps 100
