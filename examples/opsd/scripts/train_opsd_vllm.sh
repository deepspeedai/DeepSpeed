#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
#
# Launch OPSD training with vLLM rollout on a disjoint GPU group.
#
# Default config assumes 8 GPUs: ranks 0..5 train (ZeRO-3), devices 6-7 run
# vLLM with TP=2. Adjust configs/opsd_vllm_disjoint.json::rollout.gpus and
# NUM_TRAIN_GPUS to match your topology.
set -euo pipefail

CONFIG="${1:-configs/opsd_vllm_disjoint.json}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-6}"
INCLUDE_GPUS="${INCLUDE_GPUS:-0,1,2,3,4,5}"

deepspeed --num_gpus "${NUM_TRAIN_GPUS}" --include "localhost:${INCLUDE_GPUS}" \
    main.py --config "${CONFIG}"
