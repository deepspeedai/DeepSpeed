#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
#
# Launch OPSD training with vLLM rollout.
#
# The vLLM server is started **lazily** as a subprocess by training rank 0
# on first use, so no separate vLLM launch step is required.  The GPUs
# listed in ``rollout.gpus`` in the config are assigned to the vLLM server
# via ``CUDA_VISIBLE_DEVICES`` in the subprocess environment.
#
# Default config assumes 8 GPUs: ranks 0..5 train (ZeRO-3), devices 6-7
# run vLLM with TP=2.  Adjust configs/opsd_vllm_disjoint.json::rollout.gpus
# and NUM_TRAIN_GPUS to match your topology.
set -euo pipefail

CONFIG="${1:-configs/opsd_vllm_disjoint.json}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-6}"
INCLUDE_GPUS="${INCLUDE_GPUS:-0,1,2,3,4,5}"

deepspeed --num_gpus "${NUM_TRAIN_GPUS}" --include "localhost:${INCLUDE_GPUS}" \
    main.py --config "${CONFIG}"
