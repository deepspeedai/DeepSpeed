#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
#
# Launch OPSD training with the DeepSpeed hybrid-engine rollout (no vLLM).
# Assumes you're cd'd into examples/opsd/.
set -euo pipefail

CONFIG="${1:-configs/opsd_hybrid_engine.json}"
NUM_GPUS="${NUM_GPUS:-8}"

deepspeed --num_gpus "${NUM_GPUS}" main.py --config "${CONFIG}"
