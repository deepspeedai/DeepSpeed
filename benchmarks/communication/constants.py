# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.accelerator import get_accelerator

DEFAULT_WARMUPS = 5
DEFAULT_TRIALS = 50
DEFAULT_TYPE = 'float'
DEFAULT_BACKEND = get_accelerator().communication_backend_name()
DEFAULT_UNIT = 'Gbps'
DEFAULT_DIST = 'deepspeed'
DEFAULT_MAXSIZE = 24
TORCH_DISTRIBUTED_DEFAULT_PORT = 29500
