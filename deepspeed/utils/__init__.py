# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .logging import logger, log_dist
from .comms_logging import get_caller_func
#from .distributed import init_distributed
from .init_on_device import OnDevice
from .groups import *
from .nvtx import instrument_w_nvtx
# TODO: Move tensor fragment and mixed precision to zero utils
from .tensor_fragment import tensor_fragment, get_full_hp_param, get_hp_fragment_mapping, fragment_address, get_full_hp_grad
from .tensor_fragment import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state
from .mixed_precision_linkage import link_hp_params
from deepspeed.runtime.dataloader import RepeatingLoader
from .numa import get_numactl_cmd
