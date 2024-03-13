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
from .tensor_fragment import tensor_fragment, get_full_hp_param, get_hp_fragment_mapping, fragment_address, get_full_hp_grad, map_to_flat_opt_states
from .tensor_fragment import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state
from .tensor_fragment import set_full_hp_param
from .tensor_fragment import safe_set_full_fp32_param, safe_set_full_optimizer_state
from .tensor_fragment import safe_get_local_fp32_param, safe_get_local_grad, safe_get_local_optimizer_state
from .tensor_fragment import safe_set_local_fp32_param, safe_set_local_optimizer_state
from .z3_leaf_module import set_z3_leaf_modules, unset_z3_leaf_modules, get_z3_leaf_modules, z3_leaf_module, z3_leaf_parameter
from .mixed_precision_linkage import link_hp_params, lazy_init_hp_params_optimizer_state
from deepspeed.runtime.dataloader import RepeatingLoader
from .numa import get_numactl_cmd
