# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .logging import logger, log_dist, log_dist_once, set_log_level_from_string
from .comms_logging import get_caller_func
#from .distributed import init_distributed
from .init_on_device import OnDevice
from .groups import *
from .nvtx import instrument_w_nvtx

from .z3_leaf_module import set_z3_leaf_modules, unset_z3_leaf_modules, get_z3_leaf_modules, z3_leaf_module, z3_leaf_parameter, set_z3_leaf_module, set_z3_leaf_modules_by_name, set_z3_leaf_modules_by_suffix

from deepspeed.runtime.dataloader import RepeatingLoader
from .numa import get_numactl_cmd
