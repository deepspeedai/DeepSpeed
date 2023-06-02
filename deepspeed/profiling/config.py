# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.config_utils import get_scalar_param, DeepSpeedConfigObject
from deepspeed.profiling.constants import *


class DeepSpeedFlopsProfilerConfig(DeepSpeedConfigObject):

    def __init__(self, param_dict):
        super(DeepSpeedFlopsProfilerConfig, self).__init__()

        self.enabled = None
        self.recompute_fwd_factor = None
        self.profile_step = None
        self.module_depth = None
        self.top_modules = None

        if FLOPS_PROFILER in param_dict.keys():
            flops_profiler_dict = param_dict[FLOPS_PROFILER]
        else:
            flops_profiler_dict = {}

        self._initialize(flops_profiler_dict)

    def _initialize(self, flops_profiler_dict):
        self.enabled = get_scalar_param(flops_profiler_dict, FLOPS_PROFILER_ENABLED, FLOPS_PROFILER_ENABLED_DEFAULT)

        self.recompute_fwd_factor = get_scalar_param(flops_profiler_dict, FLOPS_PROFILER_RECOMPUTE_FWD_FACTOR,
                                                     FLOPS_PROFILER_RECOMPUTE_FWD_FACTOR_DEFAULT)

        self.profile_step = get_scalar_param(flops_profiler_dict, FLOPS_PROFILER_PROFILE_STEP,
                                             FLOPS_PROFILER_PROFILE_STEP_DEFAULT)

        self.module_depth = get_scalar_param(flops_profiler_dict, FLOPS_PROFILER_MODULE_DEPTH,
                                             FLOPS_PROFILER_MODULE_DEPTH_DEFAULT)

        self.top_modules = get_scalar_param(flops_profiler_dict, FLOPS_PROFILER_TOP_MODULES,
                                            FLOPS_PROFILER_TOP_MODULES_DEFAULT)

        self.detailed = get_scalar_param(flops_profiler_dict, FLOPS_PROFILER_DETAILED, FLOPS_PROFILER_DETAILED_DEFAULT)

        self.output_file = get_scalar_param(flops_profiler_dict, FLOPS_PROFILER_OUTPUT_FILE,
                                            FLOPS_PROFILER_OUTPUT_FILE_DEFAULT)
