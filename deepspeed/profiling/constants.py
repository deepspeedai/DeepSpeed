# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

#########################################
# flops profiler
#########################################
# Flops profiler. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
FLOPS_PROFILER_FORMAT = '''
flops profiler should be enabled as:
"session_params": {
  "flops_profiler": {
    "enabled": true,
    "recompute_fwd_factor": 0.0,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true,
    "output_file": null
    }
}
'''

FLOPS_PROFILER = "flops_profiler"

FLOPS_PROFILER_ENABLED = "enabled"
FLOPS_PROFILER_ENABLED_DEFAULT = False

FLOPS_PROFILER_RECOMPUTE_FWD_FACTOR = "recompute_fwd_factor"
FLOPS_PROFILER_RECOMPUTE_FWD_FACTOR_DEFAULT = 0.0

FLOPS_PROFILER_PROFILE_STEP = "profile_step"
FLOPS_PROFILER_PROFILE_STEP_DEFAULT = 1

FLOPS_PROFILER_MODULE_DEPTH = "module_depth"
FLOPS_PROFILER_MODULE_DEPTH_DEFAULT = -1

FLOPS_PROFILER_TOP_MODULES = "top_modules"
FLOPS_PROFILER_TOP_MODULES_DEFAULT = 1

FLOPS_PROFILER_DETAILED = "detailed"
FLOPS_PROFILER_DETAILED_DEFAULT = True

FLOPS_PROFILER_OUTPUT_FILE = "output_file"
FLOPS_PROFILER_OUTPUT_FILE_DEFAULT = None
