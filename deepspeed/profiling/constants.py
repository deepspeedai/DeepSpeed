"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

#########################################
# flops profiler
#########################################
# Flops profiler. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
FLOPS_PROFILER_FORMAT = '''
flops profiler should be enabled as:
"session_params": {
  "flops_profiler": {
    "enalbe": [true|false],
    "start_step": 5,
    "end_step": 6,
    "module_depth": -1,
    "top_modules": 3,
    }
}
'''

FLOPS_PROFILER = "flops_profiler"

FLOPS_PROFILER_ENABLED = "enabled"
FLOPS_PROFILER_ENABLED_DEFAULT = False

FLOPS_PROFILER_START_STEP = "start_step"
FLOPS_PROFILER_START_STEP_DEFAULT = 5

FLOPS_PROFILER_END_STEP = "end_step"
FLOPS_PROFILER_END_STEP_DEFAULT = FLOPS_PROFILER_START_STEP_DEFAULT + 1

FLOPS_PROFILER_MODULE_DEPTH = "module_depth"
FLOPS_PROFILER_MODULE_DEPTH_DEFAULT = -1

FLOPS_PROFILER_TOP_MODULES = "top_modules"
FLOPS_PROFILER_TOP_MODULES_DEFAULT = 3
