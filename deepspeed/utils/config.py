# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.config_utils import DeepSpeedConfigModel

#########################################
# Timers
#########################################
# Timers. By default, timers are enabled.
# Users can configure in ds_config.json as below example:
TIMERS_FORMAT = '''
Timers should be enabled as:
"timers": {
  "throughput": {
    "enabled": true,
    "synchronized": true
  }
}
'''

TIMERS = "timers"
TIMERS_THROUGHPUT = "throughput"


def get_timers_config(param_dict):
    if param_dict and TIMERS in param_dict and TIMERS_THROUGHPUT in param_dict[TIMERS]:
        timers_config_dict = param_dict[TIMERS][TIMERS_THROUGHPUT]
    else:
        timers_config_dict = {}
    return DeepSpeedThroughputTimerConfig(**timers_config_dict)


class DeepSpeedThroughputTimerConfig(DeepSpeedConfigModel):
    """ Configure throughput timers """

    enabled: bool = True
    """ Turn on/off throughput timers """

    synchronized: bool = True
    """ Whether to synchronize a device when measuring the time.
        Synchronizing a device is required to produce the most accurate timer measurements.
        However, this comes at the expense of performance degradation. The CPU timer provides
        sufficient accuracy in many cases.
      """
