# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.config_utils import DeepSpeedConfigModel


class CompileConfig(DeepSpeedConfigModel):
    """ Configure compile settings """

    deepcompile: bool = False
    """ Turn on/off the DeepCompile mode """

    free_activation: bool = False
    """ Turn on/off the free activation mode """

    free_activation_threshold: int = 10 * 1024 * 1024
    """ In free activation mode, activations no less than this threshold (in byte) are eagerly freed """

    offload_activation: bool = False
    """ Turn on/off the activation offloading """

    offload_opt_states: bool = False
    """ Turn on/off the optimizer states offloading """

    double_buffer: bool = True
    """ Turn on/off the double buffering """

    symmetric_memory: bool = False
    """ Turn on/off the symmetric memory """

    debug_log: bool = False
    """ Turn on/off the graph dumping """

    offload_parameters: bool = False
    """ Turn on/off the parameter offloading """

    sync_before_reduce: bool = False
    """ Turn on/off the sync before reduce """

    sync_after_reduce: bool = False
    """ Turn on/off the sync after reduce """

    sync_before_allgather: bool = False
    """ Turn on/off the sync before allgather """

    sync_after_allgather: bool = False
    """ Turn on/off the sync after allgather """

    keep_int_input_tensors: bool = True
    """ Keep real values for int tensors in InputStorage instead of using dummy values """

    keep_all_input_tensors: bool = False
    """ Keep real values for all input tensors in InputStorage instead of using dummy values """
