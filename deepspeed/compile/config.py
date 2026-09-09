# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Optional, Literal
from deepspeed.runtime.config_utils import DeepSpeedConfigModel

PassName = Literal["z1", "z3", "autosp", "autotp"]


class CompileConfig(DeepSpeedConfigModel):
    """ Configure compile settings """

    deepcompile: bool = False
    """ Turn on/off the DeepCompile mode """

    free_activation: bool = False
    """ Turn on/off the free activation mode """

    free_activation_threshold: int = 10 * 1024 * 1024
    """ In free activation mode, activations no less than this threshold (in byte) are eagerly freed """

    offload_activation: bool = False
    """ Move activations that the forward pass saves for the backward pass to pinned host memory,
    and bring each one back shortly before the backward pass reads it. Only tensors of at least 5MB
    with a fixed shape are considered, and only as many as the memory budget requires. Runs in place
    of the prefetch/selective-gather passes and is mutually exclusive with offload_parameters and
    offload_opt_states. """

    offload_activation_pin_memory: bool = True
    """ Pin host buffers used for DeepCompile activation offload. Required for
    full-bandwidth async GPU<->CPU copies. Disable only under tight memlock
    limits (ulimit -l). Defaults to True to match ZeRO offload pin_memory. """

    offload_opt_states: bool = False
    """ Offload optimizer states (fp32 master parameters and Adam moments) to pinned host memory
    during forward/backward and reload them for the optimizer step, keeping resident whatever the
    memory budget allows. Runs in place of the prefetch/selective-gather passes and is mutually
    exclusive with offload_parameters. Designed for gradient_accumulation_steps=1: the compiled
    graph runs once per micro-batch, so accumulation repeats the whole offload/reload cycle. """

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

    passes: Optional[List[PassName]] = None
    """ Composes different optimizations. """
