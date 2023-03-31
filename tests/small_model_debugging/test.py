# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.pt.deepspeed_linear import LinearModuleForZeroStage3
from deepspeed.pt.log_utils import logger
from deepspeed.accelerator import get_accelerator


def see_memory_usage(message):

    # Print message except when distributed but not rank 0
    logger.info(message)
    logger.info(
        "Memory Allocated %s GigaBytes ",
        get_accelerator().memory_allocated() / (1024 * 1024 * 1024),
    )
    logger.info(
        "Max Memory Allocated %s GigaBytes",
        get_accelerator().max_memory_allocated() / (1024 * 1024 * 1024),
    )
    logger.info(
        "Cache Allocated %s GigaBytes",
        get_accelerator().memory_cached() / (1024 * 1024 * 1024),
    )
    logger.info(
        "Max cache Allocated %s GigaBytes",
        get_accelerator().max_memory_cached() / (1024 * 1024 * 1024),
    )


tens = torch.rand(1024, 16384, dtype=torch.half, device=torch.device(get_accelerator().device_name()))
tens_back = tens.detach().clone()

#linear_bk = torch.nn.functional.linear
#torch.nn.functional.linear = deepspeed.pt.deepspeed_linear.LinearFunctionForZeroStage3.apply
model = LinearModuleForZeroStage3(16384, 16384)

model.to(get_accelerator().device_name()).half()

see_memory_usage("Before forward")
y = model(tens)

see_memory_usage("After forward")

model.weight.data = torch.zeros(1, dtype=torch.half, device=torch.device(get_accelerator().device_name()))

see_memory_usage("After weight zero")

y.backward(tens_back)
