# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import torch


def check_and_handle_empty_buffer(
    buffer_m: torch.Tensor,
    original_shape: torch.Size,
    original_size: int,
    worker_error: torch.Tensor,
    server_error: torch.Tensor,
) -> Optional[torch.Tensor]:
    if original_size == 0:
        if worker_error.numel():
            worker_error.zero_()
        if server_error.numel():
            server_error.zero_()
        if len(original_shape) > 1:
            return buffer_m.reshape(original_shape)
        return buffer_m
    return None
