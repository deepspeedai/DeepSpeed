# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.inference.v2.inference_utils import ceil_div


def test_ceil_div_basic_behavior():
    assert ceil_div(10, 4) == 3
    assert ceil_div(12, 4) == 3


def test_ceil_div_rejects_zero_divisor():
    with pytest.raises(ValueError, match="b must be non-zero"):
        ceil_div(10, 0)
