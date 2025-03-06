# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime import utils as ds_utils


def check_partition(weights, num_parts, target_diff):
    result = ds_utils.partition_balanced(weights=weights, num_parts=num_parts)

    parts_sum = []
    for b, e in zip(result[:-1], result[1:]):
        parts_sum.append(sum(weights[b:e]))

    assert max(parts_sum) - min(
        parts_sum
    ) == target_diff, f"ds_utils.partition_balanced(weights={weights}, num_parts={num_parts}) return {result}"


def test_partition_balanced():
    check_partition([1, 2, 1], 4, target_diff=2)
    check_partition([1, 1, 1, 1], 4, target_diff=0)
    check_partition([1, 1, 1, 1, 1], 4, target_diff=1)
    check_partition([1, 1, 1, 1, 0, 1], 4, target_diff=1)
