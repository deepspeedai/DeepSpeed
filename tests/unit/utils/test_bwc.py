# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.utils.bwc import (bwc_tensor_model_parallel_rank, bwc_tensor_model_parallel_world_size,
                                 bwc_tensor_model_parallel_group)

TP_RANK = 3
TP_WORLD_SIZE = 8
TP_GROUP = 'tp-group'


class CurrentMPU:
    """Current Megatron and DeepSpeed spelling."""

    def get_tensor_model_parallel_rank(self):
        return TP_RANK

    def get_tensor_model_parallel_world_size(self):
        return TP_WORLD_SIZE

    def get_tensor_model_parallel_group(self):
        return TP_GROUP


class SliceMPU:
    """Spelling used by some DeepSpeed + pipeline parallelism versions."""

    def get_slice_parallel_rank(self):
        return TP_RANK

    def get_slice_parallel_world_size(self):
        return TP_WORLD_SIZE

    def get_slice_parallel_group(self):
        return TP_GROUP


class LegacyMPU:
    """Deprecated Megatron and DeepSpeed spelling."""

    def get_model_parallel_rank(self):
        return TP_RANK

    def get_model_parallel_world_size(self):
        return TP_WORLD_SIZE

    def get_model_parallel_group(self):
        return TP_GROUP


class SequenceParallelMPU:
    """Partitions along the sequence dimension only, as Ulysses does.

    Exposes no tensor model parallel API in any of the three spellings.
    """

    def get_sequence_parallel_rank(self):
        return TP_RANK

    def get_sequence_parallel_world_size(self):
        return TP_WORLD_SIZE

    def get_sequence_parallel_group(self):
        return TP_GROUP


@pytest.mark.parametrize('mpu_cls', [CurrentMPU, SliceMPU, LegacyMPU])
class TestSupportedMPUSpellings:

    def test_rank(self, mpu_cls):
        assert bwc_tensor_model_parallel_rank(mpu_cls()) == TP_RANK

    def test_world_size(self, mpu_cls):
        assert bwc_tensor_model_parallel_world_size(mpu_cls()) == TP_WORLD_SIZE

    def test_group(self, mpu_cls):
        assert bwc_tensor_model_parallel_group(mpu_cls()) == TP_GROUP


@pytest.mark.parametrize('mpu', [None, SequenceParallelMPU()])
class TestNoTensorModelParallelism:
    """An mpu without a tensor model parallel API must read as "not split".

    Without this, Ulysses sequence parallelism raises AttributeError from the
    BF16_Optimizer gradient-norm path instead of reporting no tensor
    parallelism.
    """

    def test_rank_is_zero(self, mpu):
        assert bwc_tensor_model_parallel_rank(mpu) == 0

    def test_world_size_is_one(self, mpu):
        assert bwc_tensor_model_parallel_world_size(mpu) == 1

    def test_group_is_none(self, mpu):
        assert bwc_tensor_model_parallel_group(mpu) is None
