# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# TODO: add tests with model parallelism for activation partitioning and other features.

import sys
import torch
import pytest
from importlib import util

from deepspeed.runtime.activation_checkpointing.checkpointing import non_reentrant_checkpoint
from unit.common import DistributedTest

# the hack to clone the module `test_activation_checkpointing` and inject
# `non_reentrant_checkpoint` as the `ckpt` of the origin test module
ORG_SPEC = util.find_spec('test_activation_checkpointing')
test_act_ckpt = util.module_from_spec(ORG_SPEC)
ORG_SPEC.loader.exec_module(test_act_ckpt)
sys.modules['test_act_ckpt'] = test_act_ckpt
test_act_ckpt.ckpt = non_reentrant_checkpoint

HIDDEN_DIM = test_act_ckpt.HIDDEN_DIM

MaskedLinear = test_act_ckpt.MaskedLinear
MaskedLinearSeq = test_act_ckpt.MaskedLinearSeq
MaskedLinearSeqDup = test_act_ckpt.MaskedLinearSeqDup
DropMaskLinear = test_act_ckpt.DropMaskLinear
LinearNonTensorInput = test_act_ckpt.LinearNonTensorInput
LinearNonTensorOutput = test_act_ckpt.LinearNonTensorOutput

_test_activation_checkpoint = test_act_ckpt._test_activation_checkpoint
_mixed_mask = test_act_ckpt._mixed_mask
_bool_to_float = test_act_ckpt._bool_to_float
_test_activation_checkpoint_ordering = test_act_ckpt._test_activation_checkpoint_ordering


class TestActivationCheckpointWithGrad(test_act_ckpt.TestActivationCheckpoint):
    """test `non_reentrant_checkpoint` can still checkpoint activations for inputs with grad"""
    pass


class TestCheckpointNonTensorWithGrad(test_act_ckpt.TestCheckpointNonTensor):
    """test `non_reentrant_checkpoint` can still checkpoint activations for inputs with grad"""
    pass


class TestCheckpointNonTensorOutputOrderingWithGrad(test_act_ckpt.TestCheckpointNonTensorOutputOrdering):
    """test `non_reentrant_checkpoint` can still checkpoint activations for inputs with grad"""
    pass


# below classes are used to test the graph with inputs have no grad and parameters has grad, namely partial graph?
@pytest.mark.parametrize('mask', [
    _mixed_mask(),
    _bool_to_float(_mixed_mask()),
])
class TestActivationCheckpointWithoutGrad(DistributedTest):
    """test all input tensors without grad"""
    world_size = 1

    def test_ckpt_inputs1_outputs1(self, mask):
        module = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        inputs = torch.rand(HIDDEN_DIM)
        _test_activation_checkpoint(module, inputs)

    def test_ckpt_inputs2_outputs1(self, mask):
        module = MaskedLinear(HIDDEN_DIM, HIDDEN_DIM)
        inputs = torch.rand(HIDDEN_DIM)
        _test_activation_checkpoint(module, inputs, mask)

    def test_ckpt_inputs2_outputs2(self, mask):
        module = MaskedLinearSeq(HIDDEN_DIM, HIDDEN_DIM)
        inputs = torch.rand(HIDDEN_DIM)
        _test_activation_checkpoint(module, inputs, mask)

    def test_ckpt_inputs2_outputs3(self, mask):
        module = MaskedLinearSeqDup(HIDDEN_DIM, HIDDEN_DIM)
        inputs = torch.rand(HIDDEN_DIM)
        _test_activation_checkpoint(module, inputs, mask)

    def test_ckpt_arg_none(self, mask):
        module = DropMaskLinear(HIDDEN_DIM, HIDDEN_DIM)
        inputs = (torch.rand(HIDDEN_DIM), None)
        _test_activation_checkpoint(module, *inputs)


@pytest.mark.parametrize('non_tensor', [None, 2, True, (None, 2.5), (None, True, torch.randn(HIDDEN_DIM))])
class TestCheckpointNonTensorWithoutGrad(DistributedTest):
    """test all input tensors without grad"""
    world_size = 1

    def test_ckpt_non_tensor_input(self, non_tensor):
        module = LinearNonTensorInput(HIDDEN_DIM, HIDDEN_DIM)
        inputs = torch.rand(HIDDEN_DIM)
        _test_activation_checkpoint(module, inputs, non_tensor)

    def test_ckpt_non_tensor_output(self, non_tensor):
        module = LinearNonTensorOutput(non_tensor)
        inputs = torch.rand(HIDDEN_DIM)
        _test_activation_checkpoint(module, inputs)


@pytest.mark.parametrize('non_tensor_output', [
    None, (torch.randn(HIDDEN_DIM), 2.5), (None, torch.randn(HIDDEN_DIM), True), (None, True, torch.randn(HIDDEN_DIM))
])
class TestCheckpointNonTensorOutputOrderingWithoutGrad(DistributedTest):
    """test all input tensors without grad"""
    world_size = 1

    def test_ckpt_non_tensor_output_ordering(self, non_tensor_output):
        module = LinearNonTensorOutput(non_tensor_output)
        inputs = torch.rand(HIDDEN_DIM)

        # First return is a tensor
        ordering = [True]
        if type(non_tensor_output) in [list, tuple]:
            ordering += [torch.is_tensor(t) for t in non_tensor_output]
        else:
            ordering += [torch.is_tensor(non_tensor_output)]
        _test_activation_checkpoint_ordering(module, ordering, inputs)
