# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# TODO: add tests with model parallelism for activation partitioning and other features.

import sys
import weakref
from copy import deepcopy

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


class TestCPUActivationCheckpointNonReentrant(DistributedTest):
    """Async cpu_checkpointing through the non-reentrant path."""
    world_size = 1

    def test_cpu_offload_matches_baseline(self):
        test_act_ckpt._test_cpu_activation_checkpoint(non_reentrant_checkpoint)


class _SaveViewOfInput(torch.autograd.Function):
    """Saves a view of its input, as Liger's RMSNorm saves ``X.view(-1, dim)``."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x.view(-1, x.shape[-1]))
        return x * 2

    @staticmethod
    def backward(ctx, grad):
        ctx.saved_tensors
        return grad * 2


class TestCheckpointSavedViewIsNotRetained(DistributedTest):
    """A saved view reaches the pack hook flagged requires_grad and is_leaf; it must not be kept alive."""
    world_size = 1

    def test_saved_view_is_freed_after_forward(self):
        import weakref
        from deepspeed.accelerator import get_accelerator

        device = get_accelerator().current_device_name()
        module = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM).to(device)
        inputs = torch.rand(4, HIDDEN_DIM, device=device, requires_grad=True)
        intermediates = []

        def fn(x):
            hidden = module(x)
            intermediates.append(weakref.ref(hidden))
            return _SaveViewOfInput.apply(hidden)

        out = non_reentrant_checkpoint(fn, inputs)
        assert intermediates[0]() is None, "checkpoint kept the forward intermediate alive"

        out.sum().backward()
        expected = inputs.detach().clone().requires_grad_(True)
        _SaveViewOfInput.apply(module(expected)).sum().backward()
        assert torch.allclose(inputs.grad, expected.grad)


class _ViewOutputLinear(torch.nn.Module):
    """Returns an unsqueeze view of a 2D result, like a layer whose MLP flattens the batch dim."""

    def __init__(self, dim, bases):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
        self.bases = bases

    def forward(self, x):
        out = self.linear(x.squeeze(0))
        if torch.is_grad_enabled() and len(self.bases) < self.n_forward:
            self.bases.append(weakref.ref(out))
        return out.unsqueeze(0)


class TestCPUOffloadFreesViewArgs(DistributedTest):
    """Emptying a view arg's ``.data`` does not free its base; the offload path must not keep the view."""
    world_size = 1

    def test_offloaded_view_arg_base_is_freed(self):
        import deepspeed
        import deepspeed.runtime.activation_checkpointing.checkpointing as ds_ckpt
        from deepspeed.accelerator import get_accelerator

        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not offload activations")
        device = get_accelerator().current_device_name()
        n_layers = 6
        bases = []
        layers = torch.nn.ModuleList([_ViewOutputLinear(HIDDEN_DIM, bases) for _ in range(n_layers)]).to(device)
        for layer in layers:
            layer.n_forward = n_layers
        ref_layers = deepcopy(layers)
        x = torch.randn(1, 8, HIDDEN_DIM, device=device)

        saved = test_act_ckpt._snapshot_ckpt_config()
        try:
            deepspeed.checkpointing.configure(mpu_=None, checkpoint_in_cpu=True)
            if ds_ckpt._cpu_offload_engine is not None:
                ds_ckpt._cpu_offload_engine.reset()
            x_test = x.clone().requires_grad_()
            hidden = x_test
            for layer in layers:
                hidden = non_reentrant_checkpoint(layer, hidden)
            get_accelerator().synchronize()
            # Layer 0's output is layer 1's offloaded arg; keep-last and the forward stash hold only the newest few.
            assert bases[0]() is None, "an offloaded view arg kept its base resident on the device"
            hidden.sum().backward()
        finally:
            test_act_ckpt._restore_ckpt_config(saved)

        x_ref = x.clone().requires_grad_()
        hidden = x_ref
        for layer in ref_layers:
            hidden = layer(hidden)
        hidden.sum().backward()
        torch.testing.assert_close(x_ref.grad, x_test.grad)
        for p_ref, p_test in zip(ref_layers.parameters(), layers.parameters()):
            torch.testing.assert_close(p_ref.grad, p_test.grad)


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
