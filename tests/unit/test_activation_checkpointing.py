# TODO: add tests with model parallelism for activation partitioning and other features.

from copy import deepcopy

import pytest

import torch

import deepspeed
ckpt = deepspeed.checkpointing.checkpoint

from common import distributed_test


def _compute(module, *inputs, do_checkpoint=False):
    if do_checkpoint:
        outputs = ckpt(module, *inputs)
    else:
        outputs = module(*inputs)

    if torch.is_tensor(outputs):
        outputs = (outputs, )

    sum(o.sum() for o in outputs if o.requires_grad).backward()
    grads = [p.grad for p in module.parameters()]
    input_grads = [inp.grad for inp in inputs if torch.is_tensor(inp)]

    return {
        'outputs': outputs,
        'module_grads': grads,
        'input_grads': input_grads,
    }


def _prep_inputs(*inputs):
    _inputs = []

    for inp in inputs:
        inp = deepcopy(inp)
        if torch.is_tensor(inp):
            inp = inp.cuda()
        _inputs.append(inp)

    return tuple(_inputs)


# This is distributed because checkpoint() assumes that torch.distributed is initialized.
# torch.distributed is used with activation partitioning, but not for these simple cases.
@distributed_test(world_size=1)
def _test_activation_checkpoint(module, *inputs):
    # Move to device
    module.cuda()

    # Get rid of dropouts until we fork the RNG between tests.
    module.eval()

    module_ = deepcopy(module)
    inputs_ = _prep_inputs(*inputs)
    base = _compute(module_, *inputs_, do_checkpoint=False)

    module_ = deepcopy(module)
    inputs_ = _prep_inputs(*inputs)
    test = _compute(module_, *inputs_, do_checkpoint=True)

    for group in base.keys():
        for b, t in zip(base[group], test[group]):
            # Catch grad `None`s, etc.
            if not torch.is_tensor(b):
                assert b == t
            elif b.is_floating_point():
                assert torch.allclose(b, t)
            else:
                assert torch.equal(b, t)


#
# Helpers
#


class MaskedLinear(torch.nn.Linear):
    def forward(self, x, mask):
        out = super().forward(x)
        if mask.is_floating_point():
            out = out * mask
        else:
            # must cast BoolTensor in older torch versions
            out = out * mask.type_as(out)
        return out


class MaskedLinearSeq(MaskedLinear):
    """Tests pipeline modules by also returning the mask."""
    def forward(self, x, mask):
        return super().forward(x, mask), mask


class MaskedLinearSeqDup(MaskedLinearSeq):
    """MaskedLinearSeq, but with more outputs than inputs and in a different order."""
    def forward(self, x, mask):
        dup = x.clone().detach() * 1.38  # just an arbitrary scaling
        x, mask = super().forward(x, mask)
        return dup, x, mask


HIDDEN_DIM = 20


def _mixed_mask(size=HIDDEN_DIM):
    entries = torch.randn(size)
    mask = torch.where(entries > 0, torch.ones(size), torch.zeros(size))
    mask = mask.bool()
    return mask


def _bool_to_float(btensor, dtype=torch.float32):
    """Converts a torch.BoolTensor to an equivalent dtype. """
    ones = torch.ones(size=btensor.size(), dtype=dtype)
    zeros = torch.zeros(size=btensor.size(), dtype=dtype)
    return torch.where(btensor, ones, zeros)


#
# Tests
#


def test_ckpt_inputs1_outputs1():
    module = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
    inputs = torch.rand(HIDDEN_DIM)
    inputs.requires_grad = True
    _test_activation_checkpoint(module, inputs)


# both bool and float are important, as bool is not diffentiable
@pytest.mark.parametrize('mask',
                         [
                             _mixed_mask(),
                             _bool_to_float(_mixed_mask()),
                         ])
def test_ckpt_inputs2_outputs1(mask):
    module = MaskedLinear(HIDDEN_DIM, HIDDEN_DIM)
    inputs = torch.rand(HIDDEN_DIM)
    inputs.requires_grad = True
    _test_activation_checkpoint(module, inputs, mask)


@pytest.mark.parametrize('mask',
                         [
                             _mixed_mask(),
                             _bool_to_float(_mixed_mask()),
                         ])
def test_ckpt_inputs2_outputs2(mask):
    module = MaskedLinearSeq(HIDDEN_DIM, HIDDEN_DIM)
    inputs = torch.rand(HIDDEN_DIM)
    inputs.requires_grad = True
    _test_activation_checkpoint(module, inputs, mask)


@pytest.mark.parametrize('mask',
                         [
                             _mixed_mask(),
                             _bool_to_float(_mixed_mask()),
                         ])
def test_ckpt_inputs2_outputs3(mask):
    module = MaskedLinearSeqDup(HIDDEN_DIM, HIDDEN_DIM)
    inputs = torch.rand(HIDDEN_DIM)
    inputs.requires_grad = True
    _test_activation_checkpoint(module, inputs, mask)


class DropMaskLinear(torch.nn.Linear):
    def forward(self, x, mask):
        return super().forward(x)


def test_ckpt_arg_none():
    module = DropMaskLinear(HIDDEN_DIM, HIDDEN_DIM)
    inputs = (torch.rand(HIDDEN_DIM), None)
    inputs[0].requires_grad = True
    _test_activation_checkpoint(module, *inputs)
