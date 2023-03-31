# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import torch
from deepspeed.runtime.zero.tiling import TiledLinear, TiledLinearReturnBias

import pytest


@pytest.mark.parametrize('in_splits,out_splits', [(1, 1), (2, 2), (5, 5), (32, 32)])
def test_tiled_init(in_splits, out_splits):
    in_f = 32
    out_f = 40
    base = torch.nn.Linear(in_f, out_f, bias=True)
    l = TiledLinear(in_f,
                    out_f,
                    bias=True,
                    init_linear=copy.deepcopy(base),
                    out_splits=out_splits,
                    in_splits=in_splits)

    for out_id in range(out_splits):
        for in_id in range(in_splits):
            local_l = l.linears[out_id][in_id]
            assert isinstance(local_l, torch.nn.Linear)

            rstart = l.out_parts[out_id]
            rstop = l.out_parts[out_id + 1]
            cstart = l.in_parts[in_id]
            cstop = l.in_parts[in_id + 1]

            local_out = rstop - rstart
            local_in = cstop - cstart
            assert local_l.weight.size()[1] == local_in, f'local[{out_id}][{in_id}].size {local_l.weight.size()}'
            assert local_l.weight.size()[0] == local_out

            test = base.weight[rstart:rstop, cstart:cstop]

            assert local_l.weight.size() == test.size()
            assert torch.equal(local_l.weight.data, test.data)

            if in_id == in_splits - 1:
                assert local_l.bias is not None
                assert local_l.bias.size()[0] == local_out
            else:
                assert local_l.bias is None


@pytest.mark.parametrize('in_splits,out_splits', [(0, 0), (33, 33)])
def test_tiled_baddim(in_splits, out_splits):
    dim = 32
    with pytest.raises(RuntimeError):
        l = TiledLinear(dim, dim, out_splits=out_splits, in_splits=in_splits)


@pytest.mark.skip(reason="seeing nondeterministic failures, skipping for now")
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('in_splits,out_splits', [(1, 1), (2, 2)])
@pytest.mark.parametrize('in_f,out_f', [(32, 32), (23, 29), (29, 23)])
def test_tiled_forward(in_splits, out_splits, bias, in_f, out_f):
    base = torch.nn.Linear(in_f, out_f, bias=bias)
    test = TiledLinear(in_f,
                       out_f,
                       bias=bias,
                       init_linear=copy.deepcopy(base),
                       out_splits=out_splits,
                       in_splits=in_splits)

    inp = torch.rand(in_f)

    base_out = base(copy.deepcopy(inp))
    test_out = test(copy.deepcopy(inp))

    assert torch.allclose(base_out, test_out, rtol=1e-4)


@pytest.mark.skip(reason="seeing nondeterministic failures, skipping for now")
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('in_splits,out_splits', [(1, 1), (2, 2)])
@pytest.mark.parametrize('in_f,out_f', [(32, 32), (23, 29), (29, 23)])
def test_tiled_backward(in_splits, out_splits, bias, in_f, out_f):
    base = torch.nn.Linear(in_f, out_f, bias=bias)
    test = TiledLinear(in_f,
                       out_f,
                       bias=bias,
                       init_linear=copy.deepcopy(base),
                       out_splits=out_splits,
                       in_splits=in_splits)

    inp = torch.rand(in_f)

    base_out = base(copy.deepcopy(inp))
    test_out = test(copy.deepcopy(inp))
    assert torch.allclose(base_out, test_out, rtol=1e-4)

    base_out.sum().backward()
    test_out.sum().backward()

    # compare grads
    for row in range(out_splits):
        rstart = test.out_parts[row]
        rstop = test.out_parts[row + 1]

        for col in range(in_splits):
            cstart = test.in_parts[col]
            cstop = test.in_parts[col + 1]

            local = test.linears[row][col]
            base_grad = base.weight.grad[rstart:rstop, cstart:cstop]
            assert torch.allclose(base_grad, local.weight.grad, rtol=1e-4)

            if local.bias is not None:
                base_grad = base.bias.grad[rstart:rstop]
                assert torch.allclose(base_grad, local.bias.grad, rtol=1e-4)


class LinearWrapper(torch.nn.Linear):
    """Returns its own bias to simulate Megatron-LM's behavior.

    Megatron-LM optionally delays the bias addition to fuse with a proceeding kernel.
    """

    def forward(self, input):
        out = super().forward(input)
        return out, self.bias


@pytest.mark.skip(reason="seeing nondeterministic failures, skipping for now")
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('in_splits,out_splits', [(1, 1), (2, 2)])
@pytest.mark.parametrize('in_f,out_f', [(32, 32), (23, 29), (29, 23)])
def test_tiled_returnbias_backward(in_splits, out_splits, bias, in_f, out_f):
    base = LinearWrapper(in_f, out_f, bias=bias)
    test = TiledLinearReturnBias(in_f,
                                 out_f,
                                 bias=bias,
                                 linear_cls=LinearWrapper,
                                 init_linear=copy.deepcopy(base),
                                 out_splits=out_splits,
                                 in_splits=in_splits)

    inp = torch.rand(in_f)

    base_out_t, base_out_b = base(copy.deepcopy(inp))
    test_out_t, test_out_b = test(copy.deepcopy(inp))
    assert torch.allclose(base_out_t, test_out_t, rtol=1e-4)
    if base_out_b is None:
        assert test_out_b is None
        base_out_b = torch.zeros_like(base_out_t)
        test_out_b = torch.zeros_like(test_out_t)
    else:
        assert test_out_b is not None
        assert torch.allclose(base_out_b, test_out_b, rtol=1e-4)

    (base_out_t + base_out_b).sum().backward()
    (test_out_t + test_out_b).sum().backward()

    # compare grads
    for row in range(out_splits):
        rstart = test.out_parts[row]
        rstop = test.out_parts[row + 1]

        for col in range(in_splits):
            cstart = test.in_parts[col]
            cstop = test.in_parts[col + 1]

            local = test.linears[row][col]
            base_grad = base.weight.grad[rstart:rstop, cstart:cstop]
            assert torch.allclose(base_grad, local.weight.grad, rtol=1e-4)

            if local.bias is not None:
                base_grad = base.bias.grad[rstart:rstop]
                assert torch.allclose(base_grad, local.bias.grad, rtol=1e-4)
