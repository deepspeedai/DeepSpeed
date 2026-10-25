# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""The gather-based AutoEP expert reorder against the zero-row-and-index reference, compared with torch.equal."""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.auto_ep_layer import permute_by_local_expert, unpermute_by_local_expert
from deepspeed.moe import ep_kernels
from unit.common import DistributedTest


def _device():
    return get_accelerator().device_name()


def _reference_permute(tokens, permuted_indices):
    padded = torch.vstack((tokens, tokens.new_zeros((tokens.shape[-1], ))))
    return padded[permuted_indices, :]


def _reference_unpermute(expert_output, permuted_indices, n_tokens):
    out = expert_output.new_zeros((n_tokens + 1, expert_output.shape[-1]))
    out[permuted_indices, :] = expert_output
    return out[:-1]


def _counts(ep_degree, num_local_experts, seed, empty_experts=()):
    generator = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, 40, (ep_degree, num_local_experts), generator=generator, dtype=torch.int32)
    for expert in empty_experts:
        counts[:, expert] = 0
    return counts.to(_device())


def _grads(function, inputs, upstream):
    leaf = inputs.detach().clone().requires_grad_(True)
    output = function(leaf)
    (grad, ) = torch.autograd.grad(output, leaf, upstream)
    return output.detach(), grad


@pytest.mark.skipif(get_accelerator().device_name() != "cuda" or not ep_kernels._TRITON_AVAILABLE,
                    reason="the gather-based reorder needs CUDA and Triton")
class TestAutoEPPermuteRows(DistributedTest):
    # DeepSpeed imports can initialize CUDA during collection; the runner starts clean workers even under --forked.
    world_size = 1
    init_distributed = False

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("hidden", [2048, 100, 1])
    @pytest.mark.parametrize("ep_degree, num_local_experts, empty", [(16, 8, ()), (4, 8, (0, 5)), (1, 3, (1, ))])
    def test_permute_and_unpermute_match_the_reference(self, dtype, hidden, ep_degree, num_local_experts, empty):
        counts = _counts(ep_degree, num_local_experts, seed=hidden + ep_degree, empty_experts=empty)
        local_counts = counts if ep_degree > 1 else counts.view(-1)
        n_tokens = int(counts.sum().item())
        generator = torch.Generator().manual_seed(7)
        tokens = torch.randn(n_tokens, hidden, generator=generator).to(_device()).to(dtype)

        permuted, permuted_indices, aligned_counts, rows = permute_by_local_expert(tokens, local_counts)
        assert rows == n_tokens
        upstream = torch.randn(permuted.shape, generator=generator).to(_device()).to(dtype)

        actual = _grads(lambda leaf: permute_by_local_expert(leaf, local_counts)[0], tokens, upstream)
        expected = _grads(lambda leaf: _reference_permute(leaf, permuted_indices), tokens, upstream)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])

        expert_output = torch.randn(permuted.shape, generator=generator).to(_device()).to(dtype)
        back_upstream = torch.randn(n_tokens, hidden, generator=generator).to(_device()).to(dtype)
        actual = _grads(lambda leaf: unpermute_by_local_expert(leaf, permuted_indices, n_tokens), expert_output,
                        back_upstream)
        expected = _grads(lambda leaf: _reference_unpermute(leaf, permuted_indices, n_tokens), expert_output,
                          back_upstream)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_strided_rows_and_upstream_gradients(self, dtype):
        counts = _counts(4, 8, seed=13, empty_experts=(6, ))
        n_tokens = int(counts.sum().item())
        hidden = 96
        generator = torch.Generator().manual_seed(9)
        # Each function sees every row through a view that skips half the buffer's columns, and the upstream gradients
        # are column-major, as a transposed product gives them.
        tokens = torch.randn(n_tokens, 2 * hidden, generator=generator).to(_device()).to(dtype)
        permuted, permuted_indices, _, _ = permute_by_local_expert(tokens[:, :hidden], counts)
        upstream = torch.randn(hidden, permuted.shape[0], generator=generator).to(_device()).to(dtype).t()

        actual = _grads(lambda leaf: permute_by_local_expert(leaf[:, :hidden], counts)[0], tokens, upstream)
        expected = _grads(lambda leaf: _reference_permute(leaf[:, :hidden], permuted_indices), tokens, upstream)
        for a, e in zip(actual, expected):
            assert torch.equal(a, e) and a.stride() == e.stride()

        expert_output = torch.randn(permuted.shape[0], 2 * hidden, generator=generator).to(_device()).to(dtype)
        back_upstream = torch.randn(hidden, n_tokens, generator=generator).to(_device()).to(dtype).t()
        actual = _grads(lambda leaf: unpermute_by_local_expert(leaf[:, :hidden], permuted_indices, n_tokens),
                        expert_output, back_upstream)
        expected = _grads(lambda leaf: _reference_unpermute(leaf[:, :hidden], permuted_indices, n_tokens),
                          expert_output, back_upstream)
        for a, e in zip(actual, expected):
            assert torch.equal(a, e) and a.stride() == e.stride()

    @pytest.mark.parametrize("direction", ["permute", "unpermute"])
    def test_double_backward_matches_the_reference(self, direction):
        counts = _counts(4, 8, seed=11, empty_experts=(3, ))
        n_tokens = int(counts.sum().item())
        tokens = torch.randn(n_tokens, 48, dtype=torch.float32).to(_device())
        permuted, permuted_indices, _, _ = permute_by_local_expert(tokens, counts)
        inputs = tokens
        functions = (lambda leaf: permute_by_local_expert(leaf, counts)[0],
                     lambda leaf: _reference_permute(leaf, permuted_indices))
        if direction == "unpermute":
            inputs = torch.randn(permuted.shape, generator=torch.Generator().manual_seed(4)).to(_device())
            functions = (lambda leaf: unpermute_by_local_expert(leaf, permuted_indices, n_tokens),
                         lambda leaf: _reference_unpermute(leaf, permuted_indices, n_tokens))
        results = []
        for function in functions:
            leaf = inputs.clone().requires_grad_(True)
            output = function(leaf)
            upstream = torch.randn(output.shape,
                                   generator=torch.Generator().manual_seed(5)).to(_device()).requires_grad_(True)
            (grad, ) = torch.autograd.grad(output, leaf, upstream, create_graph=True)
            weight = torch.randn(grad.shape, generator=torch.Generator().manual_seed(6)).to(_device())
            (second, ) = torch.autograd.grad(grad, upstream, weight)
            results.append(second)
        assert torch.equal(results[0], results[1])

    def test_round_trip_restores_the_rows_and_padding_is_zero(self):
        counts = _counts(16, 8, seed=3, empty_experts=(2, ))
        n_tokens = int(counts.sum().item())
        tokens = torch.randn(n_tokens, 64).to(_device()).to(torch.bfloat16)
        permuted, permuted_indices, _, _ = permute_by_local_expert(tokens, counts)
        padding = permuted_indices < 0
        assert padding.any()
        assert torch.count_nonzero(permuted[padding]) == 0
        assert torch.equal(unpermute_by_local_expert(permuted, permuted_indices, n_tokens), tokens)

    def test_no_tokens_at_all(self):
        counts = torch.zeros(4, 8, dtype=torch.int32).to(_device())
        tokens = torch.empty(0, 32).to(_device()).to(torch.bfloat16).requires_grad_(True)
        permuted, permuted_indices, _, n_tokens = permute_by_local_expert(tokens, counts)
        assert n_tokens == 0 and torch.count_nonzero(permuted) == 0
        restored = unpermute_by_local_expert(permuted, permuted_indices, n_tokens)
        assert restored.shape == (0, 32)
        permuted.sum().backward()
        assert tokens.grad.shape == (0, 32)

    @pytest.mark.parametrize("strided_index", [False, True], ids=["contiguous-index", "strided-index"])
    def test_gather_rows_treats_both_padding_encodings_as_zero(self, strided_index):
        src = torch.arange(12, dtype=torch.float32).view(4, 3).to(_device())
        index = torch.tensor([2, -1, 4, 0], dtype=torch.int32).to(_device())
        if strided_index:
            # The same entries at every other position; reading the ones in between would pick row 1.
            index = torch.tensor([2, 1, -1, 1, 4, 1, 0, 1], dtype=torch.int32).to(_device())[::2]
        out = ep_kernels.gather_rows(src, index)
        assert torch.equal(out.cpu(), torch.tensor([[6., 7., 8.], [0., 0., 0.], [0., 0., 0.], [0., 1., 2.]]))


def test_host_tensors_keep_the_reference_path():
    counts = torch.tensor([[3, 0], [2, 1]], dtype=torch.int32)
    tokens = torch.randn(6, 5)
    assert not ep_kernels.permute_rows_supported(tokens)
    permuted, permuted_indices, _, n_tokens = permute_by_local_expert(tokens, counts)
    assert torch.equal(permuted, _reference_permute(tokens, permuted_indices))
    assert torch.equal(unpermute_by_local_expert(permuted, permuted_indices, n_tokens), tokens)
