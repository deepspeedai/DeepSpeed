# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import deepspeed
import deepspeed.comm as dist
import deepspeed.sequence.cross_entropy as cross_entropy
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.layers import VocabParallelLinear
from deepspeed.module_inject.tp_shard import get_shard_size_list
from deepspeed.utils import groups
from unit.common import DistributedTest
from unit.v1.autotp.test_autotp_training import reset_tp_model_init_state


def _require_liger():
    pytest.importorskip("liger_kernel.ops.vocab_parallel_cross_entropy")
    if not get_accelerator().is_triton_supported():
        pytest.skip("Liger CE requires a Triton-supported accelerator")


def _record_liger_calls(monkeypatch, calls):
    original = cross_entropy._liger_vocab_cross_entropy

    def record(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(cross_entropy, "_liger_vocab_cross_entropy", record)


def test_liger_without_tp_group_uses_pytorch_reference():
    logits = torch.randn(3, 18, requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    target = torch.tensor([0, -100, 17])
    actual = cross_entropy.vocab_parallel_cross_entropy(logits, target, backend="liger")
    expected = F.cross_entropy(reference, target)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad)


def test_vocab_parallel_causal_lm_loss_forwards_backend():
    assert cross_entropy.VocabParallelCausalLMLoss().backend == "torch"
    assert cross_entropy.VocabParallelCausalLMLoss(backend="liger").backend == "liger"


class LigerOutputTrainingModel(nn.Module):

    def __init__(self, vocabulary):
        super().__init__()
        self.projection = nn.Linear(32, 32)
        self.lm_head = nn.Linear(32, vocabulary)
        self.loss_function = cross_entropy.VocabParallelCausalLMLoss()

    def forward(self, inputs, labels):
        logits = self.lm_head(self.projection(inputs).tanh())
        return logits, self.loss_function(logits, labels=labels)


class TestLigerBackendSelection(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def test_missing_tp_group_uses_pytorch_reference(self, monkeypatch):
        """A None group is the default world to some Liger versions, so it must not select Liger."""
        device = get_accelerator().current_device_name()
        calls = []
        _record_liger_calls(monkeypatch, calls)
        torch.manual_seed(8173)
        logits = torch.randn(2, 3, 17, device=device, requires_grad=True)
        reference = logits.detach().clone().requires_grad_(True)
        target = torch.tensor([[0, 4, 16], [9, -100, 3]], device=device)
        actual = cross_entropy.vocab_parallel_cross_entropy(logits, target, backend="liger")
        expected = F.cross_entropy(reference.reshape(-1, reference.shape[-1]), target.reshape(-1))
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(logits.grad, reference.grad)
        assert calls == []

    def test_tp_group_selects_liger_only_for_equal_shards(self, monkeypatch):
        _require_liger()
        device = get_accelerator().current_device_name()
        rank = dist.get_rank()
        group = dist.new_group(ranks=list(range(self.world_size)))
        target = torch.tensor([[0, 4, 16], [9, -100, 3]], device=device)
        for vocabulary in (18, 17):
            calls = []
            _record_liger_calls(monkeypatch, calls)
            torch.manual_seed(8173)
            full = torch.randn(2, 3, vocabulary, device=device) * 20
            sizes = [vocabulary // 2, vocabulary - vocabulary // 2]
            start = sum(sizes[:rank])
            end = start + sizes[rank]
            local = full[..., start:end].detach().contiguous().requires_grad_(True)
            reference = full.detach().float().clone().requires_grad_(True)
            expected = F.cross_entropy(reference.reshape(-1, vocabulary), target.reshape(-1))
            actual = cross_entropy.vocab_parallel_cross_entropy(local,
                                                                target,
                                                                group,
                                                                vocab_start_index=start,
                                                                vocab_end_index=end,
                                                                backend="liger")
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            assert len(calls) == (1 if vocabulary % 2 == 0 else 0)


class TestLigerVocabParallelCE(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def test_losses_and_gradients(self):
        _require_liger()
        device = get_accelerator().current_device_name()
        rank = dist.get_rank()
        for vocabulary in (34, 35):
            # Each layer layout owns a distinct process group; metadata is cached
            # for a fixed layout throughout a training run.
            group = dist.new_group(ranks=list(range(self.world_size)))
            sizes = [vocabulary // 2, vocabulary - vocabulary // 2]
            start = sum(sizes[:rank])
            end = start + sizes[rank]
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for reduction in ("none", "sum", "mean"):
                    torch.manual_seed(8173)
                    full = (torch.randn(2, 3, vocabulary, device=device) * 20).to(dtype)
                    reference = full.detach().float().clone().requires_grad_(True)
                    local = full[..., start:end].detach().contiguous().requires_grad_(True)
                    before = local.detach().clone()
                    target = torch.tensor([[0, -100, vocabulary - 1], [1, 17, 8]], device=device)
                    expected = F.cross_entropy(reference.reshape(-1, vocabulary),
                                               target.reshape(-1),
                                               reduction=reduction)
                    if reduction == "none":
                        expected = expected.reshape_as(target)
                    actual = cross_entropy.vocab_parallel_cross_entropy(local,
                                                                        target,
                                                                        group,
                                                                        vocab_start_index=start,
                                                                        vocab_end_index=end,
                                                                        reduction=reduction,
                                                                        backend="liger")
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                    actual.sum().backward()
                    expected.sum().backward()
                    torch.testing.assert_close(local.grad,
                                               reference.grad[..., start:end].to(dtype),
                                               atol=2e-5,
                                               rtol=2e-3)
                    torch.testing.assert_close(local.detach(), before, atol=0, rtol=0)
        group = dist.new_group(ranks=list(range(self.world_size)))
        logits = torch.randn(2, 3, 17, device=device, requires_grad=True)
        ignored = torch.full((2, 3), -100, device=device)
        loss = cross_entropy.vocab_parallel_cross_entropy(logits, ignored, group, backend="liger")
        torch.testing.assert_close(loss, torch.zeros_like(loss))
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        logits = torch.randn(2, 3, 17, device=device, requires_grad=True)
        loss = cross_entropy.vocab_parallel_cross_entropy(logits, torch.zeros_like(ignored), group, backend="liger")
        loss.backward(retain_graph=True)
        with pytest.raises(RuntimeError, match="one first-order backward"):
            loss.backward()
        logits = torch.randn(2, 3, 17, device=device, requires_grad=True)
        loss = cross_entropy.vocab_parallel_cross_entropy(logits, torch.zeros_like(ignored), group, backend="liger")
        with pytest.raises(RuntimeError, match="one first-order backward"):
            torch.autograd.grad(loss, logits, create_graph=True)

    @pytest.mark.parametrize("vocabulary", [34, 35])
    def test_five_optimizer_steps(self, vocabulary):
        _require_liger()
        reset_tp_model_init_state()
        device = get_accelerator().current_device_name()
        torch.manual_seed(8173)
        model = LigerOutputTrainingModel(vocabulary).to(device)
        reference = deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 2,
            "gradient_clipping": 0.0,
            "zero_optimization": {
                "stage": 0
            },
            "tensor_parallel": {
                "autotp_size": 2,
                "vocab_parallel_lm_head": True,
                "vocab_parallel_ce_backend": "liger",
                "partition_config": {
                    "use_default_specs": False,
                    "layer_specs": []
                }
            }
        }
        try:
            engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
            head = engine.module.lm_head
            assert isinstance(head, VocabParallelLinear)
            assert engine.module.loss_function.backend == "liger"
            rank = dist.get_rank(groups.get_tensor_model_parallel_group())
            sizes = get_shard_size_list(vocabulary, 2, head.tp_meta, "lm_head")
            start = sum(sizes[:rank])
            for step in range(5):
                reference_optimizer.zero_grad()
                for micro in range(2):
                    torch.manual_seed(100 + step * 2 + micro)
                    inputs = torch.randn(2, 6, 32, device=device)
                    labels = torch.randint(vocabulary, (2, 6), device=device)
                    labels[0, 2] = -100
                    reference_logits, expected = reference(inputs, labels)
                    logits, actual = engine(inputs, labels)
                    torch.testing.assert_close(logits, reference_logits[..., start:start + sizes[rank]])
                    torch.testing.assert_close(actual, expected)
                    (expected / 2).backward()
                    engine.backward(actual)
                    torch.testing.assert_close(head.weight.grad,
                                               reference.lm_head.weight.grad[start:start + sizes[rank]],
                                               atol=1e-6,
                                               rtol=1e-5)
                    torch.testing.assert_close(engine.module.projection.weight.grad,
                                               reference.projection.weight.grad,
                                               atol=1e-6,
                                               rtol=1e-5)
                    engine.step()
                reference_optimizer.step()
                assert engine.global_steps == step + 1
                torch.testing.assert_close(head.weight, reference.lm_head.weight[start:start + sizes[rank]])
                torch.testing.assert_close(head.bias, reference.lm_head.bias[start:start + sizes[rank]])
                torch.testing.assert_close(engine.module.projection.weight, reference.projection.weight)
                torch.testing.assert_close(engine.module.projection.bias, reference.projection.bias)
                if rank == 0:
                    print(f"vocab={vocabulary} step={step + 1} loss={actual.item():.6f} reference-aligned", flush=True)
        finally:
            reset_tp_model_init_state()
