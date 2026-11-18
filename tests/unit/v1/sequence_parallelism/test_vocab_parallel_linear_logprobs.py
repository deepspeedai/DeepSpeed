# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.sequence.linear_logprobs import vocab_parallel_linear_logprobs
from deepspeed.sequence.vocab_parallel_ops import vocab_parallel_entropy, vocab_parallel_logprobs
from deepspeed.utils import safe_get_full_grad
from unit.common import DistributedTest


def _leaf(tensor):
    return tensor.detach().clone().requires_grad_(True)


def _inputs(device="cpu", dtype=torch.float32):
    torch.manual_seed(31)
    hidden = torch.randn(2, 6, 8, device=device).to(dtype)[:, :-1]
    weight = torch.randn(19, 8, device=device).mul_(0.2).to(dtype)
    bias = torch.randn(19, device=device).mul_(0.2).to(dtype)
    targets = torch.tensor([[0, -100, 8, 15, 18], [18, 12, 0, 7, 3]], device=device)
    mask = torch.tensor([[True, True, False, True, True], [True, False, True, False, True]], device=device)
    return hidden, weight, bias, targets, mask


def _reference(hidden, weight, bias, targets, mask, temperature=1.0):
    valid = mask & (targets != -100)
    # Select before projection: masked activations may deliberately contain NaNs.
    logits = F.linear(hidden[valid], weight, bias).float() / temperature
    log_p = F.log_softmax(logits, dim=-1)
    selected = log_p.gather(-1, targets[valid].unsqueeze(-1)).squeeze(-1)
    entropy = -(log_p.exp() * log_p).nan_to_num().sum(-1)
    output = log_p.new_zeros(targets.shape)
    return output.masked_scatter(valid, selected), output.masked_scatter(valid, entropy)


@pytest.mark.parametrize("with_entropy", [False, True])
@pytest.mark.parametrize("frozen", ["neither", "hidden", "head"])
@pytest.mark.parametrize("tiles", [(2, 3), (64, 128)])
def test_streamed_outputs_and_gradients_match_dense(with_entropy, frozen, tiles):
    hidden, weight, bias, targets, mask = _inputs()
    hidden[~(mask & (targets != -100))] = float("nan")
    actual_args = [_leaf(tensor) for tensor in (hidden, weight, bias)]
    reference_args = [_leaf(tensor) for tensor in (hidden, weight, bias)]
    for args in (actual_args, reference_args):
        args[0].requires_grad_(frozen != "hidden")
        for tensor in args[1:]:
            tensor.requires_grad_(frozen != "head")
    actual = vocab_parallel_linear_logprobs(*actual_args[:2],
                                            targets,
                                            bias=actual_args[2],
                                            position_mask=mask,
                                            temperature=0.7,
                                            return_entropy=with_entropy,
                                            chunk_size=tiles[0],
                                            vocab_chunk_size=tiles[1])
    expected = _reference(*reference_args, targets, mask, temperature=0.7)
    multipliers = [torch.randn_like(expected[0]), torch.randn_like(expected[1])]
    actual_loss, reference_loss = 0, 0
    for i in range(2 if with_entropy else 1):
        assert actual[i].dtype == torch.float32
        torch.testing.assert_close(actual[i], expected[i], atol=2e-6, rtol=2e-5)
        actual_loss = actual_loss + (actual[i] * multipliers[i]).sum()
        reference_loss = reference_loss + (expected[i] * multipliers[i]).sum()
    if not with_entropy:
        assert actual[1] is None
    actual_loss.backward()
    reference_loss.backward()
    for got, want in zip(actual_args, reference_args):
        if got.requires_grad:
            torch.testing.assert_close(got.grad, want.grad, atol=2e-6, rtol=2e-5)
        else:
            assert got.grad is None


@pytest.mark.parametrize("empty", [False, True])
def test_all_masked_and_empty_inputs_have_zero_outputs_and_gradients(empty):
    hidden, weight, bias, targets, mask = _inputs()
    if empty:
        hidden, targets, mask = hidden[:, :0], targets[:, :0], mask[:, :0]
    hidden = _leaf(torch.full_like(hidden, float("nan")))
    weight, bias = _leaf(weight), _leaf(bias)
    logprobs, entropy = vocab_parallel_linear_logprobs(hidden,
                                                       weight,
                                                       targets,
                                                       bias=bias,
                                                       position_mask=torch.zeros_like(mask),
                                                       return_entropy=True)
    assert torch.equal(logprobs, torch.zeros_like(logprobs))
    assert torch.equal(entropy, torch.zeros_like(entropy))
    (logprobs.sum() + entropy.sum()).backward()
    for tensor in (hidden, weight, bias):
        assert torch.equal(tensor.grad, torch.zeros_like(tensor))


def test_entropy_only_backward_with_masked_vocabulary():
    hidden = _leaf(torch.randn(4, 3))
    weight = _leaf(torch.randn(7, 3))
    bias = _leaf(torch.tensor([-float("inf")] * 3 + [0.0] * 4))
    targets, mask = torch.full((4, ), 4), torch.ones(4, dtype=torch.bool)
    args = [_leaf(tensor) for tensor in (hidden, weight, bias)]
    _, entropy = vocab_parallel_linear_logprobs(hidden,
                                                weight,
                                                targets,
                                                bias=bias,
                                                return_entropy=True,
                                                vocab_chunk_size=2)
    _, expected = _reference(*args, targets, mask)
    torch.testing.assert_close(entropy, expected)
    entropy.sum().backward()
    # Avoid the reference's undefined derivative at exp(-inf) * -inf.
    reference_logits = F.linear(*args)[..., 3:]
    log_p = F.log_softmax(reference_logits, -1)
    (-(log_p.exp() * log_p).sum()).backward()
    for actual, expected in zip((hidden, weight, bias), args):
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)


def test_bias_free_and_all_undefined_rows():
    hidden = _leaf(torch.randn(3, 4))
    weight = _leaf(torch.randn(5, 4))
    target = torch.tensor([0, 2, 4])
    logprobs, _ = vocab_parallel_linear_logprobs(hidden, weight, target, vocab_chunk_size=2)
    torch.testing.assert_close(logprobs, F.log_softmax(F.linear(hidden, weight), -1)[torch.arange(3), target])
    bias = _leaf(torch.full((5, ), -float("inf")))
    lp, ent = vocab_parallel_linear_logprobs(hidden,
                                             weight,
                                             target,
                                             bias=bias,
                                             return_entropy=True,
                                             vocab_chunk_size=2)
    assert torch.isnan(lp).all() and torch.isnan(ent).all()
    (lp.masked_fill(torch.ones_like(lp, dtype=torch.bool), 0).sum() +
     ent.masked_fill(torch.ones_like(ent, dtype=torch.bool), 0).sum()).backward()
    for tensor in (hidden, weight, bias):
        assert torch.equal(tensor.grad, torch.zeros_like(tensor))


@pytest.mark.parametrize("kwargs,error", [
    ({
        "temperature": 0.0
    }, ValueError),
    ({
        "temperature": float("nan")
    }, ValueError),
    ({
        "chunk_size": 0
    }, ValueError),
    ({
        "vocab_chunk_size": 1.5
    }, ValueError),
    ({
        "position_mask": torch.ones(2, 5)
    }, ValueError),
    ({
        "vocab_start_index": 0
    }, ValueError),
])
def test_invalid_scoring_options_fail_explicitly(kwargs, error):
    hidden, weight, _, targets, _ = _inputs()
    with pytest.raises(error):
        vocab_parallel_linear_logprobs(hidden, weight, targets, **kwargs)


def test_autocast_scores_and_gradients():
    hidden, weight, bias, targets, mask = _inputs()
    args = [_leaf(tensor) for tensor in (hidden.bfloat16(), weight, bias)]
    refs = [_leaf(tensor) for tensor in (hidden.bfloat16(), weight, bias)]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, _ = vocab_parallel_linear_logprobs(*args[:2],
                                                   targets,
                                                   bias=args[2],
                                                   position_mask=mask,
                                                   vocab_chunk_size=4)
        expected, _ = _reference(*refs, targets, mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.sum().backward()
    expected.sum().backward()
    for got, want in zip(args, refs):
        torch.testing.assert_close(got.grad, want.grad, atol=2e-2, rtol=2e-2)


class _ScoringModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(19, 8)
        self.backbone = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 19)

    def forward(self, tokens, targets, mask):
        hidden = self.backbone(self.embedding(tokens)).tanh()
        return vocab_parallel_linear_logprobs(hidden,
                                              self.head.weight,
                                              targets,
                                              bias=self.head.bias,
                                              position_mask=mask,
                                              return_entropy=True,
                                              chunk_size=3,
                                              vocab_chunk_size=4)


class TestStreamedScoringEngine(DistributedTest):
    world_size = 1

    def test_engine_optimizer_updates_match_dense(self):
        device = get_accelerator().current_device_name()
        torch.manual_seed(19)
        model = _ScoringModel().to(device)
        reference = _ScoringModel().to(device)
        reference.load_state_dict(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=optimizer,
                                               config={
                                                   "train_batch_size": 2,
                                                   "zero_optimization": {
                                                       "stage": 0
                                                   }
                                               })
        _, _, _, targets, mask = _inputs(device)
        tokens = targets.clamp_min(0)
        advantage = torch.randn_like(targets, dtype=torch.float32)
        for _ in range(3):
            lp, entropy = engine(tokens, targets, mask)
            hidden = reference.backbone(reference.embedding(tokens)).tanh()
            ref_lp, ref_entropy = _reference(hidden, reference.head.weight, reference.head.bias, targets, mask)
            loss = -(lp * advantage).mean() - 0.01 * entropy.mean()
            ref_loss = -(ref_lp * advantage).mean() - 0.01 * ref_entropy.mean()
            torch.testing.assert_close(loss, ref_loss)
            engine.backward(loss)
            ref_loss.backward()
            for got, want in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(got.grad, want.grad, atol=2e-6, rtol=2e-4)
            engine.step()
            ref_optimizer.step()
            ref_optimizer.zero_grad()
            for got, want in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
        engine.destroy()


class TestStreamedScoringZero2(DistributedTest):
    world_size = 2

    def test_accumulated_dp_gradients_and_updates_match_dense(self):
        device = get_accelerator().current_device_name()
        torch.manual_seed(19)
        model = _ScoringModel().to(device)
        reference = deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=optimizer,
                                               config={
                                                   "train_micro_batch_size_per_gpu": 2,
                                                   "gradient_accumulation_steps": 2,
                                                   "gradient_clipping": 0.0,
                                                   "zero_allow_untested_optimizer": True,
                                                   "zero_optimization": {
                                                       "stage": 2,
                                                       "overlap_comm": True,
                                                       "contiguous_gradients": True
                                                   }
                                               })
        _, _, _, targets, mask = _inputs(device)
        advantage = torch.randn_like(targets, dtype=torch.float32)
        try:
            for step in range(3):
                for micro_step in range(2):
                    tokens = (targets.clamp_min(0) + dist.get_rank() + step + micro_step) % 19
                    score_mask = mask.clone()
                    if dist.get_rank() == 1 and micro_step == 1:
                        score_mask.zero_()
                    lp, entropy = engine(tokens, targets, score_mask)
                    hidden = reference.backbone(reference.embedding(tokens)).tanh()
                    ref_lp, ref_entropy = _reference(hidden, reference.head.weight, reference.head.bias, targets,
                                                     score_mask)
                    loss = -(lp * advantage).mean() - 0.01 * entropy.mean()
                    ref_loss = -(ref_lp * advantage).mean() - 0.01 * ref_entropy.mean()
                    torch.testing.assert_close(loss, ref_loss)
                    engine.backward(loss)
                    (ref_loss / 2).backward()
                    if micro_step == 0:
                        engine.step()
                for got, want in zip(model.parameters(), reference.parameters()):
                    dist.all_reduce(want.grad)
                    want.grad.div_(dist.get_world_size())
                    actual_grad = safe_get_full_grad(got)
                    assert actual_grad is not None
                    torch.testing.assert_close(actual_grad, want.grad, atol=2e-6, rtol=2e-4)
                engine.step()
                ref_optimizer.step()
                ref_optimizer.zero_grad()
                for got, want in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
        finally:
            engine.destroy()


class TestStreamedScoringAutoTP(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("zero_stage", [0, 2])
    def test_tied_hf_head_gradients_and_updates_match_dense(self, zero_stage):
        transformers = pytest.importorskip("transformers")
        from deepspeed.module_inject.layers import VocabParallelEmbedding, VocabParallelLinear

        class StreamedLlamaForCausalLM(transformers.LlamaForCausalLM):

            def forward(self, input_ids, position_mask):
                hidden = self.get_decoder()(input_ids=input_ids, use_cache=False).last_hidden_state[:, :-1]
                head = self.get_output_embeddings()
                return vocab_parallel_linear_logprobs(hidden,
                                                      head.weight,
                                                      input_ids[:, 1:],
                                                      bias=head.bias,
                                                      position_mask=position_mask,
                                                      tp_group=head.mp_group,
                                                      vocab_start_index=head.vocab_start_index,
                                                      vocab_end_index=head.vocab_end_index,
                                                      temperature=0.7,
                                                      return_entropy=True,
                                                      chunk_size=3,
                                                      vocab_chunk_size=5)

        torch.manual_seed(42)
        config = transformers.LlamaConfig(vocab_size=37,
                                          hidden_size=32,
                                          intermediate_size=64,
                                          num_hidden_layers=1,
                                          num_attention_heads=4,
                                          num_key_value_heads=4,
                                          head_dim=8,
                                          tie_word_embeddings=True,
                                          use_cache=False,
                                          attn_implementation="eager")
        config.base_model_tp_plan = None
        device = get_accelerator().current_device_name()
        model = StreamedLlamaForCausalLM(config).to(device)
        reference = deepcopy(model)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
                                               config={
                                                   "train_micro_batch_size_per_gpu": 2,
                                                   "gradient_clipping": 0.0,
                                                   "zero_allow_untested_optimizer": True,
                                                   "tensor_parallel": {
                                                       "autotp_size": 2,
                                                       "vocab_parallel_lm_head": True
                                                   },
                                                   "zero_optimization": {
                                                       "stage": zero_stage
                                                   }
                                               })
        head = model.get_output_embeddings()
        assert isinstance(head, VocabParallelLinear)
        assert isinstance(model.get_input_embeddings(), VocabParallelEmbedding)
        assert head.weight is model.get_input_embeddings().weight
        input_ids = torch.tensor([[3, 1, 19, 0, 36, 5, 2, 3], [4, 2, 20, 1, 35, 6, 3, 4]], device=device)
        mask = torch.tensor(
            [[False, False, True, False, True, True, True], [False, False, True, True, False, True, True]],
            device=device)
        advantage = torch.randn_like(mask, dtype=torch.float32)
        try:
            for _ in range(3):
                actual = engine(input_ids=input_ids, position_mask=mask)
                ref_hidden = reference.get_decoder()(input_ids=input_ids, use_cache=False).last_hidden_state[:, :-1]
                expected = _reference(ref_hidden,
                                      reference.lm_head.weight,
                                      reference.lm_head.bias,
                                      input_ids[:, 1:],
                                      mask,
                                      temperature=0.7)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
                loss = -(actual[0] * advantage).mean() - 0.01 * actual[1].mean()
                ref_loss = -(expected[0] * advantage).mean() - 0.01 * expected[1].mean()
                engine.backward(loss)
                ref_loss.backward()
                full_grad = safe_get_full_grad(head.weight).detach().clone()
                head.gather_params([full_grad])
                torch.testing.assert_close(full_grad, reference.lm_head.weight.grad, atol=2e-6, rtol=2e-4)
                torch.testing.assert_close(safe_get_full_grad(model.model.norm.weight),
                                           reference.model.norm.weight.grad,
                                           atol=2e-6,
                                           rtol=2e-4)
                engine.step()
                ref_optimizer.step()
                ref_optimizer.zero_grad()
                full_weight = head.weight.detach().clone()
                head.gather_params([full_weight])
                torch.testing.assert_close(full_weight, reference.lm_head.weight, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(model.model.norm.weight, reference.model.norm.weight, atol=2e-6, rtol=2e-5)
        finally:
            engine.destroy()


class TestStreamedScoringTP(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("empty_tokens", [False, True])
    def test_unscored_batches_have_zero_gradients_with_empty_shard(self, empty_tokens):
        hidden, weight, bias, targets, mask = _inputs(get_accelerator().current_device_name())
        if empty_tokens:
            hidden, targets, mask = hidden[:, :0], targets[:, :0], mask[:, :0]
        start, end = (0, 0) if dist.get_rank() == 0 else (0, 19)
        args = [_leaf(hidden), _leaf(weight[start:end]), _leaf(bias[start:end])]
        lp, entropy = vocab_parallel_linear_logprobs(*args[:2],
                                                     targets,
                                                     bias=args[2],
                                                     position_mask=torch.zeros_like(mask),
                                                     tp_group=dist.get_world_group(),
                                                     vocab_start_index=start,
                                                     vocab_end_index=end,
                                                     return_entropy=True)
        for output in (lp, entropy):
            assert torch.equal(output, torch.zeros_like(output))
        (lp.sum() + entropy.sum()).backward()
        for tensor in args:
            assert torch.equal(tensor.grad, torch.zeros_like(tensor))

    def test_invalid_target_fails_on_all_peers(self):
        hidden, weight, _, targets, mask = _inputs(get_accelerator().current_device_name())
        rank = dist.get_rank()
        if rank == 1:
            targets[0, 0] = 19
        start, end = (0, 7) if rank == 0 else (7, 19)
        with pytest.raises(ValueError, match="target is out of range"):
            vocab_parallel_linear_logprobs(hidden,
                                           weight[start:end],
                                           targets,
                                           position_mask=mask,
                                           tp_group=dist.get_world_group(),
                                           vocab_start_index=start,
                                           vocab_end_index=end)

    @pytest.mark.parametrize("empty_shard", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_scores_gradients_and_updates(self, empty_shard, dtype):
        device = get_accelerator().current_device_name()
        hidden, weight, bias, targets, mask = _inputs(device, dtype)
        rank, group = dist.get_rank(), dist.get_world_group()
        split = 0 if empty_shard else 7
        start, end = (0, split) if rank == 0 else (split, 19)
        args = [_leaf(hidden), _leaf(weight[start:end]), _leaf(bias[start:end])]
        refs = [_leaf(tensor.float()) for tensor in (hidden, weight, bias)]
        actual = vocab_parallel_linear_logprobs(*args[:2],
                                                targets,
                                                bias=args[2],
                                                position_mask=mask,
                                                tp_group=group,
                                                vocab_start_index=start,
                                                vocab_end_index=end,
                                                temperature=0.9,
                                                return_entropy=True,
                                                chunk_size=2,
                                                vocab_chunk_size=3)
        expected = _reference(*refs, targets, mask, temperature=0.9)
        atol, rtol = (2e-6, 2e-4) if dtype == torch.float32 else (1e-2, 3e-2)
        upstream = [torch.randn_like(expected[0]), torch.randn_like(expected[1])]
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=atol, rtol=rtol)
        sum((a * g).sum() for a, g in zip(actual, upstream)).backward()
        sum((a * g).sum() for a, g in zip(expected, upstream)).backward()
        expected_grads = [refs[0].grad, refs[1].grad[start:end], refs[2].grad[start:end]]
        for tensor, expected_grad in zip(args, expected_grads):
            torch.testing.assert_close(tensor.grad.float(), expected_grad, atol=atol, rtol=rtol)
            updated = tensor.detach().float() - 0.01 * tensor.grad.float()
            reference_updated = tensor.detach().float() - 0.01 * expected_grad
            torch.testing.assert_close(updated, reference_updated, atol=atol, rtol=rtol)
        if not empty_shard:
            local_logits = F.linear(hidden, weight[start:end], bias[start:end]).float() / 0.9
            ids = targets.masked_fill(~mask, -100)
            existing_lp = vocab_parallel_logprobs(local_logits,
                                                  ids,
                                                  tp_group=group,
                                                  vocab_start_index=start,
                                                  vocab_end_index=end)
            existing_entropy = vocab_parallel_entropy(local_logits,
                                                      tp_group=group).masked_fill(~(mask & (targets != -100)), 0)
            torch.testing.assert_close(actual[0], existing_lp, atol=atol, rtol=rtol)
            torch.testing.assert_close(actual[1], existing_entropy, atol=atol, rtol=rtol)

    def test_mask_mismatch_fails_on_all_peers(self):
        hidden, weight, bias, targets, mask = _inputs(get_accelerator().current_device_name())
        rank = dist.get_rank()
        if rank == 1:
            mask[0, 0] = False
        start, end = (0, 7) if rank == 0 else (7, 19)
        with pytest.raises(ValueError, match="identical targets and position masks"):
            vocab_parallel_linear_logprobs(hidden,
                                           weight[start:end],
                                           targets,
                                           bias=bias[start:end],
                                           position_mask=mask,
                                           tp_group=dist.get_world_group(),
                                           vocab_start_index=start,
                                           vocab_end_index=end)
