# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Tests for the attention stash (deepspeed/runtime/activation_checkpointing/attention_stash.py)."""

import gc

import pytest
import torch
from torch.nn.attention import SDPBackend

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.activation_checkpointing import attention_stash as stash_module
from deepspeed.runtime.activation_checkpointing.attention_stash import install_attention_stash
from deepspeed.utils import safe_get_full_grad
from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import make_autoep_config, seed_everything

transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.modeling_layers", reason="the attention stash needs GradientCheckpointingLayer")
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

NUM_LAYERS = 3


@pytest.fixture(autouse=True)
def restore_sdpa():
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    yield
    ALL_ATTENTION_FUNCTIONS["sdpa"] = original


def _expand(tensor, rep):
    return tensor.repeat_interleave(rep, dim=1)


def _cpu_forward(query, key, value, scale):
    rep = query.shape[1] // key.shape[1]
    output, logsumexp = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(query,
                                                                                   _expand(key, rep),
                                                                                   _expand(value, rep),
                                                                                   0.0,
                                                                                   True,
                                                                                   scale=scale)
    return output, logsumexp, None, None, query.shape[2], key.shape[2], torch.empty(0), torch.empty(0), None


def _cpu_backward(grad, query, key, value, output, logsumexp, seed, offset, cum_q, cum_k, max_q, max_k, scale):
    rep = query.shape[1] // key.shape[1]
    grad_query, grad_key, grad_value = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward(
        grad, query, _expand(key, rep), _expand(value, rep), output, logsumexp, 0.0, True, scale=scale)
    batch, kv_heads, seq, dim = key.shape
    return (grad_query, grad_key.view(batch, kv_heads, rep, seq,
                                      dim).sum(2), grad_value.view(batch, kv_heads, rep, seq, dim).sum(2))


@pytest.fixture
def cpu_cudnn(monkeypatch):
    """Stand in for the cuDNN SDPA calls with the CPU flash kernels, counting the kept and replayed outputs."""
    calls = {"kept": 0, "replayed": 0}

    def forward(*args):
        calls["kept"] += 1
        return _cpu_forward(*args)

    def backward(*args):
        calls["replayed"] += 1
        return _cpu_backward(*args)

    monkeypatch.setattr(stash_module, "_cudnn_forward", forward)
    monkeypatch.setattr(stash_module, "_cudnn_backward", backward)
    monkeypatch.setattr(stash_module, "_sdp_backend", lambda *args: SDPBackend.CUDNN_ATTENTION.value)
    return calls


def _tiny_model(checkpointing="reentrant", attn_implementation="sdpa", dtype=torch.float32, device="cpu", **sizes):
    seed_everything(0)
    config = transformers.MixtralConfig(vocab_size=64,
                                        hidden_size=sizes.get("hidden_size", 32),
                                        intermediate_size=64,
                                        num_hidden_layers=sizes.get("num_layers", NUM_LAYERS),
                                        num_attention_heads=4,
                                        num_key_value_heads=2,
                                        max_position_embeddings=256,
                                        num_local_experts=4,
                                        num_experts_per_tok=2,
                                        tie_word_embeddings=False,
                                        use_cache=False,
                                        attn_implementation=attn_implementation)
    model = transformers.MixtralForCausalLM(config).to(device=device, dtype=dtype).train()
    if checkpointing is not None:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": checkpointing == "reentrant"})
    return model


def _batch(seed, seq_len=24):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 64, (2, seq_len), generator=generator)


def _loss_and_grads(model, input_ids):
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    return loss.detach(), {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


def _assert_close(reference, candidate):
    ref_loss, ref_grads = reference
    loss, grads = candidate
    torch.testing.assert_close(loss, ref_loss, rtol=0, atol=1e-6)
    assert grads.keys() == ref_grads.keys()
    for name, grad in grads.items():
        # The CPU stand-in sums the expanded key/value gradients itself, so this is not bitwise.
        torch.testing.assert_close(grad, ref_grads[name], rtol=1e-4, atol=1e-6, msg=name)


def _stashes(model):
    return [
        layer.self_attn._deepspeed_attention_stash for layer in model.model.layers
        if hasattr(layer.self_attn, "_deepspeed_attention_stash")
    ]


class TestAttentionStashCPU:

    @pytest.mark.parametrize("num_layers", [1, NUM_LAYERS])
    def test_matches_the_normal_recompute(self, cpu_cudnn, num_layers):
        input_ids = _batch(0)
        reference = _loss_and_grads(_tiny_model(), input_ids)
        model = _tiny_model()
        assert install_attention_stash(model, num_layers) == list(range(NUM_LAYERS - num_layers, NUM_LAYERS))
        _assert_close(reference, _loss_and_grads(model, input_ids))
        assert cpu_cudnn == {"kept": num_layers, "replayed": num_layers}
        assert all(not stash.entries for stash in _stashes(model))

    def test_each_recompute_takes_its_own_forward(self, cpu_cudnn):
        batches = [_batch(1), _batch(2)]
        reference_model = _tiny_model()
        reference_losses = [reference_model(input_ids=ids, labels=ids).loss for ids in batches]
        for loss in reversed(reference_losses):
            loss.backward()
        reference = {name: p.grad.clone() for name, p in reference_model.named_parameters() if p.grad is not None}

        model = _tiny_model()
        install_attention_stash(model, NUM_LAYERS)
        losses = [model(input_ids=ids, labels=ids).loss for ids in batches]
        assert all(len(stash.entries) == 2 for stash in _stashes(model))
        # Backward in the opposite order: each recompute must find its own forward's output by key.
        for loss in reversed(losses):
            loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                torch.testing.assert_close(param.grad, reference[name], rtol=1e-4, atol=1e-6, msg=name)
        assert cpu_cudnn == {"kept": 2 * NUM_LAYERS, "replayed": 2 * NUM_LAYERS}
        assert all(not stash.entries for stash in _stashes(model))

    def test_forwards_without_a_recompute_retain_nothing(self, cpu_cudnn):
        model = _tiny_model()
        install_attention_stash(model, NUM_LAYERS)
        input_ids = _batch(3)
        with torch.no_grad():
            model(input_ids=input_ids, labels=input_ids)
        # Reentrant checkpointing makes the embedding output require grad even under no_grad, so the first layer
        # keeps an output; it is released with that input.
        assert cpu_cudnn["kept"] == 1
        model.eval()
        model(input_ids=input_ids, labels=input_ids)
        assert cpu_cudnn["kept"] == 1
        gc.collect()
        assert all(not stash.entries for stash in _stashes(model))
        model.train()
        # A forward whose graph is dropped without a backward releases what it kept.
        loss = model(input_ids=input_ids, labels=input_ids).loss
        assert all(len(stash.entries) == 1 for stash in _stashes(model))
        del loss
        gc.collect()
        assert all(not stash.entries for stash in _stashes(model))

    def test_install_rejects_unsupported_models(self):
        with pytest.raises(ValueError, match="attn_implementation='sdpa'"):
            install_attention_stash(_tiny_model(attn_implementation="eager"), 1)
        for num_layers in (0, NUM_LAYERS + 1):
            with pytest.raises(ValueError, match="model has 3 decoder layers"):
                install_attention_stash(_tiny_model(), num_layers)
        for num_layers in (True, "2", 1.5):
            with pytest.raises(ValueError, match="needs an integer num_layers"):
                install_attention_stash(_tiny_model(), num_layers)
        model = _tiny_model()
        install_attention_stash(model, 1)
        with pytest.raises(ValueError, match="already installed"):
            install_attention_stash(model, 1)

    @pytest.mark.parametrize("checkpointing", [None, "non_reentrant"])
    def test_training_outside_a_reentrant_checkpoint_fails(self, cpu_cudnn, checkpointing):
        model = _tiny_model(checkpointing=checkpointing)
        install_attention_stash(model, 1)
        input_ids = _batch(4)
        with pytest.raises(RuntimeError, match="outside a reentrant checkpoint"):
            model(input_ids=input_ids, labels=input_ids).loss.backward()

    def test_padding_mask_fails(self, cpu_cudnn):
        model = _tiny_model()
        install_attention_stash(model, 1)
        input_ids = _batch(5)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, :4] = 0
        with pytest.raises(RuntimeError, match="without an attention mask"):
            model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

    def test_other_sdpa_backends_fail(self, cpu_cudnn, monkeypatch):
        monkeypatch.setattr(stash_module, "_sdp_backend", lambda *args: SDPBackend.MATH.value)
        model = _tiny_model()
        install_attention_stash(model, 1)
        input_ids = _batch(6)
        with pytest.raises(RuntimeError, match="selects SDPA backend"):
            model(input_ids=input_ids, labels=input_ids)

    def test_recompute_without_its_forward_fails(self, cpu_cudnn):
        model = _tiny_model()
        install_attention_stash(model, 1)
        input_ids = _batch(7)
        loss = model(input_ids=input_ids, labels=input_ids).loss
        for stash in _stashes(model):
            stash.entries.clear()
        with pytest.raises(RuntimeError, match="found no attention output kept for its input"):
            loss.backward()

    def test_install_requires_the_pytorch_entry_points(self, monkeypatch):
        monkeypatch.delattr(torch._C, "_current_graph_task_id")
        with pytest.raises(ValueError, match="_current_graph_task_id"):
            install_attention_stash(_tiny_model(), 1)


def _skip_unless_cudnn_sdpa():
    if get_accelerator().device_name() != "cuda" or not torch.backends.cudnn.is_available():
        pytest.skip("stash_attention_layers keeps cuDNN SDPA outputs")


def _rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


class TestAttentionStashCuDNN:

    def test_replay_matches_autograd(self):
        _skip_unless_cudnn_sdpa()
        device = get_accelerator().current_device_name()
        generator = torch.Generator(device=device).manual_seed(0)

        def hf_layout(heads):
            return torch.randn(2, 256, heads, 64, device=device, dtype=torch.bfloat16,
                               generator=generator).transpose(1, 2)

        query, key, value = hf_layout(8), hf_layout(2), hf_layout(2)
        scale = 64**-0.5
        grad = torch.randn(2, 256, 8, 64, device=device, dtype=torch.bfloat16, generator=generator)
        assert stash_module._sdp_backend(query, key, value, scale) == SDPBackend.CUDNN_ATTENTION.value

        def run(replay):
            leaves = [tensor.clone().requires_grad_() for tensor in (query, key, value)]
            if replay:
                kept = (*stash_module._cudnn_forward(query, key, value, scale)[:8], scale)
                output = stash_module._StashedAttention.apply(*leaves, kept)
            else:
                output = torch.nn.functional.scaled_dot_product_attention(*leaves,
                                                                          is_causal=True,
                                                                          scale=scale,
                                                                          enable_gqa=True)
            output.transpose(1, 2).contiguous().backward(grad)
            return output.detach(), [leaf.grad for leaf in leaves]

        out_ref, grads_ref = run(replay=False)
        out, grads = run(replay=True)
        assert torch.equal(out, out_ref)
        # cuDNN accumulates dQ non-deterministically; dK and dV are deterministic.
        assert torch.equal(grads[1], grads_ref[1]) and torch.equal(grads[2], grads_ref[2])
        reference = torch.nn.functional.scaled_dot_product_attention(query.float(),
                                                                     _expand(key.float(), 4),
                                                                     _expand(value.float(), 4),
                                                                     is_causal=True,
                                                                     scale=scale)
        assert _rel_l2(out, reference) < 1e-2
        leaves = [tensor.float().requires_grad_() for tensor in (query, key, value)]
        reference = torch.nn.functional.scaled_dot_product_attention(leaves[0],
                                                                     _expand(leaves[1], 4),
                                                                     _expand(leaves[2], 4),
                                                                     is_causal=True,
                                                                     scale=scale)
        reference.transpose(1, 2).contiguous().backward(grad.float())
        assert _rel_l2(grads[0], leaves[0].grad) <= 1.02 * _rel_l2(grads_ref[0], leaves[0].grad) + 1e-6


class TestAttentionStashEngine(DistributedTest):
    world_size = 1

    def test_autoep_training_step_matches_without_the_stash(self):
        _skip_unless_cudnn_sdpa()
        results = {}
        for stash_layers in (0, 2):
            model = _tiny_model(dtype=torch.bfloat16,
                                device=get_accelerator().current_device_name(),
                                hidden_size=256,
                                num_layers=2)
            if stash_layers:
                install_attention_stash(model, stash_layers)
            config = make_autoep_config(ep_size=1)
            config.pop("fp16", None)
            config["bf16"] = {"enabled": True}
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            stashes = [layer.self_attn for layer in engine.module.model.layers]
            assert all(hasattr(attn, "_deepspeed_attention_stash") == bool(stash_layers) for attn in stashes)
            input_ids = _batch(8, seq_len=128).to(engine.device)
            loss = engine(input_ids=input_ids, labels=input_ids).loss
            engine.backward(loss)
            grads = {
                name: safe_get_full_grad(p).float().clone()
                for name, p in engine.module.named_parameters() if "self_attn" in name
            }
            assert all(not attn._deepspeed_attention_stash.entries for attn in stashes if stash_layers)
            results[stash_layers] = (loss.detach().float(), grads)
            engine.destroy()
        (base_loss, base_grads), (loss, grads) = results[0], results[2]
        assert torch.equal(loss, base_loss)
        assert grads.keys() == base_grads.keys() and grads
        for name, grad in grads.items():
            # Only cuDNN's non-deterministic dQ accumulation differs.
            assert _rel_l2(grad, base_grads[name]) < 1e-2, name
