# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""DSA kernels and the GLM-5.2 attention replacement, against plain-PyTorch references and the
transformers module. CUDA and TileLang only."""

import copy

import pytest
import torch

from deepspeed.accelerator import get_accelerator

pytest.importorskip("tilelang")

if not (get_accelerator().is_available() and get_accelerator().device_name() == "cuda"):
    pytest.skip("the DSA kernels need a CUDA device", allow_module_level=True)

import deepspeed  # noqa: E402
from deepspeed.models.glm_moe_dsa.kernels import select_topk, sparse_mla  # noqa: E402
from deepspeed.models.glm_moe_dsa.kernels.indexer_fwd import indexer_logits  # noqa: E402
from unit.common import DistributedTest  # noqa: E402

LATENT, ROPE = 512, 64


def _device():
    return get_accelerator().current_device_name()


def _random_causal_selection(tokens, topk, device):
    """Per row a random subset of the causal keys, ``-1`` padded where fewer than ``topk`` exist."""
    scores = torch.rand(tokens, tokens, device=device)
    future = torch.arange(tokens, device=device)[None, :] > torch.arange(tokens, device=device)[:, None]
    scores = scores.masked_fill(future, float("-inf"))
    values, indices = scores.topk(topk, dim=-1)
    return indices.to(torch.int32).masked_fill(values == float("-inf"), -1)


def _reference_sparse_mla(q, kv, indices, scale):
    """The sparse attention written out: softmax over the selected keys only, float32 math."""
    safe = indices[:, 0, :].clamp(min=0).long()
    kv_selected = kv[:, 0, :][safe]  # [tokens, topk, latent + rope]
    scores = torch.einsum("thd,tkd->thk", q, kv_selected) * scale
    scores = scores.masked_fill((indices[:, 0, :] < 0)[:, None, :], float("-inf"))
    return torch.einsum("thk,tkd->thd", scores.softmax(dim=-1), kv_selected[..., :LATENT])


@pytest.mark.parametrize("heads", [8, 64])  # 8 exercises the kernel's head padding to 16; 64 is GLM-5.2
def test_sparse_mla_matches_reference(heads):
    tokens, topk, scale = 256, 64, 256**-0.5
    torch.manual_seed(0)
    device = _device()
    q = torch.randn(tokens, heads, LATENT + ROPE, device=device).to(torch.bfloat16).requires_grad_(True)
    kv = torch.randn(tokens, 1, LATENT + ROPE, device=device).to(torch.bfloat16).requires_grad_(True)
    indices = _random_causal_selection(tokens, topk, device).view(tokens, 1, topk)

    out = sparse_mla(q, kv, indices, scale, LATENT)
    q32 = q.detach().float().requires_grad_(True)
    kv32 = kv.detach().float().requires_grad_(True)
    ref = _reference_sparse_mla(q32, kv32, indices, scale)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)

    grad = torch.randn_like(ref)
    out.backward(grad.to(torch.bfloat16))
    ref.backward(grad)
    # bf16 kernel against float32 math; dKV is accumulated with atomic adds in float32 and rounded once.
    for got, want in ((q.grad, q32.grad), (kv.grad, kv32.grad)):
        torch.testing.assert_close(got.float(), want, rtol=3e-2, atol=3e-2 * want.abs().max().item())


@pytest.mark.parametrize("heads", [8, 32])  # 32 is GLM-5.2
def test_indexer_logits_and_selection_match_reference(heads):
    tokens, dim, topk = 256, 128, 64
    torch.manual_seed(0)
    device = _device()
    index_q = torch.randn(tokens, heads, dim, device=device).to(torch.bfloat16)
    index_k = torch.randn(tokens, dim, device=device).to(torch.bfloat16)
    weights = torch.rand(tokens, heads, device=device)
    positions = torch.arange(tokens, device=device, dtype=torch.int32)
    starts, ends = torch.zeros_like(positions), positions + 1

    logits = indexer_logits(index_q, index_k, weights, starts, ends)
    ref = torch.einsum("thd,kd->thk", index_q.float(), index_k.float()).relu()
    ref = torch.einsum("thk,th->tk", ref, weights)
    future = positions[None, :] > positions[:, None]
    ref = ref.masked_fill(future, float("-inf"))
    assert torch.equal(logits == float("-inf"), future)
    torch.testing.assert_close(logits[~future], ref[~future], rtol=1e-2, atol=1e-2)

    selected = select_topk(index_q, index_k, weights, starts, ends, topk, block_rows=128)
    assert selected.shape == (tokens, topk)
    valid = selected >= 0
    assert torch.equal(valid.sum(dim=-1), torch.clamp(positions + 1, max=topk))
    assert bool((selected[valid] <= positions[:, None].expand_as(selected)[valid]).all())
    # The selection is the reference top-k wherever the k-th and (k+1)-th scores are clearly apart.
    ref_sorted = ref.sort(dim=-1, descending=True).values
    clear = ref_sorted[:, topk - 1] - ref_sorted[:, topk] > 1e-3
    clear &= positions >= topk
    ref_topk = ref.topk(topk, dim=-1).indices
    for row in clear.nonzero().flatten().tolist():
        assert set(selected[row].tolist()) == set(ref_topk[row].tolist()), row
    assert clear.sum() > tokens // 4


def _tiny_config(transformers, dense_mlps=False):
    """A 2-layer GLM-5.2 config. ``dense_mlps`` gives every layer a dense MLP, for tests of the attention alone:
    the MoE backward can hang in torch 2.8's ``_grouped_mm`` when an expert receives no tokens. It must be set when
    the config is built: the config derives ``mlp_layer_types`` from ``first_k_dense_replace`` in its constructor,
    so changing ``first_k_dense_replace`` afterwards has no effect."""
    layers = 2
    config = transformers.GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=layers,
        first_k_dense_replace=layers if dense_mlps else 1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        q_lora_rank=128,
        kv_lora_rank=LATENT,
        qk_nope_head_dim=192,
        qk_rope_head_dim=ROPE,
        v_head_dim=256,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=64,
        indexer_types=["full", "shared"],
        max_position_embeddings=1024,
        attention_bias=False,
        tie_word_embeddings=False,
        use_cache=False,
    )
    if dense_mlps:
        assert config.mlp_layer_types == ["dense"] * layers, config.mlp_layer_types
    return config


def test_replacement_matches_transformers_module():
    transformers = pytest.importorskip("transformers")
    modeling = pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
    from deepspeed.models.glm_moe_dsa import DeepSpeedGlmMoeDsaAttention

    torch.manual_seed(0)
    device, dtype = _device(), torch.bfloat16
    config = _tiny_config(transformers)
    config._attn_implementation = "sdpa"
    tokens, topk = 256, config.index_topk
    rotary = modeling.GlmMoeDsaRotaryEmbedding(config).to(device)
    x = torch.randn(1, tokens, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(tokens, device=device)[None]
    cos, sin = rotary(x, position_ids)

    # Layer 0 runs the indexer: the two selections must agree except at ties.
    stock_full = modeling.GlmMoeDsaAttention(config, 0).to(device, dtype)
    ref_out, _, ref_topk = stock_full(x, (cos, sin), None, position_ids=position_ids)
    fast_full = DeepSpeedGlmMoeDsaAttention(stock_full)
    out, _, got_topk = fast_full(x, (cos, sin), None, position_ids=position_ids)
    assert out.shape == ref_out.shape and got_topk.shape == ref_topk.shape
    full_rows = slice(topk - 1, None)  # rows that have at least topk keys, where -1 padding plays no part
    overlap = [
        len(set(a) & set(b)) / topk for a, b in zip(got_topk[0, full_rows].tolist(), ref_topk[0, full_rows].tolist())
    ]
    assert sum(overlap) / len(overlap) > 0.9, sum(overlap) / len(overlap)

    # Layer 1 reuses a given selection: with the same indices the outputs and input gradients must
    # agree, on the rows whose selection is complete (the stock module encodes the earliest rows'
    # missing keys through its causal mask, this module through -1, so those rows are left out).
    stock_shared = modeling.GlmMoeDsaAttention(config, 1).to(device, dtype)
    x_ref = x.clone().requires_grad_(True)
    x_new = x.clone().requires_grad_(True)
    ref_out, _, _ = stock_shared(x_ref, (cos, sin), None, position_ids=position_ids, prev_topk_indices=ref_topk)
    fast_shared = DeepSpeedGlmMoeDsaAttention(stock_shared)
    out, _, _ = fast_shared(x_new, (cos, sin), None, position_ids=position_ids, prev_topk_indices=ref_topk)
    scale = ref_out[:, full_rows].abs().max().item()
    torch.testing.assert_close(out[:, full_rows], ref_out[:, full_rows], rtol=2e-2, atol=2e-2 * scale)
    ref_out[:, full_rows].sum().backward()
    # The two modules share their parameters: keep the stock gradients, then let the replacement
    # write its own, and compare every parameter's gradient as well as the input's.
    ref_grads = {name: p.grad.clone() for name, p in stock_shared.named_parameters()}
    stock_shared.zero_grad()
    out[:, full_rows].sum().backward()
    grad_scale = x_ref.grad.abs().max().item()
    torch.testing.assert_close(x_new.grad, x_ref.grad, rtol=3e-2, atol=3e-2 * grad_scale)
    for name, p in stock_shared.named_parameters():
        want = ref_grads[name]
        torch.testing.assert_close(p.grad, want, rtol=3e-2, atol=3e-2 * want.abs().max().item(), msg=name)


def test_replace_walks_the_model_and_keeps_parameter_names():
    transformers = pytest.importorskip("transformers")
    from deepspeed.models.glm_moe_dsa import DeepSpeedGlmMoeDsaAttention, replace_attention

    torch.manual_seed(0)
    config = _tiny_config(transformers)
    config._attn_implementation = "sdpa"
    model = transformers.GlmMoeDsaForCausalLM(config)
    names_before = sorted(name for name, _ in model.named_parameters())
    assert replace_attention(model) == config.num_hidden_layers
    assert all(isinstance(layer.self_attn, DeepSpeedGlmMoeDsaAttention) for layer in model.model.layers)
    assert sorted(name for name, _ in model.named_parameters()) == names_before


def _replaced_model(transformers, attn_implementation):
    from deepspeed.models.glm_moe_dsa import replace_attention

    torch.manual_seed(0)
    config = _tiny_config(transformers, dense_mlps=True)
    config._attn_implementation = attn_implementation
    model = transformers.GlmMoeDsaForCausalLM(config).to(_device(), torch.bfloat16)
    replace_attention(model)
    return model


def test_unsupported_inputs_raise():
    """The kernels compute causal attention over whole sequences and never form attention weights: a padding mask,
    packed sequences or output_attentions must fail loudly instead of returning something else."""
    transformers = pytest.importorskip("transformers")
    model = _replaced_model(transformers, "sdpa")
    tokens = 256
    input_ids = torch.randint(0, model.config.vocab_size, (1, tokens), device=_device())

    with pytest.raises(NotImplementedError):
        model(input_ids=input_ids, output_attentions=True)
    padding = torch.ones_like(input_ids)
    padding[:, -16:] = 0
    with pytest.raises(NotImplementedError):
        model(input_ids=input_ids, attention_mask=padding)
    packed = torch.cat([torch.arange(tokens // 2), torch.arange(tokens // 2)]).to(_device())[None]
    with pytest.raises(NotImplementedError):
        model(input_ids=input_ids, position_ids=packed)

    # Eager attention makes transformers build the full causal mask, which is what the kernels compute.
    with torch.no_grad():
        want = model(input_ids=input_ids).logits
        model.config._attn_implementation = "eager"
        got = model(input_ids=input_ids).logits
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_kernel_widths_below_four_are_rejected():
    """Widths 1 and 2 are powers of two, but the backward kernel splits both widths into groups of 4."""
    transformers = pytest.importorskip("transformers")
    modeling = pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
    from deepspeed.models.glm_moe_dsa import DeepSpeedGlmMoeDsaAttention

    config = _tiny_config(transformers)
    config.qk_rope_head_dim = 2
    with pytest.raises(ValueError):
        DeepSpeedGlmMoeDsaAttention(modeling.GlmMoeDsaAttention(config, 0))


def test_first_forward_under_activation_checkpointing(monkeypatch):
    """The kernel warm-up runs inside the first forward. Under activation checkpointing that forward is recomputed
    without the warm-up, so the warm-up must not leave tensors saved for backward: the checkpoint compares the
    number saved in the forward and in its recomputation."""
    transformers = pytest.importorskip("transformers")
    from deepspeed.models.glm_moe_dsa import attention

    monkeypatch.setattr(attention, "_WARMED_UP", set())  # earlier tests warmed up these shapes on this device
    model = _replaced_model(transformers, "sdpa")
    # Non-reentrant checkpointing is the one that compares the saved tensors of the two passes.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 256), device=_device())
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.skipif(get_accelerator().device_count() < 2, reason="needs two GPUs")
def test_warm_up_runs_on_the_input_device():
    """replace_attention() runs before deepspeed.initialize(), while the current device may still be GPU 0 in every
    local process: the warm-up must not allocate there when the model runs on another GPU."""
    transformers = pytest.importorskip("transformers")
    from deepspeed.models.glm_moe_dsa import replace_attention

    torch.manual_seed(0)
    config = _tiny_config(transformers, dense_mlps=True)
    config._attn_implementation = "sdpa"
    model = transformers.GlmMoeDsaForCausalLM(config).to("cuda:1", torch.bfloat16)
    input_ids = torch.randint(0, config.vocab_size, (1, 256), device="cuda:1")
    get_accelerator().set_device(0)
    try:
        allocations_before = get_accelerator().memory_stats(0).get("allocation.all.allocated", 0)
        replace_attention(model)
        get_accelerator().set_device(1)  # as deepspeed.initialize() does for this process's local rank
        model(input_ids=input_ids).logits.sum().backward()
        assert get_accelerator().memory_stats(0).get("allocation.all.allocated", 0) == allocations_before
    finally:
        get_accelerator().set_device(0)


class TestDsaAttentionZeroStage3(DistributedTest):
    """The absorbed form reads ``kv_b_proj.weight`` as a tensor and never calls ``kv_b_proj``, so
    ZeRO-3 gets no forward hook for that weight: without the external-parameter registration the
    weight is a 0-element placeholder in the forward and its gradient is never reduced."""
    world_size = 2

    def test_loss_and_external_weight_grad_match_single_process(self):
        transformers = pytest.importorskip("transformers")
        from deepspeed.models.glm_moe_dsa import replace_attention
        from deepspeed.utils import safe_get_full_grad

        config = _tiny_config(transformers, dense_mlps=True)
        config._attn_implementation = "sdpa"
        device = _device()
        torch.manual_seed(0)  # the same weights on every rank and in the reference
        model = transformers.GlmMoeDsaForCausalLM(config).to(torch.bfloat16)
        replace_attention(model)
        generator = torch.Generator().manual_seed(1)
        input_ids = torch.randint(0, config.vocab_size, (1, 256), generator=generator).to(device)

        reference = copy.deepcopy(model).to(device)
        ref_loss = reference(input_ids=input_ids, labels=input_ids).loss
        ref_loss.backward()
        ref_grad = reference.model.layers[0].self_attn.kv_b_proj.weight.grad.float()

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 3
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3,
                    "torch_adam": True
                }
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=model.parameters())
        loss = engine(input_ids=input_ids, labels=input_ids).loss
        engine.backward(loss)
        grad = safe_get_full_grad(engine.module.model.layers[0].self_attn.kv_b_proj.weight)
        # Every rank sees the same batch, so the averaged gradient equals the single-process one.
        torch.testing.assert_close(loss.float(), ref_loss.float(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(grad.float(), ref_grad, rtol=3e-2, atol=3e-2 * ref_grad.abs().max().item())
        engine.step()


class TestDsaAttentionContextParallel(DistributedTest):
    """Each sequence split over the ranks (context parallel) must give the outputs and gradients of the whole
    sequence on one rank. index_topk equals the sequence length, so every query keeps all its causal keys and the
    selection has no near-ties: what is compared is the gathered keys, the global causal bounds of the indexer (a
    wrong position lets a query see future keys) and the reduce-scattered key gradients."""
    world_size = 2

    def test_split_matches_whole_sequence(self):
        transformers = pytest.importorskip("transformers")
        import deepspeed.comm as dist
        from deepspeed.models.glm_moe_dsa import replace_attention, set_context_parallel_group

        tokens = 256
        config = _tiny_config(transformers, dense_mlps=True)
        config._attn_implementation = "sdpa"
        config.index_topk = tokens
        device = _device()
        torch.manual_seed(0)  # the same weights on every rank
        model = transformers.GlmMoeDsaForCausalLM(config).to(device, torch.bfloat16)
        replace_attention(model)
        split = copy.deepcopy(model)
        input_ids = torch.randint(0, config.vocab_size, (1, tokens), generator=torch.Generator().manual_seed(1))
        input_ids = input_ids.to(device)
        predicted = tokens - 1

        def loss_sum(module, ids, positions, labels):
            logits = module(input_ids=ids, position_ids=positions).logits
            rows = labels.shape[1]
            loss = torch.nn.functional.cross_entropy(logits[0, :rows].float(), labels[0], reduction="sum")
            return logits, loss

        ref_logits, ref_sum = loss_sum(model, input_ids, torch.arange(tokens, device=device)[None], input_ids[:, 1:])
        (ref_sum / predicted).backward()

        rank, size = dist.get_rank(), dist.get_world_size()
        local = tokens // size
        start = rank * local
        set_context_parallel_group(dist.new_group(list(range(size))))
        try:
            logits, local_sum = loss_sum(split, input_ids[:, start:start + local],
                                         torch.arange(start, start + local, device=device)[None],
                                         input_ids[:, start + 1:start + local + 1])
            (local_sum / predicted).backward()
        finally:
            set_context_parallel_group(None)

        # Every collective first: a rank that fails an assertion must not leave the other waiting in one.
        total = local_sum.detach().clone()
        dist.all_reduce(total)
        grads = {}
        for name, p in split.named_parameters():
            if p.grad is not None:
                grads[name] = p.grad.float().clone()
                dist.all_reduce(grads[name])

        torch.testing.assert_close(logits, ref_logits[:, start:start + local], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(total, ref_sum.detach(), rtol=1e-3, atol=0)
        ref_grads = {name: p.grad.float() for name, p in model.named_parameters() if p.grad is not None}
        assert grads.keys() == ref_grads.keys()
        for name, want in ref_grads.items():
            torch.testing.assert_close(grads[name], want, rtol=3e-2, atol=3e-2 * want.abs().max().item(), msg=name)


GLM52_HEADS, GLM52_INDEX_HEADS, GLM52_INDEX_DIM, GLM52_TOPK = 64, 32, 128, 2048


def test_sparse_mla_at_glm52_sizes():
    """64 heads, top-k 2048 (32 index blocks of 64 per row) and 16,384 tokens, the shapes of a GLM-5.2 run.
    The float32 reference is evaluated on 256 random query rows: a row's output and ``dq`` depend on that
    row alone, and ``dkv`` is checked by sending a gradient into those rows only."""
    tokens, rows, scale = 16384, 256, (LATENT + ROPE)**-0.5
    torch.manual_seed(0)
    device = _device()
    q = torch.randn(tokens, GLM52_HEADS, LATENT + ROPE, device=device).to(torch.bfloat16).requires_grad_(True)
    kv = torch.randn(tokens, 1, LATENT + ROPE, device=device).to(torch.bfloat16).requires_grad_(True)
    indices = _random_causal_selection(tokens, GLM52_TOPK, device).view(tokens, 1, GLM52_TOPK)
    subset = torch.randperm(tokens, device=device)[:rows].sort().values

    out = sparse_mla(q, kv, indices, scale, LATENT)
    q32 = q.detach()[subset].float().requires_grad_(True)
    kv32 = kv.detach().float().requires_grad_(True)
    ref = _reference_sparse_mla(q32, kv32, indices[subset], scale)
    torch.testing.assert_close(out[subset].float(), ref, rtol=2e-2, atol=2e-2)

    grad = torch.randn_like(ref).to(torch.bfloat16)
    grad_out = torch.zeros_like(out)
    grad_out[subset] = grad
    out.backward(grad_out)
    ref.backward(grad.float())
    torch.testing.assert_close(q.grad[subset].float(), q32.grad, rtol=3e-2, atol=3e-2 * q32.grad.abs().max().item())
    torch.testing.assert_close(kv.grad.float(), kv32.grad, rtol=3e-2, atol=3e-2 * kv32.grad.abs().max().item())


def test_indexer_selection_at_glm52_sizes():
    """32 index heads (4 query rows per kernel program), 16,384 tokens (two 8,192-row blocks in
    ``select_topk``) and top-k 2048. On 256 random rows with a full selection: exactly k distinct causal
    keys, each scoring at least the k-th best float32 score minus a rounding allowance, so no better key
    was left out."""
    tokens, rows = 16384, 256
    torch.manual_seed(0)
    device = _device()
    index_q = torch.randn(tokens, GLM52_INDEX_HEADS, GLM52_INDEX_DIM, device=device).to(torch.bfloat16)
    index_k = torch.randn(tokens, GLM52_INDEX_DIM, device=device).to(torch.bfloat16)
    weights = torch.rand(tokens, GLM52_INDEX_HEADS, device=device)
    positions = torch.arange(tokens, device=device, dtype=torch.int32)
    starts, ends = torch.zeros_like(positions), positions + 1

    selected = select_topk(index_q, index_k, weights, starts, ends, GLM52_TOPK, block_rows=8192)
    assert selected.shape == (tokens, GLM52_TOPK)
    valid = selected >= 0
    assert torch.equal(valid.sum(dim=-1), torch.clamp(positions + 1, max=GLM52_TOPK))
    assert bool((selected[valid] <= positions[:, None].expand_as(selected)[valid]).all())

    subset = torch.randperm(tokens, device=device)[:rows]
    subset = subset[subset >= GLM52_TOPK]
    picks = selected[subset].long()
    assert bool((picks.sort(dim=-1).values.diff(dim=-1) > 0).all()), "a key was selected twice"
    ref = torch.einsum("thd,kd->thk", index_q[subset].float(), index_k.float()).relu()
    ref = torch.einsum("thk,th->tk", ref, weights[subset])
    ref = ref.masked_fill(positions[None, :] > positions[subset][:, None], float("-inf"))
    kth_best = ref.topk(GLM52_TOPK, dim=-1).values[:, -1:]
    allowance = 1e-3 * ref.masked_fill(ref == float("-inf"), 0).abs().amax(dim=-1, keepdim=True)
    assert bool((ref.gather(1, picks) >= kth_best - allowance).all())
