# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""MiniMax-M3 block-sparse attention (the Triton kernels and the PyTorch path) against a dense
masked-attention reference and against the transformers ``minimax_m3_vl`` modules, with and without the
sequence split over a context-parallel group."""

import copy

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.models.minimax_m3_vl import block_sparse_attention, selected_attention
from deepspeed.ops.triton_ops import is_triton_available
from unit.common import DistributedTest


def _cuda_available():
    return get_accelerator().is_available() and get_accelerator().device_name() == "cuda"


def _triton_available():
    return _cuda_available() and is_triton_available() and get_accelerator().is_triton_supported()


def _random_selection(batch, kv_heads, seq_len, block_size, topk, generator, always=(), offset=0):
    """Per query, ``topk`` distinct blocks from its causal past, always including its own block and
    the blocks in ``always``; ``-1`` where fewer exist. The contract of MiniMax-M3's indexer output.
    The queries are positions ``offset .. offset + seq_len - 1``."""
    num_blocks = -(-(offset + seq_len) // block_size)
    own = (offset + torch.arange(seq_len)) // block_size
    scores = torch.rand(batch, kv_heads, seq_len, num_blocks, generator=generator)
    scores = scores.masked_fill(torch.arange(num_blocks)[None, :] > own[:, None], float("-inf"))
    scores.scatter_(-1, own.view(1, 1, -1, 1).expand(batch, kv_heads, -1, 1), float("inf"))
    for block in always:
        scores[..., block] = torch.where(own[:, None] >= block, float("inf"), float("-inf")).squeeze(-1)
    values, blocks = scores.topk(min(topk, num_blocks), dim=-1)
    return blocks.masked_fill(values == float("-inf"), -1)


def _dense_reference(q, k, v, block_indices, positions, block_size, scale):
    """Dense float32 attention under the mask transformers' ``build_block_mask`` builds from the same
    selection: a key is visible if its block is selected and its position is at most the query's."""
    batch, heads, seq_len, _ = q.shape
    kv_heads, kv_len = k.shape[1], k.shape[2]
    group = heads // kv_heads
    num_blocks = -(-kv_len // block_size)
    safe = block_indices.masked_fill(block_indices < 0, num_blocks)
    keep = torch.zeros(batch, kv_heads, seq_len, num_blocks + 1, dtype=torch.bool, device=q.device)
    keep.scatter_(-1, safe, True)
    keep = keep[..., :num_blocks].repeat_interleave(block_size, dim=-1)[..., :kv_len]
    keep = keep & (torch.arange(kv_len, device=q.device)[None, None, None, :] <= positions[:, None, :, None])
    keep = keep.repeat_interleave(group, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float().repeat_interleave(group, dim=1)) * scale
    probs = scores.masked_fill(~keep, float("-inf")).softmax(dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", probs, v.float().repeat_interleave(group, dim=1))
    return out.transpose(1, 2)


def _selected(q, k, v, blocks, positions, block_size, scale, query_chunk):
    """:func:`selected_attention` with :func:`block_sparse_attention`'s layout; its queries are the last
    positions of the keys."""
    return selected_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), blocks.permute(0, 2, 1, 3),
                              block_size, scale)


def _run_both(batch,
              heads,
              kv_heads,
              seq_len,
              dim,
              block_size,
              topk,
              query_chunk,
              device,
              dtype,
              always=(),
              kv_len=None,
              attention=block_sparse_attention):
    """``attention`` and the dense float32 reference on the same inputs; the queries are the last ``seq_len``
    of ``kv_len`` positions."""
    kv_len = kv_len or seq_len
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(batch, heads, seq_len, dim, generator=generator).to(device, dtype)
    k, v = (torch.randn(batch, kv_heads, kv_len, dim, generator=generator).to(device, dtype) for _ in range(2))
    offset = kv_len - seq_len
    blocks = _random_selection(batch, kv_heads, seq_len, block_size, topk, generator, always, offset).to(device)
    positions = torch.arange(offset, kv_len, device=device).expand(batch, -1)
    grad = torch.randn(batch, seq_len, heads, dim, generator=generator).to(device)
    scale = dim**-0.5

    inputs = [t.clone().requires_grad_(True) for t in (q, k, v)]
    out = attention(*inputs, blocks, positions, block_size, scale, query_chunk)
    out.backward(grad.to(dtype))
    reference_inputs = [t.detach().float().requires_grad_(True) for t in (q, k, v)]
    reference = _dense_reference(*reference_inputs, blocks, positions, block_size, scale)
    reference.backward(grad)
    got = [out] + [t.grad for t in inputs]
    want = [reference] + [t.grad for t in reference_inputs]
    return got, want


@pytest.mark.parametrize("query_chunk", [64, 1000])
def test_matches_dense_reference_float32(query_chunk):
    # 203 tokens: not a multiple of the block size or of the chunk. 16 query heads on 4 key-value heads.
    got, want = _run_both(batch=2,
                          heads=16,
                          kv_heads=4,
                          seq_len=203,
                          dim=32,
                          block_size=16,
                          topk=3,
                          query_chunk=query_chunk,
                          device="cpu",
                          dtype=torch.float32)
    for name, a, b in zip(("out", "dq", "dk", "dv"), got, want):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5, msg=name)


@pytest.mark.skipif(not _cuda_available(), reason="bf16 matmuls need a CUDA device")
def test_matches_dense_reference_bf16():
    got, want = _run_both(batch=1,
                          heads=16,
                          kv_heads=4,
                          seq_len=2000,
                          dim=128,
                          block_size=128,
                          topk=4,
                          query_chunk=512,
                          device=get_accelerator().current_device_name(),
                          dtype=torch.bfloat16)
    for name, a, b in zip(("out", "dq", "dk", "dv"), got, want):
        torch.testing.assert_close(a.float(), b, rtol=2e-2, atol=2e-2 * b.abs().max().item(), msg=name)


@pytest.mark.skipif(not _triton_available(), reason="the Triton kernels need a GPU with Triton")
@pytest.mark.parametrize("seq_len, kv_len, topk", [(203, 203, 4), (256, 1000, 4), (256, 1000, 3)])
def test_selected_attention_matches_dense_reference(seq_len, kv_len, topk):
    """The Triton kernels against dense float32 attention, and no farther from it than the PyTorch path in bf16:
    queries that are the last ``seq_len`` of ``kv_len`` keys (one rank of the sequence split) and lengths that
    are not multiples of the block size, and a selected-block count that is not a power of 2. 32 query heads on 2
    key-value heads (16 per group)."""
    shapes = dict(batch=1,
                  heads=32,
                  kv_heads=2,
                  seq_len=seq_len,
                  dim=64,
                  block_size=64,
                  topk=topk,
                  query_chunk=512,
                  device=get_accelerator().current_device_name(),
                  dtype=torch.bfloat16,
                  kv_len=kv_len)
    got, want = _run_both(**shapes, attention=_selected)
    pytorch, _ = _run_both(**shapes)
    for name, a, b, c in zip(("out", "dq", "dk", "dv"), got, want, pytorch):
        torch.testing.assert_close(a.float(), b, rtol=2e-2, atol=2e-2 * b.abs().max().item(), msg=name)
        error = ((a.float() - b).norm() / b.norm()).item()
        error_pytorch = ((c.float() - b).norm() / b.norm()).item()
        assert error <= 1.5 * error_pytorch + 1e-4, f"{name}: {error:.2e} against the PyTorch path's {error_pytorch:.2e}"


@pytest.mark.skipif(not _cuda_available(), reason="bf16 matmuls need a CUDA device")
@pytest.mark.parametrize("kernel", ["torch", "triton"])
def test_key_gradients_summed_over_every_query_keep_float32_accuracy(kernel):
    """Every one of 8,192 queries selects block 0, so each key of block 0 sums a gradient term from
    every query. Added one at a time in bf16, as autograd's gather backward does, those gradients have
    a relative error of about 4e-2; added in float32 about 3e-3. The Triton kernel also splits block 0's
    list of 8,192 queries over several programs."""
    if kernel == "triton" and not _triton_available():
        pytest.skip("Triton is not available")
    got, want = _run_both(batch=1,
                          heads=16,
                          kv_heads=1,
                          seq_len=8192,
                          dim=128,
                          block_size=64,
                          topk=2,
                          query_chunk=1024,
                          device=get_accelerator().current_device_name(),
                          dtype=torch.bfloat16,
                          always=(0, ),
                          attention=_selected if kernel == "triton" else block_sparse_attention)
    for name, a, b in zip(("dk", "dv"), got[2:], want[2:]):
        a, b = a[:, :, :64].float(), b[:, :, :64]
        relative = ((a - b).norm() / b.norm()).item()
        assert relative < 1e-2, f"{name} of block 0: relative error {relative:.3e}"


def _tiny_config(transformers, group_of_16=False):
    """``group_of_16`` gives 16 query heads per key-value head, the smallest group the Triton kernels take."""
    kv_heads = 2 if group_of_16 else 4
    return transformers.MiniMaxM3VLTextConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=128,
        dense_intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=16 * kv_heads if group_of_16 else 16,
        num_key_value_heads=kv_heads,
        head_dim=32,
        max_position_embeddings=1024,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 5000000.0,
            "partial_rotary_factor": 0.5
        },
        mlp_layer_types=["dense"] * 3,  # attention is under test
        layer_types=["full_attention", "minimax_m3_sparse", "minimax_m3_sparse"],
        index_n_heads=kv_heads,
        index_head_dim=32,
        index_block_size=16,
        index_topk_blocks=3,
        index_local_blocks=1,
        use_cache=False,
        tie_word_embeddings=False,
        bos_token_id=0,
        eos_token_id=1,
    )


def _loss_and_grads(model, input_ids):
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads


def test_replacement_matches_transformers_model(monkeypatch):
    """Loss and every parameter gradient of a small model, float32 on CPU, against the stock SDPA
    path, which builds the dense block mask. The same set of parameters must receive gradients: the
    indexer receives none in either."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.minimax_m3_vl")
    from deepspeed.models.minimax_m3_vl import attention, replace_attention

    # The replacement runs the stock forward when it gets a mask; count the sparse calls so the
    # comparison cannot silently be stock against stock.
    sparse_calls = []

    def counting_block_sparse_attention(*args, **kwargs):
        sparse_calls.append(1)
        return block_sparse_attention(*args, **kwargs)

    monkeypatch.setattr(attention, "block_sparse_attention", counting_block_sparse_attention)

    config = _tiny_config(transformers)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    stock = transformers.MiniMaxM3VLForCausalLM(config).float()
    fast = copy.deepcopy(stock)
    assert replace_attention(fast, query_chunk=48, index_query_chunk=64) == 2
    input_ids = torch.randint(0, config.vocab_size, (2, 203), generator=torch.Generator().manual_seed(1))

    ref_loss, ref_grads = _loss_and_grads(stock, input_ids)
    loss, grads = _loss_and_grads(fast, input_ids)
    assert len(sparse_calls) == 2
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-5)
    assert grads.keys() == ref_grads.keys()
    assert not any(".indexer." in name for name in grads)
    for name, want in ref_grads.items():
        torch.testing.assert_close(grads[name], want, rtol=1e-4, atol=1e-5 * want.abs().max().item(), msg=name)


@pytest.mark.skipif(not _cuda_available(), reason="bf16 autocast on the GPU")
def test_block_selection_ignores_autocast():
    """The indexer scores are float32, as in the model code. The DeepSpeed engine runs the forward under bf16
    autocast when ``torch_autocast`` is enabled, which would make the score matmul bf16 and change which blocks win.
    A long sequence (128 blocks per query, 3 selected) with large scores so that bf16 rounding changes the ranking."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.minimax_m3_vl")
    from deepspeed.models.minimax_m3_vl.attention import select_blocks

    config = _tiny_config(transformers)
    device = get_accelerator().current_device_name()
    torch.manual_seed(0)
    model = transformers.MiniMaxM3VLForCausalLM(config).to(device, torch.bfloat16)
    indexer = model.model.layers[1].self_attn.indexer
    seq = 2048
    hidden = 4 * torch.randn(1, seq, config.hidden_size, device=device, dtype=torch.bfloat16)
    positions = torch.arange(seq, device=device)[None]
    cos, sin = model.model.rotary_emb(hidden, positions)
    with torch.no_grad():
        want = select_blocks(indexer, hidden, cos, sin, positions, 512)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
            got = select_blocks(indexer, hidden, cos, sin, positions, 512)
    assert torch.equal(got, want)


def test_replace_keeps_names_and_runs_stock_forward_with_cache_or_mask():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.minimax_m3_vl")
    from deepspeed.models.minimax_m3_vl import DeepSpeedMiniMaxM3VLAttention, replace_attention

    config = _tiny_config(transformers)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    stock = transformers.MiniMaxM3VLForCausalLM(config).float().eval()
    fast = copy.deepcopy(stock)
    replace_attention(fast)
    kinds = [type(layer.self_attn) for layer in fast.model.layers]
    assert kinds[0] is not DeepSpeedMiniMaxM3VLAttention  # the dense layer stays
    assert kinds[1:] == [DeepSpeedMiniMaxM3VLAttention] * 2
    assert [name for name, _ in fast.named_parameters()] == [name for name, _ in stock.named_parameters()]

    # A padded batch with a key-value cache: both need the stock path, which must give the same result.
    input_ids = torch.randint(0, config.vocab_size, (2, 40), generator=torch.Generator().manual_seed(1))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 30:] = 0
    with torch.no_grad():
        want = stock(input_ids=input_ids, attention_mask=attention_mask, use_cache=True).logits
        got = fast(input_ids=input_ids, attention_mask=attention_mask, use_cache=True).logits
    torch.testing.assert_close(got, want)


@pytest.mark.skipif(not _cuda_available(), reason="bf16 ZeRO-3 training needs a CUDA device")
class TestMiniMaxM3AttentionZeroStage3(DistributedTest):
    """One training step under ZeRO-3 with partitioned parameters: the replaced layers and the
    indexer, which runs under no_grad, must see the gathered weights."""
    world_size = 2

    def test_loss_and_grad_match_single_process(self):
        transformers = pytest.importorskip("transformers")
        pytest.importorskip("transformers.models.minimax_m3_vl")
        from deepspeed.models.minimax_m3_vl import replace_attention
        from deepspeed.utils import safe_get_full_grad

        config = _tiny_config(transformers)
        config._attn_implementation = "sdpa"
        device = get_accelerator().current_device_name()
        torch.manual_seed(0)  # the same weights on every rank and in the reference
        model = transformers.MiniMaxM3VLForCausalLM(config).to(torch.bfloat16)
        assert replace_attention(model) == 2
        input_ids = torch.randint(0, config.vocab_size, (1, 256), generator=torch.Generator().manual_seed(1))
        input_ids = input_ids.to(device)

        reference = copy.deepcopy(model).to(device)
        ref_loss = reference(input_ids=input_ids, labels=input_ids).loss
        ref_loss.backward()
        ref_grad = reference.model.layers[1].self_attn.k_proj.weight.grad.float()

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
        grad = safe_get_full_grad(engine.module.model.layers[1].self_attn.k_proj.weight)
        # Every rank sees the same batch, so the averaged gradient equals the single-process one.
        torch.testing.assert_close(loss.float(), ref_loss.float(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(grad.float(), ref_grad, rtol=3e-2, atol=3e-2 * ref_grad.abs().max().item())
        engine.step()


@pytest.mark.skipif(not _triton_available(), reason="the Triton kernels need a GPU with Triton")
def test_auto_kernel_runs_triton_for_bf16_on_gpu(monkeypatch):
    """``kernel="auto"`` must take the Triton kernels for bf16 on a GPU (attention and indexer), not silently the
    PyTorch path."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.minimax_m3_vl")
    from deepspeed.models.minimax_m3_vl import attention, replace_attention

    calls = {"triton": 0, "torch": 0, "index_kernel": 0}

    def counting(kind, fn):

        def wrapped(*args, **kwargs):
            calls[kind] += 1
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(attention, "selected_attention", counting("triton", selected_attention))
    monkeypatch.setattr(attention, "block_sparse_attention", counting("torch", block_sparse_attention))
    monkeypatch.setattr(attention, "index_block_max", counting("index_kernel", attention.index_block_max))
    config = _tiny_config(transformers, group_of_16=True)
    config._attn_implementation = "sdpa"
    device = get_accelerator().current_device_name()
    model = transformers.MiniMaxM3VLForCausalLM(config).to(device, torch.bfloat16)
    assert replace_attention(model) == 2
    input_ids = torch.randint(0, config.vocab_size, (1, 64), device=device)
    model(input_ids=input_ids, labels=input_ids).loss.backward()
    assert calls == {"triton": 2, "torch": 0, "index_kernel": 2}


@pytest.mark.skipif(not _triton_available(), reason="the Triton kernels need a GPU with Triton")
@pytest.mark.parametrize("query_offset", [0, 700])
def test_index_block_max_matches_reference(query_offset):
    """The indexer kernel's block maxima must equal the float32 scores masked to the visible keys and reduced per
    block, including a last block that is only partly filled (1,000 keys, blocks of 128) and queries that start
    later in the key sequence, as on a later rank of a sequence split (blocks entirely in the future are -inf)."""
    from deepspeed.models.minimax_m3_vl.index_kernel import index_block_max

    device = get_accelerator().current_device_name()
    torch.manual_seed(0)
    heads, num_queries, key_len, dim, block = 4, 300, 1000, 128, 128
    q = torch.randn(1, heads, num_queries, dim, device=device, dtype=torch.bfloat16)
    keys = torch.randn(1, 1, key_len, dim, device=device, dtype=torch.bfloat16)
    positions = torch.arange(query_offset, query_offset + num_queries, device=device)[None]
    num_blocks = -(-key_len // block)
    scores = torch.matmul(q.float(), keys.float().transpose(-1, -2))
    future = torch.arange(key_len, device=device)[None, None, None, :] > positions[:, None, :, None]
    scores = scores.masked_fill(future, float("-inf"))
    scores = torch.nn.functional.pad(scores, (0, num_blocks * block - key_len), value=float("-inf"))
    want = scores.view(1, heads, num_queries, num_blocks, block).amax(dim=-1)
    got = index_block_max(q, keys, positions, block)
    assert got.shape == want.shape
    assert torch.equal(torch.isinf(got), torch.isinf(want))
    finite = torch.isfinite(want)
    torch.testing.assert_close(got[finite], want[finite], rtol=1e-5, atol=1e-4)


@pytest.mark.skipif(not _triton_available(), reason="the Triton kernels need a GPU with Triton")
def test_index_kernel_selects_the_float32_blocks(monkeypatch):
    """The indexer kernel forms the same products as the float32 matmul and sums them in float32 in another order,
    so the selection may differ only where two blocks nearly tie (allowed for at most 0.1% of the slots)."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.minimax_m3_vl")
    from deepspeed.models.minimax_m3_vl import attention

    kernel_calls = []
    index_block_max = attention.index_block_max

    def counting_index_block_max(*args, **kwargs):
        kernel_calls.append(1)
        return index_block_max(*args, **kwargs)

    monkeypatch.setattr(attention, "index_block_max", counting_index_block_max)

    config = _tiny_config(transformers)
    device = get_accelerator().current_device_name()
    torch.manual_seed(0)
    model = transformers.MiniMaxM3VLForCausalLM(config).to(device, torch.bfloat16)
    indexer = model.model.layers[1].self_attn.indexer
    seq = 2048
    hidden = 4 * torch.randn(1, seq, config.hidden_size, device=device, dtype=torch.bfloat16)
    positions = torch.arange(seq, device=device)[None]
    cos, sin = model.model.rotary_emb(hidden, positions)
    with torch.no_grad():
        want = attention.select_blocks(indexer, hidden, cos, sin, positions, 512)
        got = attention.select_blocks(indexer, hidden, cos, sin, positions, 512, use_kernel=True)
    assert len(kernel_calls) == 1
    assert got.shape == want.shape
    assert (got != want).float().mean().item() <= 1e-3


@pytest.mark.skipif(not _cuda_available(), reason="the sequence split needs CUDA devices")
class TestMiniMaxM3AttentionContextParallel(DistributedTest):
    """Each sequence split over the ranks (context parallel) must give the outputs and gradients of the whole
    sequence on one rank. index_topk_blocks covers every block, so every query keeps all its causal keys and the
    selection has no near-ties: what is compared is the gathered keys and values, the global positions of the
    indexer and the attention, and the reduce-scattered key and value gradients."""
    world_size = 2

    @pytest.mark.parametrize("kernel", ["torch", "triton"])
    def test_split_matches_whole_sequence(self, kernel):
        transformers = pytest.importorskip("transformers")
        pytest.importorskip("transformers.models.minimax_m3_vl")
        if kernel == "triton" and not _triton_available():
            pytest.skip("Triton is not available")
        import deepspeed.comm as dist
        from deepspeed.models.minimax_m3_vl import replace_attention, set_context_parallel_group

        tokens = 256
        config = _tiny_config(transformers, group_of_16=True)
        config._attn_implementation = "sdpa"
        config.index_topk_blocks = tokens // config.index_block_size
        device = get_accelerator().current_device_name()
        torch.manual_seed(0)  # the same weights on every rank
        model = transformers.MiniMaxM3VLForCausalLM(config).to(device, torch.bfloat16)
        assert replace_attention(model, dense=True, kernel=kernel) == 3
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
