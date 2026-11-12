# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Contract tests for the native rollout decoder (CPU, one rank).

The oracle is HuggingFace transformers' Qwen3: the decoder loaded with the same weights must produce the same
logits on left-padded prompts. The layout it plans must be what its loader writes into the arena, which is what lets
weight sync write the arena in place without the loader and without reallocating anything a CUDA graph reads.
"""

import gc
import weakref

import pytest
import torch

from deepspeed.runtime.rollout.native import DecoderSpec, TPDecoder
from deepspeed.runtime.rollout.native.generator import prefill_inputs

transformers = pytest.importorskip("transformers")


def tiny_qwen3(tie: bool):
    cfg = transformers.Qwen3Config(vocab_size=101,
                                   hidden_size=64,
                                   intermediate_size=96,
                                   num_hidden_layers=2,
                                   num_attention_heads=4,
                                   num_key_value_heads=2,
                                   head_dim=16,
                                   max_position_embeddings=64,
                                   tie_word_embeddings=tie)
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    # Random norms, so a weight loaded into the wrong slot cannot hide behind an all-ones initialization.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
    return cfg, model


def left_padded_batch():
    ids = torch.tensor([[0, 0, 0, 5, 9, 17, 3], [11, 4, 8, 23, 42, 7, 2]])
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
    return ids, mask


def native_logits(dec: TPDecoder, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, T = ids.shape
    positions, attend = prefill_inputs(T - mask.sum(1), T)
    with torch.no_grad():
        return dec.logits(dec.prefill(ids, positions, attend))


@pytest.mark.parametrize("tie", [True, False])
def test_logits_match_huggingface_on_left_padded_prompts(tie):
    cfg, model = tiny_qwen3(tie)
    dec = TPDecoder(DecoderSpec.from_hf_config(cfg), dtype=torch.float32, device="cpu", max_batch_size=2, max_len=16)
    dec.load_weights(model.state_dict().items())
    ids, mask = left_padded_batch()
    positions = (mask.cumsum(1) - 1).clamp(min=0)
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=mask, position_ids=positions).logits
    got = native_logits(dec, ids, mask)
    real = mask.bool()
    torch.testing.assert_close(got[real], ref[real], atol=1e-4, rtol=1e-4)


def test_decode_steps_match_huggingface_on_the_whole_sequence():
    # Prefill a prefix, then feed the rest one token per step through the KV cache: each step's logits must be
    # HuggingFace's at that position of the whole sequence (catches wrong cache slots, positions or masks).
    cfg, model = tiny_qwen3(tie=False)
    dec = TPDecoder(DecoderSpec.from_hf_config(cfg), dtype=torch.float32, device="cpu", max_batch_size=2, max_len=16)
    dec.load_weights(model.state_dict().items())
    ids, mask = left_padded_batch()
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(1) - 1).clamp(min=0)).logits
    T = 5
    pad = (1 - mask[:, :T]).sum(1)
    got = [native_logits(dec, ids[:, :T], mask[:, :T])[:, -1]]
    with torch.no_grad():
        for t in range(T, ids.shape[1]):
            hidden = dec.decode(ids[:, t:t + 1], torch.tensor([t]), pad)
            got.append(dec.logits(hidden[:, -1]))
    torch.testing.assert_close(torch.stack(got, 1), ref[:, T - 1:], atol=1e-4, rtol=1e-4)


def test_fused_decoder_matches_and_is_freed():
    # The fused elementwise kernels compute the same logits, and the compile cache does not keep a decoder alive
    # (reconfiguration builds a new decoder each time).
    cfg, model = tiny_qwen3(tie=False)
    spec = DecoderSpec.from_hf_config(cfg)
    ids, mask = left_padded_batch()
    logits = []
    for fuse in (False, True):
        dec = TPDecoder(spec, dtype=torch.float32, device="cpu", max_batch_size=2, max_len=16, fuse=fuse)
        dec.load_weights(model.state_dict().items())
        logits.append(native_logits(dec, ids, mask))
    alive = weakref.ref(dec)
    del dec
    gc.collect()
    assert alive() is None
    torch.testing.assert_close(logits[1], logits[0], atol=1e-5, rtol=1e-5)


def test_planned_layout_is_what_the_loader_writes():
    # A sync writes the arena through layout_pieces, without the loader: copying each weight through its pieces must
    # give the arena the loader gives, and the loader must leave the arena where it was.
    cfg, model = tiny_qwen3(tie=False)
    spec = DecoderSpec.from_hf_config(cfg)
    weights = {n: w + 0.01 * torch.randn_like(w) for n, w in model.state_dict().items()}
    for tp_size, tp_rank in [(1, 0), (2, 0), (2, 1)]:
        dec = TPDecoder(spec, dtype=torch.float32, device="cpu", max_len=16, tp_rank=tp_rank, tp_size=tp_size)
        address = dec.arena.address()
        written = torch.zeros_like(dec.arena.buffer)
        for name, pieces in dec.layout_pieces({n: t.shape for n, t in weights.items()}).items():
            for local, piece in pieces:
                assert local == "arena"
                src = weights[name].as_strided(piece.shape, piece.source_strides, piece.source_offset)
                written.as_strided(piece.shape, piece.dest_strides, piece.dest_offset).copy_(src)
        dec.load_weights(weights.items())
        assert torch.equal(written, dec.arena.buffer)
        assert dec.arena.address() == address


def test_compaction_keeps_every_rows_output():
    # Dropping finished rows mid-generation (NativeGenerator(compact=True)) must leave every row's greedy tokens,
    # length and log-probs as an uncompacted run gives them: a wrong KV move, padding or row mapping changes them.
    from deepspeed.runtime.rollout.native.generator import NativeGenerator
    cfg, model = tiny_qwen3(tie=False)
    B, T, N = 32, 6, 160
    dec = TPDecoder(DecoderSpec.from_hf_config(cfg),
                    dtype=torch.float32,
                    device="cpu",
                    max_batch_size=B,
                    max_len=T + N)
    dec.load_weights(model.state_dict().items())
    g = torch.Generator().manual_seed(1)
    lens = torch.randint(2, T + 1, (B, ), generator=g)
    mask = (torch.arange(T).view(1, T) >= (T - lens).view(B, 1)).long()
    ids = torch.randint(0, cfg.vocab_size, (B, T), generator=g) * mask
    eos = list(range(0, cfg.vocab_size, 5))  # a fifth of the vocabulary ends a row: stops scatter, a few run long
    t0, lp0, n0 = NativeGenerator(dec, use_graph=False).generate(ids, mask, N, temperature=0.0, eos_token_ids=eos)
    compact = NativeGenerator(dec, use_graph=False, compact=True, min_batch=2)
    t1, lp1, n1 = compact.generate(ids, mask, N, temperature=0.0, eos_token_ids=eos)
    assert compact.compactions >= 1, "the batch never got below half: choose an EOS set that ends more rows"
    assert int(n0.max()) > 64 and int(n0.min()) < 32
    assert torch.equal(n0, n1)
    for b in range(B):
        k = int(n0[b])
        assert torch.equal(t0[b, :k], t1[b, :k]), f"row {b}"
        torch.testing.assert_close(lp0[b, :k], lp1[b, :k], atol=1e-5, rtol=1e-5)
