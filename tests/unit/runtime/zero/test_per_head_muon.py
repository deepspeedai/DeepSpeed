# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Per-head Muon: Newton-Schulz on each attention head instead of the whole projection.

Full-matrix orthogonalization treats every head as one coupled block, so heads with larger
momentum dominate the shared update direction. Kimi K3 (arXiv:2607.24653 5.2.5) and GLM-5
"Muon Split" (arXiv:2602.15763) both orthogonalize per head instead. See #8367.

Three concerns, in order:

1. **The arithmetic** -- per-head Newton-Schulz equals orthogonalizing each head alone.
2. **The tagging** -- which parameters get a head count, and what it is. `set_optimizer_flags`
   already tags `use_muon` per parameter; head structure rides along the same way, so it does
   not depend on AutoTP being enabled.
3. **Tensor parallelism** -- `set_optimizer_flags` runs before the engine partitions the model,
   so the count it records counts the model's heads. Column-parallel TP splits attention
   projections on dim 0, the axis the heads are on: the per-head width survives the split and
   the count does not. Nothing catches that on its own, because the stale count usually still
   divides the shard.

CPU-only: these pin the arithmetic and the bookkeeping, not the accelerator path. The
multi-GPU side lives in `tests/unit/ops/muon/test_per_head_muon_accelerator.py`.
"""

from types import SimpleNamespace

import pytest
import torch

import deepspeed
from deepspeed import _attention_head_count, resolve_per_head_muon_after_sharding
from deepspeed.runtime.config import MUON_OPTIMIZER
from deepspeed.runtime.zero.muon.original_muon import (
    muon_update,
    zeropower_via_gram_newtonschulz,
    ns_compute_dtype,
    zeropower_via_newtonschulz5,
)

# ---------------------------------------------------------------------------
# 1. The arithmetic
# ---------------------------------------------------------------------------


def _ns_tolerance(ns_method):
    """A few ulps of whatever dtype the kernel iterates in.

    `gram` runs in fp16 and `newtonschulz5` in bf16 (fp32 where unsupported), and the iteration
    amplifies rounding, so batched and unbatched NS agree to a handful of ulps rather than
    bitwise. Deriving the bound from the dtype keeps it honest instead of tuned to pass.
    """
    eps = torch.finfo(ns_compute_dtype(ns_method)).eps
    return dict(rtol=8 * eps, atol=8 * eps)


def _norm_rtol(ns_method):
    """Scale agreement: a couple of ulps of the compute dtype, and never looser than 1%."""
    return max(1e-2, 2 * torch.finfo(ns_compute_dtype(ns_method)).eps)


def _update_only(grad, momentum, beta=0.95, nesterov=True):
    """The pre-orthogonalization update muon_update forms, without mutating the caller's tensors."""
    grad, momentum = grad.clone(), momentum.clone()
    momentum.lerp_(grad, 1 - beta)
    return grad.lerp_(momentum, beta) if nesterov else momentum


@pytest.mark.parametrize("ns_method", ["gram", "newtonschulz5"])
@pytest.mark.parametrize("num_heads,head_dim,in_features", [(4, 8, 32), (2, 16, 32), (8, 4, 64)])
def test_per_head_matches_orthogonalizing_each_head_alone(ns_method, num_heads, head_dim, in_features):
    """The batched path must equal running NS on each head block on its own."""
    torch.manual_seed(0)
    out_features = num_heads * head_dim
    grad = torch.randn(out_features, in_features)
    momentum = torch.randn(out_features, in_features)

    got = muon_update(grad.clone(), momentum.clone(), ns_method=ns_method, num_heads=num_heads)

    update = _update_only(grad, momentum)
    ns_fn = zeropower_via_gram_newtonschulz if ns_method == "gram" else zeropower_via_newtonschulz5
    scale = max(1, head_dim / in_features)**0.5
    expected = torch.cat([ns_fn(update[h * head_dim:(h + 1) * head_dim], steps=5) * scale
                          for h in range(num_heads)]).to(got.dtype)

    assert got.shape == (out_features, in_features)
    torch.testing.assert_close(got, expected, **_ns_tolerance(ns_method))
    # Elementwise agreement is ulp-limited, so also pin the overall scale.
    torch.testing.assert_close(got.norm(), expected.norm(), rtol=_norm_rtol(ns_method), atol=0.0)


@pytest.mark.parametrize("ns_method", ["gram", "newtonschulz5"])
def test_single_head_reproduces_the_full_matrix_path(ns_method):
    """num_heads=1 is the whole projection, so it has to agree with the existing behaviour."""
    torch.manual_seed(0)
    grad = torch.randn(16, 32)
    momentum = torch.randn(16, 32)

    per_head = muon_update(grad.clone(), momentum.clone(), ns_method=ns_method, num_heads=1)
    full = muon_update(grad.clone(), momentum.clone(), ns_method=ns_method)

    # Batched and unbatched NS take the same arithmetic path but not bit-identically in the
    # half-precision compute dtype, so compare at that granularity.
    torch.testing.assert_close(per_head, full, **_ns_tolerance(ns_method))
    torch.testing.assert_close(per_head.norm(), full.norm(), rtol=_norm_rtol(ns_method), atol=0.0)


def test_per_head_differs_from_full_matrix_when_heads_are_unbalanced():
    """The point of the change: one loud head must stop setting the direction for the quiet ones.

    Without this the test would pass even if num_heads were ignored.
    """
    torch.manual_seed(0)
    num_heads, head_dim, in_features = 4, 8, 32
    grad = torch.randn(num_heads * head_dim, in_features)
    grad[:head_dim] *= 100.0  # one head with a far larger gradient scale
    momentum = torch.zeros_like(grad)

    per_head = muon_update(grad.clone(), momentum.clone(), num_heads=num_heads)
    full = muon_update(grad.clone(), momentum.clone())

    quiet = slice(head_dim, None)
    assert not torch.allclose(per_head[quiet], full[quiet], rtol=1e-2, atol=1e-2)
    # Every head should come out with a comparable update scale.
    norms = torch.stack([per_head[h * head_dim:(h + 1) * head_dim].norm() for h in range(num_heads)])
    assert norms.max() / norms.min() < 1.5


def test_rejects_shapes_that_do_not_split_into_heads():
    grad = torch.randn(15, 32)
    momentum = torch.zeros_like(grad)

    with pytest.raises(ValueError, match="not divisible by num_heads"):
        muon_update(grad.clone(), momentum.clone(), num_heads=4)

    conv_like = torch.randn(4, 4, 3, 3)
    with pytest.raises(ValueError, match="expects a 2D attention projection"):
        muon_update(conv_like.clone(), torch.zeros_like(conv_like), num_heads=4)


# ---------------------------------------------------------------------------
# 2. Which parameters are tagged, and with how many heads
# ---------------------------------------------------------------------------


class _Attn(torch.nn.Module):

    def __init__(self, hidden=64, q_heads=8, kv_heads=2, head_dim=8, fused=False):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden, q_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(q_heads * head_dim, hidden, bias=False)
        self.mlp = torch.nn.Linear(hidden, hidden, bias=False)
        self.embed_tokens = torch.nn.Embedding(16, hidden)
        if fused:
            self.qkv_proj = torch.nn.Linear(hidden, (q_heads + 2 * kv_heads) * head_dim, bias=False)
        # Real configs carry the per-head width, either as head_dim or derivably from
        # hidden_size. Without one the shape cannot confirm the name, and the tagger declines.
        self.config = SimpleNamespace(num_attention_heads=q_heads,
                                      num_key_value_heads=kv_heads,
                                      hidden_size=hidden,
                                      head_dim=head_dim)


def _flags(model, per_head=True):
    cfg = SimpleNamespace(optimizer_name=MUON_OPTIMIZER,
                          optimizer_params={"per_head_muon": per_head} if per_head else {})
    deepspeed.set_optimizer_flags(cfg, model)
    return {name: getattr(p, "muon_num_heads", "MISSING") for name, p in model.named_parameters()}


def test_query_projection_uses_the_query_head_count():
    tags = _flags(_Attn(q_heads=8, kv_heads=2))

    assert tags["q_proj.weight"] == 8


def test_output_projection_is_left_alone():
    """o_proj is `[hidden, num_heads * head_dim]` - its heads are on the input axis.

    The split is on dim 0, so tagging it would cut across the wrong axis, and with the usual
    hidden == num_heads * head_dim it still divides evenly, i.e. silently wrong rather than an
    error. Regression test: it was tagged in the first version of this.
    """
    tags = _flags(_Attn(q_heads=8, kv_heads=2))

    assert tags["o_proj.weight"] is None


@pytest.mark.parametrize("mlp_name", [
    "intermediate.dense.weight",
    "output.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
])
def test_mlp_matrices_named_dense_are_not_treated_as_attention(mlp_name):
    """`dense` names an MLP matrix as often as an attention one.

    Matching it anywhere in the path tagged `intermediate.dense` and `dense_h_to_4h` with a head
    count, splitting a matrix that has no head structure. Regression test: the first version of
    this matched on the full path and did exactly that.
    """
    from deepspeed import _attention_head_count

    model = _Attn(q_heads=8, kv_heads=2)
    weight = torch.zeros(4 * 64, 64)

    assert _attention_head_count(f"encoder.layer.0.{mlp_name}", weight, model) is None


def test_kv_projections_use_the_kv_head_count_under_gqa():
    """K/V have fewer heads than Q under GQA, and splitting them by the query count would be wrong."""
    tags = _flags(_Attn(q_heads=8, kv_heads=2))

    assert tags["k_proj.weight"] == 2
    assert tags["v_proj.weight"] == 2


def test_non_attention_parameters_are_left_on_the_full_matrix_path():
    tags = _flags(_Attn())

    assert tags["mlp.weight"] is None
    assert tags["embed_tokens.weight"] is None


def test_fused_qkv_is_skipped():
    """One matrix holding Q, K and V does not split into uniform heads under GQA."""
    tags = _flags(_Attn(fused=True))

    assert tags["qkv_proj.weight"] is None


def test_opt_in_is_required():
    tags = _flags(_Attn(), per_head=False)

    assert all(v is None for v in tags.values()), tags


def test_shape_that_does_not_divide_is_skipped():
    """A projection whose output dim is not a multiple of the head count is not that layout."""
    model = _Attn(q_heads=8, kv_heads=2)
    model.q_proj = torch.nn.Linear(64, 63, bias=False)  # 63 % 8 != 0

    assert _flags(model)["q_proj.weight"] is None


def test_use_muon_tagging_is_unchanged():
    model = _Attn()
    _flags(model)

    assert model.q_proj.weight.use_muon is True
    assert model.embed_tokens.weight.use_muon is False


@pytest.mark.parametrize("q_heads,kv_heads", [(8, 8), (8, 1), (12, 4)])
def test_head_counts_track_the_config(q_heads, kv_heads):
    tags = _flags(_Attn(q_heads=q_heads, kv_heads=kv_heads, hidden=64, head_dim=8))

    assert tags["q_proj.weight"] == q_heads
    assert tags["k_proj.weight"] == kv_heads


@pytest.mark.parametrize("arch", ["llama", "qwen2", "mistral"])
def test_split_qkv_architectures_tag_only_qkv(arch):
    """Real HF configs rather than a stand-in, so the leaf names are the ones models actually use."""
    transformers = pytest.importorskip("transformers")
    cfg_cls = {
        "llama": transformers.LlamaConfig,
        "qwen2": transformers.Qwen2Config,
        "mistral": transformers.MistralConfig,
    }[arch]
    cfg = cfg_cls(hidden_size=64,
                  num_attention_heads=8,
                  num_key_value_heads=2,
                  num_hidden_layers=1,
                  intermediate_size=128,
                  vocab_size=32)
    model = transformers.AutoModelForCausalLM.from_config(cfg)

    tags = {n.split(".")[-2]: _attention_head_count(n, p, model) for n, p in model.named_parameters() if p.ndim == 2}

    assert tags["q_proj"] == 8
    assert tags["k_proj"] == 2, "GQA: k/v are blocked by num_key_value_heads, not the query count"
    assert tags["v_proj"] == 2
    assert tags["o_proj"] is None, "o_proj's heads are on the input axis"
    for mlp_leaf in ("gate_proj", "up_proj", "down_proj"):
        assert tags[mlp_leaf] is None, f"{mlp_leaf} has no head structure"


@pytest.mark.parametrize("arch", ["gpt_neox", "falcon"])
def test_fused_qkv_architectures_tag_nothing(arch):
    """These name their MLP matrices `dense_h_to_4h` / `dense_4h_to_h` and their output proj `dense`.

    Matching `dense` anywhere in the path tagged all three; this pins that none of them are.
    """
    transformers = pytest.importorskip("transformers")
    cfg_cls = {"gpt_neox": transformers.GPTNeoXConfig, "falcon": transformers.FalconConfig}[arch]
    kwargs = dict(hidden_size=64, num_attention_heads=8, num_hidden_layers=1, vocab_size=32)
    if arch == "gpt_neox":
        kwargs["intermediate_size"] = 128
    model = transformers.AutoModelForCausalLM.from_config(cfg_cls(**kwargs))

    tags = {n: _attention_head_count(n, p, model) for n, p in model.named_parameters() if p.ndim == 2}

    assert all(v is None for v in tags.values()), \
        {k: v for k, v in tags.items() if v is not None}


# Shapes and config values read off the real checkpoints delock pointed at in #8367:
# inference-optimization/GLM-5.2-0.8B-A0.8B and inference-optimization/Kimi-K3-0.40B.
def _glm52_mla_config():
    return SimpleNamespace(num_attention_heads=16,
                           num_key_value_heads=16,
                           hidden_size=2048,
                           head_dim=64,
                           q_lora_rank=512,
                           kv_lora_rank=128,
                           qk_nope_head_dim=192,
                           qk_rope_head_dim=64,
                           v_head_dim=128)


@pytest.mark.parametrize(
    "leaf,shape,expected",
    [
        ("q_b_proj", (4096, 512), 16),  # 16 * (qk_nope 192 + qk_rope 64)
        ("kv_b_proj", (5120, 128), 16),  # 16 * (qk_nope 192 + v_head_dim 128)
        ("q_a_proj", (512, 2048), None),  # down-projection, no head structure
        ("kv_a_proj_with_mqa", (192, 2048), None),  # latent + rope, does not split into heads
        ("o_proj", (2048, 2048), None),  # heads on the input axis
    ])
def test_mla_tags_only_the_up_projections(leaf, shape, expected):
    """MLA blocks its two up-projections by head, which is the split GLM-5's Muon Split applies.

    Their per-head width is not `head_dim`: q_b is qk_nope + qk_rope and kv_b is
    qk_nope + v_head_dim, so a tagger that assumes `num_heads * head_dim` rejects both.
    """
    model = SimpleNamespace(config=_glm52_mla_config())
    name = f"model.layers.0.self_attn.{leaf}.weight"

    assert _attention_head_count(name, torch.zeros(shape), model) == expected


def test_mla_head_width_is_checked_not_assumed():
    """A shape that does not equal num_heads * per-head width is not the layout we think it is."""
    model = SimpleNamespace(config=_glm52_mla_config())

    ok = _attention_head_count("l.0.self_attn.q_b_proj.weight", torch.zeros(4096, 512), model)
    wrong = _attention_head_count("l.0.self_attn.q_b_proj.weight", torch.zeros(4080, 512), model)

    assert ok == 16
    assert wrong is None


# Shapes measured by instantiating DeepseekV2Attention on the released
# deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct config under transformers 5.16.1.
def _deepseek_v2_lite_mla_config():
    return SimpleNamespace(num_attention_heads=16,
                           num_key_value_heads=16,
                           hidden_size=2048,
                           head_dim=64,
                           q_lora_rank=None,
                           kv_lora_rank=512,
                           qk_nope_head_dim=128,
                           qk_rope_head_dim=64,
                           v_head_dim=128)


@pytest.mark.parametrize(
    "leaf,shape,expected",
    [
        ("q_proj", (3072, 2048), 16),  # 16 * (qk_nope 128 + qk_rope 64)
        ("kv_b_proj", (4096, 512), 16),  # 16 * (qk_nope 128 + v_head_dim 128)
        ("kv_a_proj_with_mqa", (576, 2048), None),  # kv_lora_rank + qk_rope, no head structure
        ("o_proj", (2048, 2048), None),  # heads on the input axis, and 2048 still divides by 16
    ])
def test_mla_without_q_lora_rank_tags_the_plain_q_proj(leaf, shape, expected):
    """Without a q_lora_rank there is no q_a/q_b pair; the up-projection is `q_proj` itself.

    Its per-head width stays `qk_nope + qk_rope`, so reading `head_dim` gives 16 * 64 = 1024
    against a real 3072 and drops the model off the per-head path.
    """
    model = SimpleNamespace(config=_deepseek_v2_lite_mla_config())
    name = f"model.layers.0.self_attn.{leaf}.weight"

    assert _attention_head_count(name, torch.zeros(shape), model) == expected


def test_head_dim_alone_would_reject_the_mla_q_proj():
    """Guards the width source rather than the outcome: head_dim is present and wrong here."""
    config = _deepseek_v2_lite_mla_config()

    assert config.head_dim is not None
    assert config.num_attention_heads * config.head_dim == 1024
    assert _attention_head_count("l.0.self_attn.q_proj.weight", torch.zeros(3072, 2048),
                                 SimpleNamespace(config=config)) == 16


def test_q_proj_on_a_non_mla_config_still_uses_head_dim():
    """The MLA width only applies where the config carries the MLA head dimensions."""
    config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=8, hidden_size=512, head_dim=64)
    model = SimpleNamespace(config=config)

    assert _attention_head_count("l.0.self_attn.q_proj.weight", torch.zeros(512, 512), model) == 8
    assert _attention_head_count("l.0.self_attn.q_proj.weight", torch.zeros(768, 512), model) is None


def test_linear_attention_named_like_standard_attention_is_rejected():
    """Kimi-K3-0.40B is `kimi_linear`, not MLA: q_proj is [256, 1024] with 8 heads of 74.

    The names match the standard-attention list, so only the shape check keeps it off the
    per-head path.
    """
    text = SimpleNamespace(num_attention_heads=8,
                           num_key_value_heads=8,
                           hidden_size=1024,
                           head_dim=74,
                           qk_nope_head_dim=64,
                           qk_rope_head_dim=32,
                           v_head_dim=64)
    model = SimpleNamespace(config=SimpleNamespace(text_config=text))

    for leaf in ("q_proj", "k_proj", "v_proj"):
        name = f"model.layers.0.self_attn.{leaf}.weight"
        assert _attention_head_count(name, torch.zeros(256, 1024), model) is None


def test_head_count_comes_from_the_shared_extractor():
    """Head counts are read through AutoTPMeta, so alternative config spellings work."""
    model = SimpleNamespace(config=SimpleNamespace(n_head=8, hidden_size=64, head_dim=8))

    assert _attention_head_count("l.0.attn.q_proj.weight", torch.zeros(64, 64), model) == 8


# --- candidate resolution ------------------------------------------------------
#
# `q_proj` can be either the standard query projection or, on an MLA model without a
# q_lora_rank, the query up-projection. Both candidates are evaluated and the shape decides,
# so the outcome does not depend on which config fields happen to be present.


def _kimi_k3_hybrid_config():
    """Kimi-K3-0.40B: linear-attention layers on a config that also carries MLA fields.

    `linear_attn_config` gives `num_heads: 8, head_dim: 32`, so its `q_proj` is (256, 1024).
    The top-level MLA dimensions belong to the model's two MLA layers, and `head_dim` is 74.
    Neither top-level geometry describes the KDA projection.
    """
    return SimpleNamespace(num_attention_heads=8,
                           num_key_value_heads=8,
                           hidden_size=1024,
                           head_dim=74,
                           qk_nope_head_dim=64,
                           qk_rope_head_dim=32,
                           v_head_dim=64)


@pytest.mark.parametrize("leaf", ["q_proj", "k_proj", "v_proj"])
def test_linear_attention_on_a_config_with_mla_leftovers_is_declined(leaf):
    """No candidate confirms, so it stays on the full-matrix path.

    8 x (qk_nope 64 + qk_rope 32) = 768 and 8 x head_dim 74 = 592, against 256 rows. This is
    the case tracked in #8420; until the linear-attention geometry is read, declining is the
    correct outcome and it must come from the shape rather than from branch ordering.
    """
    model = SimpleNamespace(config=_kimi_k3_hybrid_config())

    assert _attention_head_count(f"model.layers.0.self_attn.{leaf}.weight", torch.zeros(256, 1024), model) is None


def test_two_candidates_agreeing_on_the_head_count_are_not_ambiguous():
    """Ambiguity is about the answer, not the route.

    The output is a head count, so two candidates that confirm with the same count give the
    same answer and there is nothing to be ambiguous about.
    """
    config = SimpleNamespace(num_attention_heads=8,
                             num_key_value_heads=8,
                             hidden_size=512,
                             head_dim=96,
                             qk_nope_head_dim=64,
                             qk_rope_head_dim=32,
                             v_head_dim=64)
    model = SimpleNamespace(config=config)

    # 8 x 96 = 768 by head_dim, and 8 x (64 + 32) = 768 by the MLA width.
    assert _attention_head_count("l.0.self_attn.q_proj.weight", torch.zeros(768, 512), model) == 8


def test_candidates_that_disagree_on_the_head_count_are_skipped():
    """A real ambiguity: both confirm the shape, and they give different answers."""
    from deepspeed import _confirm

    candidates = [(8, 96, "mla-q"), (12, 64, "head-dim")]
    num_heads, reason = _confirm(torch.zeros(768, 512), candidates)

    assert num_heads is None
    assert reason.startswith("ambiguous:")
    assert "mla-q=8" in reason and "head-dim=12" in reason


def test_a_config_without_a_per_head_width_is_declined():
    """Divisibility alone is not confirmation.

    `rows % heads == 0` holds for matrices that are not head-blocked at all, which is how
    o_proj used to slip through. Without a width there is nothing to confirm against.
    """
    config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=8)
    model = SimpleNamespace(config=config)

    assert _attention_head_count("l.0.self_attn.q_proj.weight", torch.zeros(512, 512), model) is None


def test_head_dim_is_derived_when_the_config_omits_it():
    """Configs that leave head_dim out still define it as hidden_size // num_attention_heads."""
    config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=8, hidden_size=512)
    model = SimpleNamespace(config=config)

    assert _attention_head_count("l.0.self_attn.q_proj.weight", torch.zeros(512, 512), model) == 8


def test_the_flag_errors_rather_than_silently_doing_nothing():
    """An explicit opt-in that tags nothing is the tensor-parallel failure mode.

    Under TP the config describes the whole model while each rank holds a shard, so every
    projection fails its width check and per-head is off model-wide while the user believes it
    is on. There is no partial result to keep, so this is an error.
    """

    class _NoAttention(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Linear(64, 64, bias=False)
            self.config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=8, hidden_size=64, head_dim=8)

    with pytest.raises(ValueError, match="no attention projection could be tagged"):
        _flags(_NoAttention())

    # ...and with the flag off it is simply not asked for.
    assert _flags(_NoAttention(), per_head=False)["mlp.weight"] is None


def test_the_head_count_is_read_from_the_layer_shape_under_zero_init():
    """`zero.Init` leaves a flat placeholder and records the layer's shape as `ds_shape`.

    Reading `param.shape` there sees a 1-D tensor for every parameter, so nothing confirms and
    the flag raises on a model it could describe perfectly well.
    """
    model = _Attn(hidden=64, q_heads=8, kv_heads=2, head_dim=8)
    for p in model.parameters():
        p.ds_shape = torch.Size(p.shape)
        p.data = torch.zeros(0, dtype=p.dtype)

    tags = _flags(model)

    assert model.q_proj.weight.ndim == 1, "the partitioned parameter really is 1-D here"
    assert tags["q_proj.weight"] == 8
    assert tags["k_proj.weight"] == 2
    assert tags["o_proj.weight"] is None


# ---------------------------------------------------------------------------
# 3. Tensor parallelism: re-resolving the count against the shard
# ---------------------------------------------------------------------------


class _ShardedAttn(torch.nn.Module):
    """8 heads of 32, as the whole model sees it."""

    def __init__(self, hidden=256, heads=8, head_dim=32):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden, heads * head_dim, bias=False)
        self.mlp = torch.nn.Linear(hidden, hidden, bias=False)
        for p, num_heads in ((self.q_proj.weight, heads), (self.k_proj.weight, heads), (self.mlp.weight, None)):
            p.muon_num_heads = num_heads
            p.muon_head_dim = p.shape[0] // num_heads if num_heads else None

    def shard(self, tp: int, rows=None, leaves=("q_proj", "k_proj")):
        """Replace the weights with column-parallel shards, as AutoTP's `_tp_partition` does."""
        for leaf in leaves:
            weight = getattr(self, leaf).weight
            keep = rows if rows is not None else weight.shape[0] // tp
            weight.data = weight.data[:keep].clone()


def test_the_head_count_follows_the_shard():
    """tp=2 leaves 4 heads on this rank; the tag has to say 4, not the model's 8."""
    attn = _ShardedAttn()
    attn.shard(tp=2)

    resolve_per_head_muon_after_sharding(attn)

    assert attn.q_proj.weight.shape == (128, 256)
    assert attn.q_proj.weight.muon_num_heads == 4
    assert attn.k_proj.weight.muon_num_heads == 4


def test_the_stale_count_would_have_split_heads_in_half():
    """Why this is not caught by the existing shape check: 128 % 8 == 0.

    The divisibility assert in `_per_head_orthogonalize` passes on the stale count, so without
    this pass Newton-Schulz runs on 8 blocks of 16 - half of each head - and says nothing.
    """
    attn = _ShardedAttn()
    attn.shard(tp=2)

    rows, stale = attn.q_proj.weight.shape[0], 8
    assert rows % stale == 0, "the stale count divides, which is why it needs correcting rather than asserting"
    assert rows // stale == 16 != 32


def test_a_shard_that_splits_a_head_is_dropped():
    """A shard that does not hold whole heads has no per-head structure to use."""
    attn = _ShardedAttn()
    attn.shard(tp=2, rows=144, leaves=("q_proj", ))  # 4.5 heads of 32

    resolve_per_head_muon_after_sharding(attn)

    assert attn.q_proj.weight.muon_num_heads is None
    assert attn.k_proj.weight.muon_num_heads == 8, "one bad shard does not turn the feature off elsewhere"


def test_an_unsharded_model_keeps_its_count():
    attn = _ShardedAttn()

    resolve_per_head_muon_after_sharding(attn)

    assert attn.q_proj.weight.muon_num_heads == 8
    assert attn.q_proj.weight.shape[0] == 256


def test_untagged_parameters_are_left_alone():
    attn = _ShardedAttn()
    attn.shard(tp=2)

    resolve_per_head_muon_after_sharding(attn)

    assert getattr(attn.mlp.weight, "muon_num_heads", "MISSING") is None


def test_a_model_with_no_tags_at_all_is_a_no_op():
    """Muon without `per_head_muon`, and every non-Muon model: nothing to resolve, no error."""
    model = torch.nn.Linear(8, 8, bias=False)

    resolve_per_head_muon_after_sharding(model)  # must not raise


def test_it_raises_when_the_flag_ends_up_doing_nothing():
    """Same contract as the tagging pass: an opt-in that silently does nothing is the failure."""
    attn = _ShardedAttn()
    attn.shard(tp=2, rows=144)

    with pytest.raises(ValueError, match="sharded across head boundaries"):
        resolve_per_head_muon_after_sharding(attn)
