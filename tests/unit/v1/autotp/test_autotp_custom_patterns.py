# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed.comm as dist
import deepspeed
from copy import deepcopy
from torch import nn

from unit.common import DistributedTest, preferred_dtype
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import groups
from deepspeed.module_inject.layers import (GateUpPack_LinearLayer, LinearAllreduce, LinearLayer,
                                            SubParamLinearAllreduce, SubParamLinearLayer, fused_LinearLayer)
from deepspeed.module_inject.layers import collect_autotp_universal_checkpoint_info
from deepspeed.checkpoint.constants import (DS_AUTOTP_UC_META, PARAMETER_WITH_ROW_PARALLELISM_PATTERNS,
                                            TP_REPLICATED_PARAMETER_PATTERNS)
from deepspeed.module_inject.autotp_config import AutoTPConfig
from deepspeed.module_inject.tp_shard import AutoTPMeta, get_shard_size, get_shard_size_list
from deepspeed.module_inject.auto_tp import AutoTP
from deepspeed.module_inject.auto_tp_model_utils import (build_bloom_alibi_tensor, build_mpt_alibi_tensor,
                                                         get_alibi_mask, install_head_sharded_helper)


def skip_on_device():
    if get_accelerator().device_name() == 'xpu':
        pytest.skip("XPU requires a higher version for test")


class SequentialLinearModel(torch.nn.Module):

    def __init__(self, hidden_dim, nlayers=1):
        super(SequentialLinearModel, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(nlayers)])

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x


class CustomLinearModule(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(CustomLinearModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = torch.nn.Parameter(torch.empty(hidden_dim))
        torch.nn.init.uniform_(self.weight, -0.02, 0.02)
        torch.nn.init.uniform_(self.bias, -0.02, 0.02)

    def forward(self, x):
        return torch.matmul(x, self.weight.transpose(-1, -2)) + self.bias


class CustomLinearModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(CustomLinearModel, self).__init__()
        self.custom = CustomLinearModule(hidden_dim)

    def forward(self, x):
        return self.custom(x)


class QKVLinearModule(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(QKVLinearModule, self).__init__()
        self.qkv_proj = torch.nn.Linear(hidden_dim, hidden_dim * 3)

    def forward(self, x):
        return self.qkv_proj(x)


class QKVLinearModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(QKVLinearModel, self).__init__()
        self.self_attn = QKVLinearModule(hidden_dim)

    def forward(self, x):
        return self.self_attn(x)


class DeepAttention(torch.nn.Module):
    """Mimics HF attention module with separate projection layers."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


class DeepBlock(torch.nn.Module):
    """Mimics a single HF transformer block."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = DeepAttention(hidden_dim)

    def forward(self, x):
        return self.self_attn(x)


class DeepModel(torch.nn.Module):
    """Mimics HF transformer structure: model.layers.[N].self_attn.{q,o}_proj.

    This creates a 4-level-deep module hierarchy to test that _replace_module
    correctly propagates the full module path during recursion.
    """

    def __init__(self, hidden_dim, nlayers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([DeepBlock(hidden_dim) for _ in range(nlayers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def init_tp_engine(tp_size, partition_config=None):
    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-6
            }
        },
        "tensor_parallel": {
            "autotp_size": tp_size,
        },
        "zero_optimization": {
            "stage": 0,
        }
    }
    if partition_config is not None:
        config_dict["tensor_parallel"]["partition_config"] = partition_config
    else:
        config_dict["tensor_parallel"]["partition_config"] = {
            "use_default_specs": False,
            "layer_specs": [{
                "patterns": [".*\\.weight$"],
                "partition_type": "skip",
            }],
        }
    if preferred_dtype() is torch.float16:
        config_dict["fp16"] = {"enabled": True}
    elif preferred_dtype() is torch.bfloat16:
        config_dict["bf16"] = {"enabled": True}

    model = SequentialLinearModel(hidden_dim=8)
    deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)


def apply_autotp_with_partition_config(model, tp_size, partition_config):
    groups._init_tp_mesh_device(tensor_model_parallel_size=tp_size)
    autotp_config = AutoTPConfig.from_dict(partition_config)
    autotp = AutoTP(module=model,
                    all_reduce_linears=[],
                    prefix="",
                    state_dict=None,
                    linear_layer_setting=None,
                    orig_layer_impl=None,
                    keep_module_on_host=False,
                    partition_config=autotp_config,
                    model_config=getattr(model, "config", None))
    autotp.set_tensor_parallel_config(tp_size, groups.get_tensor_model_parallel_group())
    autotp.update_linear_policies()
    autotp._replace_module(model)
    return model


def gather_subparam_output(output, subparam_sizes, mp_group):
    tp_world_size = dist.get_world_size(group=mp_group)
    local_sizes = [size // tp_world_size for size in subparam_sizes]
    output_chunks = torch.split(output, local_sizes, dim=-1)
    gathered_chunks = []
    for chunk in output_chunks:
        chunk = chunk.contiguous()
        gathered = [torch.empty_like(chunk) for _ in range(tp_world_size)]
        dist.all_gather(gathered, chunk, group=mp_group)
        gathered_chunks.append(torch.cat(gathered, dim=-1))
    return torch.cat(gathered_chunks, dim=-1)


def assert_close_for_preferred_dtype(actual, expected):
    atol = 1e-3
    rtol = 2e-2
    if preferred_dtype() is torch.float32:
        atol = 1e-5
        rtol = 1e-5
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


class TestAutoTPCustomPatterns(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def test_custom_pattern_replacement(self):
        skip_on_device()
        partition_config = {
            "use_default_specs":
            False,
            "layer_specs": [
                {
                    "patterns": [".*linears\\.0\\.weight$"],
                    "partition_type": "row",
                },
                {
                    "patterns": [".*linears\\.1\\.weight$"],
                    "partition_type": "column",
                },
                {
                    "patterns": [".*linears\\.2\\.weight$"],
                    "partition_type": "skip",
                },
            ],
        }
        model = SequentialLinearModel(hidden_dim=16, nlayers=3)
        model = apply_autotp_with_partition_config(model, tp_size=2, partition_config=partition_config)

        assert isinstance(model.linears[0], LinearAllreduce)
        assert isinstance(model.linears[1], LinearLayer)
        assert isinstance(model.linears[2], nn.Linear)

    def test_custom_patterns_applied_via_config(self):
        skip_on_device()
        partition_config = {
            "use_default_specs":
            False,
            "layer_specs": [
                {
                    "patterns": [".*linears\\.0\\.weight$"],
                    "partition_type": "row",
                },
                {
                    "patterns": [".*linears\\.1\\.weight$"],
                    "partition_type": "column",
                },
                {
                    "patterns": [".*linears\\.2\\.weight$"],
                    "partition_type": "skip",
                },
            ],
        }
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": 2,
                "partition_config": partition_config,
            },
            "zero_optimization": {
                "stage": 0,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        model = SequentialLinearModel(hidden_dim=16, nlayers=3)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        assert isinstance(engine.module.linears[0], LinearAllreduce)
        assert isinstance(engine.module.linears[1], LinearLayer)
        assert isinstance(engine.module.linears[2], nn.Linear)

    def test_use_default_specs_false_skips_unmatched_layers(self):
        skip_on_device()
        # Verify unmatched layers remain unsharded when defaults are disabled.
        partition_config = {
            "use_default_specs":
            False,
            "layer_specs": [
                {
                    "patterns": [".*linears\\.0\\.weight$"],
                    "partition_type": "row",
                },
                {
                    "patterns": [".*linears\\.1\\.weight$"],
                    "partition_type": "column",
                },
            ],
        }
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": 2,
                "partition_config": partition_config,
            },
            "zero_optimization": {
                "stage": 0,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        model = SequentialLinearModel(hidden_dim=16, nlayers=3)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        assert isinstance(engine.module.linears[0], LinearAllreduce)
        assert isinstance(engine.module.linears[1], LinearLayer)
        assert isinstance(engine.module.linears[2], nn.Linear)

    def test_custom_module_replacement_with_patterns(self):
        skip_on_device()
        # Verify custom linear-like modules are partitioned via patterns.
        partition_config = {
            "use_default_specs": False,
            "layer_specs": [
                {
                    "patterns": [".*custom\\.weight$"],
                    "partition_type": "column",
                },
            ],
        }
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": 2,
                "partition_config": partition_config,
            },
            "zero_optimization": {
                "stage": 0,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        model = CustomLinearModel(hidden_dim=16)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        assert isinstance(engine.module.custom, LinearLayer)

    def test_custom_pattern_disables_fused_qkv_heuristic(self):
        skip_on_device()
        # Use a qkv_proj name that would trigger the fused-QKV heuristic, then
        # verify custom patterns override that path and preserve correctness.
        torch.manual_seed(1234)
        hidden_dim = 16
        qkv_sizes = (hidden_dim, hidden_dim, hidden_dim)
        partition_config = {
            "use_default_specs":
            False,
            "layer_specs": [
                {
                    "patterns": [".*self_attn\\.qkv_proj\\.weight$"],
                    "partition_type": "column",
                    "shape": [list(qkv_sizes), -1],
                    "partition_dim": 0,
                },
            ],
        }
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": 2,
                "partition_config": partition_config,
            },
            "zero_optimization": {
                "stage": 0,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        model = QKVLinearModel(hidden_dim=hidden_dim)
        baseline = deepcopy(model).to(get_accelerator().current_device_name(), dtype=preferred_dtype())
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        qkv_layer = engine.module.self_attn.qkv_proj
        # Custom pattern should force SubParamLinearLayer (shape-based path),
        # and avoid the legacy fused-QKV heuristic despite the qkv_proj name.
        assert isinstance(qkv_layer, SubParamLinearLayer)
        assert not isinstance(qkv_layer, fused_LinearLayer)

        assert qkv_layer.partition_dim == 0
        assert qkv_layer._subparam_sizes == qkv_sizes
        assert qkv_layer._orig_weight_shape == (hidden_dim * 3, hidden_dim)

        qkv_layer.gather_params([qkv_layer.weight, qkv_layer.bias])
        torch.testing.assert_close(qkv_layer.weight, baseline.self_attn.qkv_proj.weight)
        if qkv_layer.bias is not None:
            torch.testing.assert_close(qkv_layer.bias, baseline.self_attn.qkv_proj.bias)

        torch.manual_seed(4321)
        inputs = torch.randn(2, hidden_dim, dtype=preferred_dtype(), device=get_accelerator().current_device_name())
        full_output = baseline(inputs)
        tp_output = engine.module(inputs)
        assert_close_for_preferred_dtype(tp_output, full_output)

    def test_first_match_precedence(self):
        skip_on_device()
        partition_config = {
            "use_default_specs":
            False,
            "layer_specs": [
                {
                    "patterns": [".*linears\\.0\\.weight$"],
                    "partition_type": "skip",
                },
                {
                    "patterns": [".*linears\\.0\\.weight$"],
                    "partition_type": "column",
                },
            ],
        }
        model = SequentialLinearModel(hidden_dim=16, nlayers=1)
        model = apply_autotp_with_partition_config(model, tp_size=2, partition_config=partition_config)

        assert isinstance(model.linears[0], nn.Linear)

    def test_deep_model_full_path_propagation(self):
        """Verify _replace_module propagates accumulated paths through deep hierarchies.

        Uses a 4-level-deep model (layers.N.self_attn.{q,o}_proj) with patterns
        that require intermediate path components (layers.N). Without correct
        full_name propagation, the recursive path is truncated and patterns
        that include intermediate levels will silently fail to match.
        """
        skip_on_device()
        partition_config = {
            "use_default_specs":
            False,
            "layer_specs": [
                {
                    "patterns": [r".*layers\.\d+\.self_attn\.q_proj\.weight$"],
                    "partition_type": "column",
                },
                {
                    "patterns": [r".*layers\.\d+\.self_attn\.o_proj\.weight$"],
                    "partition_type": "row",
                },
            ],
        }
        model = DeepModel(hidden_dim=16, nlayers=2)
        model = apply_autotp_with_partition_config(model, tp_size=2, partition_config=partition_config)

        # All 4 projections (2 layers x {q_proj, o_proj}) must be replaced.
        # Before the full_name fix, 0 modules were replaced because the mangled
        # path "self_attn.q_proj.weight" could not match "layers.N.self_attn...".
        for i in range(2):
            assert isinstance(model.layers[i].self_attn.q_proj, LinearLayer), \
                f"layers.{i}.self_attn.q_proj was not replaced (path propagation bug?)"
            assert isinstance(model.layers[i].self_attn.o_proj, LinearAllreduce), \
                f"layers.{i}.self_attn.o_proj was not replaced (path propagation bug?)"


def test_invalid_custom_shape_rejected():
    bad_config = {
        "layer_specs": [{
            "patterns": [".*"],
            "partition_type": "column",
            "shape": [2, [1, 1]],
        }]
    }
    with pytest.raises(ValueError, match="nested tuple only allowed at partition_dim"):
        AutoTPConfig.from_dict(bad_config)


def test_update_mp_params_uses_group_local_rank(monkeypatch):
    tp_group = object()
    autotp = object.__new__(AutoTP)
    autotp.mp_group = tp_group
    autotp.mp_size = 2
    child = nn.Module()
    child.proj = nn.Linear(1, 1, bias=False)
    setattr(child.proj.weight, DS_AUTOTP_UC_META, {})
    child.num_heads = 12

    monkeypatch.setattr(dist, "get_rank", lambda group=None: 1 if group is tp_group else 0)
    autotp.tp_meta = AutoTPMeta(num_kv_heads=3)
    autotp.update_mp_params(child)

    # Three KV groups split as [2, 1], so the second TP rank owns four query heads.
    assert child.num_heads == 4


def test_update_mp_params_shards_attributes_like_their_weights(monkeypatch):
    tp_group = object()
    autotp = object.__new__(AutoTP)
    autotp.mp_group = tp_group
    autotp.mp_size = 2
    child = nn.Module()
    child.proj = nn.Linear(1, 1, bias=False)
    setattr(child.proj.weight, DS_AUTOTP_UC_META, {})
    child.hidden_size = 12

    monkeypatch.setattr(dist, "get_rank", lambda group=None: 1 if group is tp_group else 0)
    meta = AutoTPMeta(num_kv_heads=3)
    autotp.tp_meta = meta
    autotp.update_mp_params(child, "model.layers.0.mlp")

    # MLP layers are excluded from the KV-head split, so the attribute has to follow the same
    # near-even split that the MLP weights use rather than the [2, 1] KV-group split.
    assert child.hidden_size == get_shard_size(12, 2, meta, "model.layers.0.mlp", rank=1)


def test_update_mp_params_preserves_unsharded_high_dimensional_modules(monkeypatch):
    tp_group = object()
    autotp = object.__new__(AutoTP)
    autotp.mp_group = tp_group
    autotp.mp_size = 2
    child = nn.Module()
    child.embed_dim = 12
    child.proj = nn.Conv3d(3, child.embed_dim, kernel_size=1)

    monkeypatch.setattr(dist, "get_rank", lambda group=None: 1 if group is tp_group else 0)
    autotp.tp_meta = AutoTPMeta(num_kv_heads=3)
    autotp.update_mp_params(child, "model.visual.patch_embed")

    assert child.embed_dim == 12


def test_replace_module_preserves_metadata_for_unsharded_subtree(monkeypatch):
    tp_group = object()
    autotp = object.__new__(AutoTP)
    autotp.mp_group = tp_group
    autotp.mp_size = 2
    autotp.partition_config = None
    autotp.state_dict = None
    autotp.linear_policies = {}
    autotp.prefix = ""
    visual = nn.Module()
    visual.attn = nn.Module()
    visual.attn.num_heads = 16
    visual.attn.qkv = nn.Linear(12, 36)
    visual.attn.proj = nn.Linear(12, 12)

    monkeypatch.setattr(dist, "get_rank", lambda group=None: 1 if group is tp_group else 0)
    autotp.tp_meta = AutoTPMeta()
    autotp._replace_module(visual, "model.visual")

    assert visual.attn.num_heads == 16


def test_update_mp_params_follows_actual_tp_parameter_metadata(monkeypatch):
    tp_group = object()
    autotp = object.__new__(AutoTP)
    autotp.mp_group = tp_group
    autotp.mp_size = 2
    child = nn.Module()
    child.hidden_size = 12
    child.proj = nn.Linear(12, 12)
    setattr(child.proj.weight, DS_AUTOTP_UC_META, {})

    monkeypatch.setattr(dist, "get_rank", lambda group=None: 1 if group is tp_group else 0)
    autotp.tp_meta = AutoTPMeta()
    autotp.update_mp_params(child, "model.layers.0.mlp")

    assert child.hidden_size == 6


def test_sliced_embedding_publishes_row_partition_metadata(monkeypatch):
    tp_group = object()
    autotp = object.__new__(AutoTP)
    autotp.mp_group = tp_group
    autotp.mp_size = 2
    autotp.tp_meta = AutoTPMeta()
    embedding = nn.Embedding(5, 4)

    monkeypatch.setattr(dist, "get_rank", lambda group=None: 1 if group is tp_group else 0)
    sliced = autotp._slice_embedding(embedding, "embed_tokens", False)
    model = nn.Module()
    model.embed_tokens = sliced

    uc_info = collect_autotp_universal_checkpoint_info(model)

    assert r"^embed_tokens\.weight$" in uc_info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS]
    assert r"^embed_tokens\.weight$" not in uc_info[TP_REPLICATED_PARAMETER_PATTERNS]


class TestAutoTPAlibiHelpers(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def test_mpt_alibi_covers_every_head_of_an_uneven_split(self):
        skip_on_device()
        init_tp_engine(tp_size=2)

        num_heads = 5

        class MptTransformer(nn.Module):

            def build_mpt_alibi_tensor(self, heads, sequence_length, alibi_bias_max=8, device=None):
                return torch.arange(heads, dtype=torch.float32).view(heads, 1, 1).expand(heads, 1, sequence_length)

        transformer = MptTransformer()
        install_head_sharded_helper(transformer,
                                    'build_mpt_alibi_tensor',
                                    build_mpt_alibi_tensor,
                                    meta=AutoTPMeta(num_attention_heads=num_heads, num_kv_heads=num_heads))

        alibi = transformer.build_mpt_alibi_tensor(num_heads, 3)

        # AutoTP splits 5 heads over 2 ranks as [3, 2]; an even split would give every rank
        # 2 heads and drop the last one entirely.
        expected_heads = get_shard_size_list(num_heads, dist.get_world_size(), AutoTPMeta(num_kv_heads=num_heads))
        offset = sum(expected_heads[:dist.get_rank()])
        assert alibi.shape[0] == expected_heads[dist.get_rank()]
        torch.testing.assert_close(alibi[:, 0, 0].cpu(),
                                   torch.arange(offset, offset + alibi.shape[0], dtype=torch.float32))

    def test_head_sharded_helper_leaves_the_class_untouched(self):
        skip_on_device()
        init_tp_engine(tp_size=2)

        num_heads = 5

        class MptTransformer(nn.Module):

            def build_mpt_alibi_tensor(self, heads, sequence_length, alibi_bias_max=8, device=None):
                return torch.arange(heads, dtype=torch.float32).view(heads, 1, 1).expand(heads, 1, sequence_length)

        class MptSubclass(MptTransformer):
            pass

        first = MptTransformer()
        install_head_sharded_helper(first,
                                    'build_mpt_alibi_tensor',
                                    build_mpt_alibi_tensor,
                                    meta=AutoTPMeta(num_attention_heads=num_heads, num_kv_heads=num_heads))
        expected = first.build_mpt_alibi_tensor(num_heads, 3)

        # Injecting a second model of the same architecture must not make either of them
        # delegate to the other's wrapper.
        second = MptTransformer()
        install_head_sharded_helper(second,
                                    'build_mpt_alibi_tensor',
                                    build_mpt_alibi_tensor,
                                    meta=AutoTPMeta(num_attention_heads=num_heads, num_kv_heads=num_heads))
        torch.testing.assert_close(second.build_mpt_alibi_tensor(num_heads, 3), expected)
        torch.testing.assert_close(first.build_mpt_alibi_tensor(num_heads, 3), expected)

        # A model of the same class that was never injected keeps its own method, and a
        # subclass of it inherits that method rather than an installed wrapper.
        plain = MptTransformer()
        assert plain.build_mpt_alibi_tensor(num_heads, 3).shape[0] == num_heads

        derived = MptSubclass()
        install_head_sharded_helper(derived,
                                    'build_mpt_alibi_tensor',
                                    build_mpt_alibi_tensor,
                                    meta=AutoTPMeta(num_attention_heads=num_heads, num_kv_heads=num_heads))
        torch.testing.assert_close(derived.build_mpt_alibi_tensor(num_heads, 3), expected)

    def test_head_sharded_helper_freezes_the_models_split(self):
        skip_on_device()
        init_tp_engine(tp_size=2)

        class MptTransformer(nn.Module):

            def build_mpt_alibi_tensor(self, heads, sequence_length, alibi_bias_max=8, device=None):
                return torch.arange(heads, dtype=torch.float32).view(heads, 1, 1).expand(heads, 1, sequence_length)

        num_heads = 6
        transformer = MptTransformer()
        install_head_sharded_helper(transformer,
                                    'build_mpt_alibi_tensor',
                                    build_mpt_alibi_tensor,
                                    meta=AutoTPMeta(num_attention_heads=num_heads, num_kv_heads=3))

        # The helper freezes the [4, 2] split from this model's own num_kv_heads at install
        # time.
        expected_sizes = [4, 2]
        # AutoTP replaces the model's public head count with this rank's local count.
        local_num_heads = expected_sizes[dist.get_rank()]
        alibi = transformer.build_mpt_alibi_tensor(local_num_heads, 3)

        offset = sum(expected_sizes[:dist.get_rank()])
        assert alibi.shape[0] == expected_sizes[dist.get_rank()]
        torch.testing.assert_close(alibi[:, 0, 0].cpu(),
                                   torch.arange(offset, offset + alibi.shape[0], dtype=torch.float32))

    def test_bloom_alibi_uses_original_total_after_injection(self):
        skip_on_device()
        init_tp_engine(tp_size=2)

        shard_sizes = [4, 2]
        local_num_heads = shard_sizes[dist.get_rank()]
        alibi = build_bloom_alibi_tensor(torch.ones(1, 3),
                                         local_num_heads,
                                         torch.float32,
                                         head_shard_sizes=shard_sizes,
                                         total_num_heads=6)

        assert alibi.shape == (local_num_heads, 1, 3)

    def test_alibi_mask_uses_original_total_after_injection(self):
        skip_on_device()
        init_tp_engine(tp_size=2)

        class AlibiModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.n_head = 5
                self.calls = 0
                self.cached_heads = None

            def get_alibi_mask(self, tensor, sequence_length):
                self.calls += 1
                self.cached_heads = self.n_head
                return torch.arange(self.n_head,
                                    dtype=torch.float32).view(-1, 1, 1).expand(-1, sequence_length, sequence_length)

        model = AlibiModel()
        install_head_sharded_helper(model,
                                    'get_alibi_mask',
                                    get_alibi_mask,
                                    meta=AutoTPMeta(num_attention_heads=5, num_kv_heads=5))

        shard_sizes = [3, 2]
        model.n_head = shard_sizes[dist.get_rank()]
        mask = model.get_alibi_mask(None, 3)
        second_mask = model.get_alibi_mask(None, 3)

        offset = sum(shard_sizes[:dist.get_rank()])
        assert mask.shape == (shard_sizes[dist.get_rank()], 3, 3)
        assert model.n_head == shard_sizes[dist.get_rank()]
        assert model.calls == 2
        assert model.cached_heads == 5
        torch.testing.assert_close(second_mask, mask)
        torch.testing.assert_close(mask[:, 0, 0],
                                   torch.arange(offset, offset + shard_sizes[dist.get_rank()], dtype=torch.float32))


class TestAutoTPFusedWeights(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def test_gate_up_gather_restores_sub_param_order(self):
        skip_on_device()
        init_tp_engine(tp_size=2)

        hidden_dim = 4
        gate_up_dim = 6
        torch.manual_seed(17)
        linear = nn.Linear(hidden_dim,
                           gate_up_dim * 2,
                           bias=False,
                           dtype=preferred_dtype(),
                           device=get_accelerator().current_device_name())
        full_weight = deepcopy(linear.weight.data)
        layer = GateUpPack_LinearLayer(deepcopy(linear),
                                       groups.get_tensor_model_parallel_group(),
                                       tp_meta=AutoTPMeta(num_kv_heads=3))

        # The gate and the up halves are each cut in two, so a rank-order concatenation of
        # the shards would interleave them instead of restoring the original weight.
        gathered = nn.Parameter(layer.weight.data.clone())
        layer.gather_params([gathered, None])
        torch.testing.assert_close(gathered.data, full_weight)

    def test_gate_up_fused_weight_partition(self):
        skip_on_device()
        init_tp_engine(tp_size=2)

        hidden_dim = 8
        torch.manual_seed(42)
        linear = nn.Linear(hidden_dim,
                           hidden_dim * 2,
                           bias=True,
                           dtype=preferred_dtype(),
                           device=get_accelerator().current_device_name())
        full_weight = deepcopy(linear.weight.data)
        full_bias = deepcopy(linear.bias.data)

        layer = SubParamLinearLayer(deepcopy(linear),
                                    groups.get_tensor_model_parallel_group(),
                                    shape=(2, -1),
                                    partition_dim=0,
                                    name="mlp.gate_up_proj")
        assert layer._subparam_sizes == (hidden_dim, hidden_dim)
        assert layer.weight.shape == (hidden_dim, hidden_dim)

        layer.gather_params([layer.weight, layer.bias])
        torch.testing.assert_close(layer.weight.data, full_weight)
        torch.testing.assert_close(layer.bias.data, full_bias)

    def test_gate_up_single_param_bias_gather_partition(self):
        skip_on_device()
        init_tp_engine(tp_size=2)

        hidden_dim = 8
        torch.manual_seed(43)
        linear = nn.Linear(hidden_dim,
                           hidden_dim * 2,
                           bias=True,
                           dtype=preferred_dtype(),
                           device=get_accelerator().current_device_name())
        full_bias = deepcopy(linear.bias.data)

        layer = GateUpPack_LinearLayer(deepcopy(linear), groups.get_tensor_model_parallel_group())
        bias_shard = layer.bias.data.clone()
        layer.bias.gather_params([layer.bias])
        torch.testing.assert_close(layer.bias.data, full_bias)
        layer.bias._tp_partition([layer.bias])
        torch.testing.assert_close(layer.bias.data, bias_shard)

    def test_gqa_uneven_qkv_fused_weight_partition(self):
        skip_on_device()
        init_tp_engine(tp_size=2)

        hidden_dim = 8
        q_size, k_size, v_size = 8, 4, 4
        torch.manual_seed(123)
        linear = nn.Linear(hidden_dim,
                           q_size + k_size + v_size,
                           bias=True,
                           dtype=preferred_dtype(),
                           device=get_accelerator().current_device_name())
        full_weight = deepcopy(linear.weight.data)
        full_bias = deepcopy(linear.bias.data)

        layer = SubParamLinearLayer(deepcopy(linear),
                                    groups.get_tensor_model_parallel_group(),
                                    shape=((q_size, k_size, v_size), -1),
                                    partition_dim=0,
                                    name="self_attn.qkv_proj")
        assert layer._subparam_sizes == (q_size, k_size, v_size)
        assert layer.weight.shape == ((q_size + k_size + v_size) // 2, hidden_dim)

        layer.gather_params([layer.weight, layer.bias])
        torch.testing.assert_close(layer.weight.data, full_weight)
        torch.testing.assert_close(layer.bias.data, full_bias)

    def test_subparam_linear_single_param_weight_and_bias_roundtrip(self):
        skip_on_device()
        init_tp_engine(tp_size=2)

        hidden_dim = 8
        q_size, k_size, v_size = 8, 4, 4
        torch.manual_seed(124)
        linear = nn.Linear(hidden_dim,
                           q_size + k_size + v_size,
                           bias=True,
                           dtype=preferred_dtype(),
                           device=get_accelerator().current_device_name())
        full_weight = deepcopy(linear.weight.data)
        full_bias = deepcopy(linear.bias.data)

        layer = SubParamLinearLayer(deepcopy(linear),
                                    groups.get_tensor_model_parallel_group(),
                                    shape=((q_size, k_size, v_size), -1),
                                    partition_dim=0,
                                    name="self_attn.qkv_proj")

        weight_shard = layer.weight.data.clone()
        bias_shard = layer.bias.data.clone()

        layer.weight.gather_params([layer.weight])
        torch.testing.assert_close(layer.weight.data, full_weight)
        layer.weight._tp_partition([layer.weight])
        torch.testing.assert_close(layer.weight.data, weight_shard)

        layer.bias.gather_params([layer.bias])
        torch.testing.assert_close(layer.bias.data, full_bias)
        layer.bias._tp_partition([layer.bias])
        torch.testing.assert_close(layer.bias.data, bias_shard)

    def test_subparam_allreduce_single_param_weight_and_bias_roundtrip(self):
        # engine.py hands the layer one parameter at a time, so bias has to be recognized by
        # identity rather than by its position in the list.
        skip_on_device()
        init_tp_engine(tp_size=2)

        out_dim = 8
        q_size, k_size, v_size = 8, 4, 4
        torch.manual_seed(125)
        linear = nn.Linear(q_size + k_size + v_size,
                           out_dim,
                           bias=True,
                           dtype=preferred_dtype(),
                           device=get_accelerator().current_device_name())
        full_weight = deepcopy(linear.weight.data)
        full_bias = deepcopy(linear.bias.data)

        layer = SubParamLinearAllreduce(deepcopy(linear),
                                        groups.get_tensor_model_parallel_group(),
                                        shape=(-1, (q_size, k_size, v_size)),
                                        partition_dim=1,
                                        name="self_attn.o_proj")

        weight_shard = layer.weight.data.clone()
        # Row parallel replicates the bias, so partitioning must leave it at full size.
        torch.testing.assert_close(layer.bias.data, full_bias)

        layer.gather_params([layer.weight])
        torch.testing.assert_close(layer.weight.data, full_weight)
        layer._tp_partition([layer.weight])
        torch.testing.assert_close(layer.weight.data, weight_shard)

        layer.gather_params([layer.bias])
        torch.testing.assert_close(layer.bias.data, full_bias)
        layer._tp_partition([layer.bias])
        torch.testing.assert_close(layer.bias.data, full_bias)

    def test_gather_uses_the_layers_own_shard_widths(self):
        # The gather has to undo the exact uneven cut this layer was partitioned with, so it
        # reads the widths recorded on the layer rather than re-deriving a split.
        skip_on_device()
        init_tp_engine(tp_size=2)

        hidden_dim = 8
        head_size = 12
        torch.manual_seed(7)
        linear = nn.Linear(hidden_dim,
                           head_size * 3,
                           bias=True,
                           dtype=preferred_dtype(),
                           device=get_accelerator().current_device_name())
        full_weight = deepcopy(linear.weight.data)
        full_bias = deepcopy(linear.bias.data)

        layer = SubParamLinearLayer(deepcopy(linear),
                                    groups.get_tensor_model_parallel_group(),
                                    shape=((head_size, head_size, head_size), -1),
                                    partition_dim=0,
                                    name="self_attn.qkv_proj",
                                    tp_meta=AutoTPMeta(num_kv_heads=3))
        assert layer._subparam_shard_widths == [[8, 4], [8, 4], [8, 4]]

        layer.gather_params([layer.weight, layer.bias])
        torch.testing.assert_close(layer.weight.data, full_weight)
        torch.testing.assert_close(layer.bias.data, full_bias)

    def test_gqa_uneven_qkv_fused_forward(self):
        skip_on_device()
        groups._init_tp_mesh_device(tensor_model_parallel_size=2)

        hidden_dim = 8
        q_size, k_size, v_size = 8, 4, 4
        torch.manual_seed(321)
        linear = nn.Linear(hidden_dim,
                           q_size + k_size + v_size,
                           bias=True,
                           dtype=preferred_dtype(),
                           device=get_accelerator().current_device_name())
        layer = SubParamLinearLayer(deepcopy(linear),
                                    groups.get_tensor_model_parallel_group(),
                                    shape=((q_size, k_size, v_size), -1),
                                    partition_dim=0,
                                    name="self_attn.qkv_proj")

        torch.manual_seed(42)
        inputs = torch.randn(2, hidden_dim, dtype=preferred_dtype(), device=get_accelerator().current_device_name())
        full_output = linear(inputs)
        tp_output = layer(inputs)

        gathered_output = gather_subparam_output(tp_output, (q_size, k_size, v_size),
                                                 groups.get_tensor_model_parallel_group())
        assert_close_for_preferred_dtype(gathered_output, full_output)


# Multi-model (teacher + student in one process) and multimodal-config regressions live in
# tests/unit/v1/autotp/test_autotp_multiple_models.py so they run on the GPU workflow too.
