# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from packaging.version import Version

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.compile.init_tp import AUTOTP_MIN_TORCH_VERSION, BROKEN_TRANSFORMERS_MOE_VERSIONS
from deepspeed.utils import groups
from deepspeed.utils.torch import required_torch_version

from unit.common import DistributedTest

pytestmark = pytest.mark.skipif(not required_torch_version(min_version=AUTOTP_MIN_TORCH_VERSION),
                                reason=f"The AutoTP compile pass requires PyTorch >= {AUTOTP_MIN_TORCH_VERSION}")

HIDDEN_DIM = 64
INTERMEDIATE_DIM = 128


class MLPBlock(torch.nn.Module):
    """Llama-style MLP: gate/up are column-parallel and down is row-parallel."""

    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(HIDDEN_DIM, INTERMEDIATE_DIM, bias=False)
        self.up_proj = torch.nn.Linear(HIDDEN_DIM, INTERMEDIATE_DIM, bias=False)
        self.down_proj = torch.nn.Linear(INTERMEDIATE_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x):
        # The residual is what makes this block interesting for the pass: x feeds the two
        # column-parallel matmuls and the addition, and only the matmuls may be routed through the
        # backward all-reduce. Reducing the residual gradient too would scale it by the TP size.
        return x + self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class MLPModel(torch.nn.Module):

    def __init__(self, nlayers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([MLPBlock() for _ in range(nlayers)])
        self.head = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def build_config(tp_size, use_compile_pass, gather_output_head=False):
    head_spec = {
        "patterns": ["(.*\\.)?head\\.weight$"],
        "partition_type": "column",
        "gather_output": gather_output_head,
    }
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-6
            }
        },
        "tensor_parallel": {
            "autotp_size": tp_size,
            "partition_config": {
                "use_default_specs":
                False,
                "layer_specs": [{
                    "patterns": [".*\\.gate_proj\\.weight$", ".*\\.up_proj\\.weight$"],
                    "partition_type": "column",
                }, {
                    "patterns": [".*\\.down_proj\\.weight$"],
                    "partition_type": "row",
                }, head_spec],
            },
        },
        "zero_optimization": {
            "stage": 0,
        },
    }
    if use_compile_pass:
        config["compile"] = {"deepcompile": True, "passes": ["autotp"]}
    return config


def build_engine(tp_size, use_compile_pass, gather_output_head=False):
    # Both engines are built from the same seed so they hold identical shards, which lets the
    # gradients be compared directly without gathering them first.
    torch.manual_seed(42)
    model = MLPModel()
    engine, _, _, _ = deepspeed.initialize(model=model,
                                           model_parameters=model.parameters(),
                                           config=build_config(tp_size, use_compile_pass, gather_output_head))
    if use_compile_pass:
        engine.compile()
    return engine


def test_gather_from_tp_region_supports_uneven_shards():
    from deepspeed.compile.custom_ops import tp_collectives

    local = torch.tensor([[1.0, 2.0, 3.0]])
    with patch.object(tp_collectives, 'get_tp_group', return_value='group'), \
         patch.object(dist, 'get_world_size', return_value=2), \
         patch.object(dist, 'get_rank', return_value=1), \
         patch.object(dist, 'all_gather_into_tensor',
                      side_effect=lambda out, inp, group: out.copy_(torch.tensor([[[7.0, 8.0, 0.0]],
                                                                                 [[1.0, 2.0, 3.0]]]).reshape(
                                                                                     2 * inp.shape[0], *inp.shape[1:]))):
        gathered = tp_collectives.gather_from_tp_region(local, [2, 3])

    assert torch.equal(gathered, torch.tensor([[7.0, 8.0, 1.0, 2.0, 3.0]]))


class TestAutoTPCompileEquivalence(DistributedTest):
    """The compile pass must reproduce the module-injection AutoTP path exactly.

    Both paths shard the weights the same way, so the compiled model is compared against the
    module-level collectives it replaces rather than against a single-device run.
    """

    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.sequential
    @pytest.mark.parametrize("gather_output_head", [False, True])
    def test_matches_module_injection(self, gather_output_head):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        device = torch.device(get_accelerator().current_device_name())
        reference_engine = build_engine(self.world_size, use_compile_pass=False, gather_output_head=gather_output_head)
        compiled_engine = build_engine(self.world_size, use_compile_pass=True, gather_output_head=gather_output_head)

        # The TP group must see identical inputs on every rank.
        torch.manual_seed(1234)
        x = torch.randn(1, 8, HIDDEN_DIM, device=device, dtype=torch.float32, requires_grad=True)
        compiled_x = x.detach().clone().requires_grad_(True)

        reference_out = reference_engine(x)
        compiled_out = compiled_engine(compiled_x)
        assert torch.allclose(reference_out, compiled_out, atol=1e-5), \
            "AutoTP compile pass changed the forward result"

        # Comparing the two paths cannot catch a gather that both of them dropped, so the width of
        # the head output is checked against the partitioning it was configured with.
        expected_head_width = HIDDEN_DIM if gather_output_head else HIDDEN_DIM // self.world_size
        assert compiled_out.shape[-1] == expected_head_width, \
            f"Expected a head output of width {expected_head_width}, got {compiled_out.shape[-1]}"

        reference_engine.backward(reference_out.sum())
        compiled_engine.backward(compiled_out.sum())

        # A missing or duplicated collective usually leaves the forward pass intact and only
        # corrupts gradients, so the gradients are what this test really checks.
        for (name, reference_param), (_, compiled_param) in zip(reference_engine.module.named_parameters(),
                                                                compiled_engine.module.named_parameters()):
            assert torch.allclose(reference_param.grad, compiled_param.grad, atol=1e-5), \
                f"AutoTP compile pass changed the gradient of {name}"

        assert torch.allclose(x.grad, compiled_x.grad, atol=1e-5), \
            "AutoTP compile pass changed the gradient reaching the model input"


class FusedQKVBlock(torch.nn.Module):
    """Attention-shaped block whose projections are both shaped sub-param layers.

    AutoTP injects a fused QKV projection as SubParamLinearLayer rather than LinearLayer, which is
    the case an exact-type check in the pass would miss. The output projection is likewise given a
    shape so it is injected as SubParamLinearAllreduce, whose forward must give up its module-level
    all-reduce once the pass emits one into the graph — keeping both would reduce twice.
    """

    def __init__(self):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(HIDDEN_DIM, 3 * HIDDEN_DIM, bias=False)
        self.o_proj = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x):
        query, key, value = self.qkv_proj(x).chunk(3, dim=-1)
        return x + self.o_proj(query * key + value)


class FusedQKVModel(torch.nn.Module):

    def __init__(self, nlayers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([FusedQKVBlock() for _ in range(nlayers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def build_fused_qkv_engine(tp_size, use_compile_pass):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-6
            }
        },
        "tensor_parallel": {
            "autotp_size": tp_size,
            "partition_config": {
                "use_default_specs":
                False,
                "layer_specs": [
                    {
                        "patterns": [".*\\.qkv_proj\\.weight$"],
                        "partition_type": "column",
                        "shape": [3, -1],
                        "partition_dim": 0,
                    },
                    {
                        # A single sub-param spanning the input dim shards exactly like the plain row
                        # split; the shape's only effect is injecting SubParamLinearAllreduce.
                        "patterns": [".*\\.o_proj\\.weight$"],
                        "partition_type": "row",
                        "shape": [-1, 1],
                        "partition_dim": 1,
                    }
                ],
            },
        },
        "zero_optimization": {
            "stage": 0
        },
    }
    if use_compile_pass:
        config["compile"] = {"deepcompile": True, "passes": ["autotp"]}

    torch.manual_seed(42)
    model = FusedQKVModel()
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    if use_compile_pass:
        engine.compile()
    return engine


class TestAutoTPCompileLayerVariants(DistributedTest):
    """Every injected tensor-parallel layer variant has to be rewritten, not just the base classes.

    AutoTP injects a family of variants per partitioning style (fused QKV, conv, packed gate/up,
    Yuan, shaped sub-params). A variant left on the module-level path is not merely unoptimized:
    the pass compiles with fullgraph=True, and ColumnParallel's forward is an identity, so tracing
    folds it away and drops its backward all-reduce. The forward still matches and only the
    gradients are wrong.
    """

    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.sequential
    def test_fused_qkv_matches_module_injection(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        device = torch.device(get_accelerator().current_device_name())
        reference_engine = build_fused_qkv_engine(self.world_size, use_compile_pass=False)
        compiled_engine = build_fused_qkv_engine(self.world_size, use_compile_pass=True)

        from deepspeed.module_inject.layers import SubParamLinearAllreduce, SubParamLinearLayer
        qkv = compiled_engine.module.layers[0].qkv_proj
        assert isinstance(qkv, SubParamLinearLayer), f"expected a shaped sub-param layer, got {type(qkv).__name__}"
        assert qkv.defer_collectives_to_compiler, "the fused QKV layer was left on the module-level path"
        o_proj = compiled_engine.module.layers[0].o_proj
        assert isinstance(o_proj, SubParamLinearAllreduce), f"expected a shaped row layer, got {type(o_proj).__name__}"
        assert o_proj.defer_collectives_to_compiler, "the shaped row layer was left on the module-level path"

        torch.manual_seed(1234)
        x = torch.randn(1, 8, HIDDEN_DIM, device=device, dtype=torch.float32, requires_grad=True)
        compiled_x = x.detach().clone().requires_grad_(True)

        reference_out = reference_engine(x)
        compiled_out = compiled_engine(compiled_x)
        assert torch.allclose(reference_out, compiled_out, atol=1e-5)

        reference_engine.backward(reference_out.sum())
        compiled_engine.backward(compiled_out.sum())

        for (name, reference_param), (_, compiled_param) in zip(reference_engine.module.named_parameters(),
                                                                compiled_engine.module.named_parameters()):
            assert torch.allclose(reference_param.grad, compiled_param.grad, atol=1e-5), \
                f"AutoTP compile pass changed the gradient of {name}"

        assert torch.allclose(x.grad, compiled_x.grad, atol=1e-5), \
            "AutoTP compile pass changed the gradient reaching the model input"


class TestAutoTPCompileMoE(DistributedTest):
    """A mixture-of-experts model must survive the pass.

    Mixtral's expert and router entries use tp_plan styles AutoTP does not implement, and the
    converter rejects a plan containing any unsupported style rather than applying it partially,
    so the attention sharding is spelled out as an explicit partition config. The experts, router
    and lm_head stay replicated, so the pass only has to leave them alone and rewrite the
    attention projections.
    """

    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.sequential
    def test_mixtral_matches_module_injection(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "MixtralForCausalLM"):
            pytest.skip("transformers build has no Mixtral")
        tf_version = Version(transformers.__version__)
        # The 4.x eager experts route tokens through .nonzero(), a dynamic-shape op no full graph
        # can capture; the static-shape batched_mm implementation exists from 5.0.
        if tf_version.major < 5:
            pytest.skip("the static-shape batched_mm experts implementation requires transformers >= 5")
        first_broken, first_fixed = BROKEN_TRANSFORMERS_MOE_VERSIONS
        if Version(first_broken) <= tf_version < Version(first_fixed):
            pytest.skip(f"transformers {first_broken}..{first_fixed} mutate the MoE routing tensor in place "
                        "(huggingface/transformers#45621, fixed by #45634)")
        if not required_torch_version(min_version=2.7):
            pytest.skip("tracing the transformers input-check wrapper requires torch >= 2.7")

        def build(use_compile_pass):
            config = transformers.MixtralConfig(vocab_size=256,
                                                hidden_size=HIDDEN_DIM,
                                                intermediate_size=INTERMEDIATE_DIM,
                                                num_hidden_layers=2,
                                                num_attention_heads=8,
                                                num_key_value_heads=8,
                                                num_local_experts=4,
                                                num_experts_per_tok=2,
                                                max_position_embeddings=64,
                                                use_cache=False,
                                                tie_word_embeddings=False)
            config._attn_implementation = "sdpa"
            # The default eager experts route tokens with data-dependent indexing, which a full
            # graph cannot capture; batched_mm is the static-shape implementation.
            config._experts_implementation = "batched_mm"
            torch.manual_seed(42)
            model = transformers.MixtralForCausalLM(config)
            ds_config = {
                "train_micro_batch_size_per_gpu": 1,
                "optimizer": {
                    "type": "Adam",
                    "params": {
                        "lr": 1e-6
                    }
                },
                "tensor_parallel": {
                    "autotp_size": self.world_size,
                    "partition_config": {
                        "use_default_specs":
                        False,
                        "layer_specs": [{
                            "patterns": [
                                ".*\\.self_attn\\.q_proj\\.weight$",
                                ".*\\.self_attn\\.k_proj\\.weight$",
                                ".*\\.self_attn\\.v_proj\\.weight$",
                            ],
                            "partition_type":
                            "column",
                        }, {
                            "patterns": [".*\\.self_attn\\.o_proj\\.weight$"],
                            "partition_type": "row",
                        }],
                    },
                },
                "zero_optimization": {
                    "stage": 0
                },
            }
            if use_compile_pass:
                ds_config["compile"] = {"deepcompile": True, "passes": ["autotp"]}
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
            if use_compile_pass:
                engine.compile()
            return engine

        device = torch.device(get_accelerator().current_device_name())
        reference_engine = build(False)
        compiled_engine = build(True)

        torch.manual_seed(1234)
        input_ids = torch.randint(0, 256, (1, 16), device=device)
        labels = torch.randint(0, 256, (1, 16), device=device)

        def step(engine):
            logits = engine(input_ids=input_ids, use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256).float(), labels.reshape(-1))
            engine.backward(loss)
            return loss

        reference_loss = step(reference_engine)
        compiled_loss = step(compiled_engine)
        assert torch.allclose(reference_loss, compiled_loss, atol=1e-5), \
            f"MoE loss changed: {reference_loss.item()} vs {compiled_loss.item()}"

        for (name, reference_param), (_, compiled_param) in zip(reference_engine.module.named_parameters(),
                                                                compiled_engine.module.named_parameters()):
            if reference_param.grad is None or compiled_param.grad is None:
                continue
            assert torch.allclose(reference_param.grad, compiled_param.grad, atol=1e-5), \
                f"AutoTP compile pass changed the gradient of {name}"


class TestAutoTPCompileDataParallelGradients(DistributedTest):
    """Gradients must still be reduced across data-parallel replicas.

    The engine skips its own gradient reduction when DeepCompile is active because the ZeRO passes
    emit that reduction into the graph. The AutoTP pass does not: its collectives only sum partial
    results inside a TP group and never touch the DP axis. Only a run with more than one
    data-parallel replica shows whether the reduction still happens.
    """

    world_size = 4
    non_daemonic_procs = True

    @pytest.mark.sequential
    def test_gradients_are_reduced_across_dp_group(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        tp_size = 2
        device = torch.device(get_accelerator().current_device_name())
        engine = build_engine(tp_size, use_compile_pass=True)

        dp_group = groups.get_data_parallel_group()
        assert dist.get_world_size(group=dp_group) == self.world_size // tp_size

        # Every data-parallel replica gets different data, so an unreduced gradient differs between
        # replicas. Ranks inside a TP group must still agree, hence seeding on the replica index.
        replica_index = dist.get_rank() // tp_size
        torch.manual_seed(1234 + replica_index)
        x = torch.randn(1, 8, HIDDEN_DIM, device=device, dtype=torch.float32)

        out = engine(x)
        engine.backward(out.sum())

        for name, param in engine.module.named_parameters():
            gathered = [torch.empty_like(param.grad) for _ in range(dist.get_world_size(group=dp_group))]
            dist.all_gather(gathered, param.grad.contiguous(), group=dp_group)
            assert torch.allclose(gathered[0], gathered[-1], atol=1e-5), \
                f"Gradient of {name} was not reduced across the data-parallel group"


def _make_tp_layer(cls):
    # Only the flag logic is under test, so the layer is built without weights or a process group.
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.mp_group = object()
    layer.defer_collectives_to_compiler = False
    return layer


def test_defer_collectives_is_all_or_nothing():
    """A rejected layer must leave every other layer's collectives untouched.

    If the rejection is raised after some flags are already set and a caller catches it to fall
    back to eager execution, the flagged layers would silently skip their collectives.
    """
    from deepspeed.compile.passes.tp_compile import defer_collectives_to_compiler
    from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer, LmHeadLinearAllreduce

    model = torch.nn.Module()
    model.column = _make_tp_layer(LinearLayer)
    model.row = _make_tp_layer(LinearAllreduce)
    model.head = _make_tp_layer(LmHeadLinearAllreduce)

    with pytest.raises(NotImplementedError, match="cannot rewrite"):
        defer_collectives_to_compiler(model)

    assert not model.column.defer_collectives_to_compiler
    assert not model.row.defer_collectives_to_compiler


@pytest.mark.parametrize("experts_implementation, should_raise", [("batched_mm", True), ("eager", False),
                                                                  ("grouped_mm", False)])
def test_broken_transformers_moe_raises(monkeypatch, experts_implementation, should_raise):
    """Affected releases must reject only MoE models selecting the broken batched_mm path."""
    transformers = pytest.importorskip("transformers")
    from deepspeed.compile.init_tp import _check_broken_transformers_moe

    first_broken, first_fixed = BROKEN_TRANSFORMERS_MOE_VERSIONS
    monkeypatch.setattr(transformers, "__version__", first_broken)

    moe_model = torch.nn.Module()
    moe_model.experts = torch.nn.Module()
    # The markers transformers' experts-implementation decorator leaves on a wrapped module.
    moe_model.experts._apply_gate = lambda x: x
    moe_model.experts.is_concatenated = True
    moe_model.experts.config = SimpleNamespace(_experts_implementation=experts_implementation)
    if should_raise:
        with pytest.raises(RuntimeError, match="45634"):
            _check_broken_transformers_moe(moe_model)
    else:
        _check_broken_transformers_moe(moe_model)

    _check_broken_transformers_moe(torch.nn.Module())

    monkeypatch.setattr(transformers, "__version__", first_fixed)
    _check_broken_transformers_moe(moe_model)


class TestAutoTPCompileRejectsUnsupportedCombinations(DistributedTest):

    world_size = 1

    def test_autotp_with_zero_pass_raises(self):
        model = MLPModel()
        config = build_config(tp_size=1, use_compile_pass=True)
        config["compile"]["passes"] = ["autotp", "z1"]
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        with pytest.raises(NotImplementedError, match="cannot yet be combined"):
            engine.compile()


class ACMLPModel(torch.nn.Module):
    """Same model as MLPModel, but each block runs under activation checkpointing.

    use_reentrant=False is what HuggingFace's gradient_checkpointing_enable defaults to, and it is
    the variant Dynamo captures into a tag_activation_checkpoint subgraph rather than graph-breaking
    on. That subgraph is the failure mode this exercises: a pass that walks only the top-level graph
    never sees the parallel matmuls inside it, while their module-level collectives have already
    been switched off.
    """

    def __init__(self, nlayers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([MLPBlock() for _ in range(nlayers)])
        self.head = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x):
        import torch.utils.checkpoint as ckpt
        for layer in self.layers:
            x = ckpt.checkpoint(layer, x, use_reentrant=False)
        return self.head(x)


def build_ac_engine(tp_size, use_compile_pass):
    torch.manual_seed(42)
    model = ACMLPModel()
    engine, _, _, _ = deepspeed.initialize(model=model,
                                           model_parameters=model.parameters(),
                                           config=build_config(tp_size, use_compile_pass))
    if use_compile_pass:
        engine.compile()
    return engine


class TestAutoTPCompileActivationCheckpointing(DistributedTest):
    """Layers inside a checkpointed block must still get their collectives.

    Dynamo lifts a checkpointed region into a child GraphModule. Walking only the top-level graph
    left those layers with neither a module-level collective nor a graph-level one, which is not a
    missed optimization: the row-parallel outputs stay partial sums, so the forward and the
    gradients are both wrong and nothing raises.
    """

    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.sequential
    def test_checkpointed_layers_match_module_injection(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        device = torch.device(get_accelerator().current_device_name())
        reference_engine = build_ac_engine(self.world_size, use_compile_pass=False)
        compiled_engine = build_ac_engine(self.world_size, use_compile_pass=True)

        torch.manual_seed(1234)
        x = torch.randn(1, 8, HIDDEN_DIM, device=device, dtype=torch.float32, requires_grad=True)
        compiled_x = x.detach().clone().requires_grad_(True)

        reference_out = reference_engine(x)
        compiled_out = compiled_engine(compiled_x)
        assert torch.allclose(reference_out, compiled_out, atol=1e-5), \
            "a dropped row-parallel all-reduce leaves the forward a partial sum"

        reference_engine.backward(reference_out.sum())
        compiled_engine.backward(compiled_out.sum())

        for (name, reference_param), (_, compiled_param) in zip(reference_engine.module.named_parameters(),
                                                                compiled_engine.module.named_parameters()):
            assert torch.allclose(reference_param.grad, compiled_param.grad, atol=1e-5), \
                f"AutoTP compile pass changed the gradient of {name}"


def test_unhandled_parallel_layer_raises():
    """A layer reached in the graph but given no collective must stop the compile, not run wrong.

    The pass switches the module-level collectives off before it rewrites anything, so a layer it
    fails to recognize does not merely stay unoptimized -- it stops communicating, and both the
    forward and the gradients go quietly wrong.
    """
    from deepspeed.compile.passes import tp_compile
    from deepspeed.module_inject.layers import LinearAllreduce

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    matmul = graph.call_function(torch.matmul, args=(x, x))
    matmul.meta["nn_module_stack"] = {"key": ("layers.0.down_proj", LinearAllreduce)}
    graph.output(matmul)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    original_targets = set(tp_compile._MATMUL_TARGETS)
    try:
        # Stand in for a matmul that traced to an op the pass does not know about.
        tp_compile._MATMUL_TARGETS.clear()
        with pytest.raises(RuntimeError, match="without inserting a collective"):
            tp_compile.pass_insert_tp_collectives(gm, (), deferred_names={"layers.0.down_proj"})
    finally:
        tp_compile._MATMUL_TARGETS.clear()
        tp_compile._MATMUL_TARGETS.update(original_targets)

    # With the target restored the same graph is rewritten cleanly.
    tp_compile.pass_insert_tp_collectives(gm, (), deferred_names={"layers.0.down_proj"})
    assert any(n.target is tp_compile.ROW_PARALLEL_OP for n in gm.graph.nodes)


def test_deferred_layer_run_eagerly_raises():
    """Running a deferred layer outside a compiled region must raise rather than skip the collective."""
    from deepspeed.module_inject.layers import LinearAllreduce

    layer = _make_tp_layer(LinearAllreduce)
    layer.defer_collectives_to_compiler = True
    layer.weight = torch.nn.Parameter(torch.randn(4, 4))
    layer.bias = None

    with pytest.raises(RuntimeError, match="running eagerly"):
        layer(torch.randn(2, 4))


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("L['self']._modules['layers']._modules['0'].down_proj", "layers.0.down_proj"),
        ("L['self']._modules['layers']._modules['0']", "layers.0"),
        ("L['self'].head", "head"),
        ("L['self']._modules['model']._modules['layers']._modules['0']._modules['self_attn']._modules['q_proj']",
         "model.layers.0.self_attn.q_proj"),
        ("self.head", "head"),
        ("layers.0.down_proj", "layers.0.down_proj"),
        # The root-prefix strip must be anchored: these start with "self" but the module is not the root.
        ("self_attn.q_proj", "self_attn.q_proj"),
        ("selfie.q_proj", "selfie.q_proj"),
    ])
def test_normalize_fqn_on_real_dynamo_strings(raw, expected):
    """The deferred-layer comparison is only as good as this function.

    An unanchored prefix strip turns 'self_attn.q_proj' into '_attn.q_proj', which then fails to
    match deferred_names -- and because the check intersects with that set, the miss is erased
    rather than reported. Silent, and in the unsafe direction.
    """
    from deepspeed.compile.passes.tp_compile import _normalize_fqn
    assert _normalize_fqn(raw) == expected


def test_iter_graphs_finds_a_graphmodule_under_a_plain_module():
    """A GraphModule can be nested under a plain nn.Module, which a GraphModule-only recursion misses."""
    from deepspeed.compile.passes.tp_compile import iter_graphs

    inner = torch.fx.symbolic_trace(torch.nn.Linear(4, 4))
    wrapper = torch.nn.Module()
    wrapper.sub = inner  # GraphModule under a plain Module
    root = torch.fx.symbolic_trace(torch.nn.Linear(4, 4))
    root.wrapper = wrapper

    assert len(list(iter_graphs(root))) == 2, "iter_graphs did not descend past the plain nn.Module"


def test_collective_is_inserted_inside_a_checkpointed_subgraph():
    """CPU-only cover for the traversal fix, so it is not gated behind a 2-GPU run.

    Also pins the coverage check: with the walk restricted to the top-level graph, the layers inside
    the checkpointed block are reached (their lifted weights appear as placeholders there) but never
    handled, so the pass must raise rather than silently emit nothing.
    """
    import torch.utils.checkpoint as ckpt
    from deepspeed.compile.passes import tp_compile
    from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer

    def make(cls, out_features, in_features):
        layer = _make_tp_layer(cls)
        layer.defer_collectives_to_compiler = True
        layer.gather_output = False
        layer.support_training = True
        layer.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        layer.bias = None
        return layer

    class Block(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.gate_proj = make(LinearLayer, 64, 32)
            self.down_proj = make(LinearAllreduce, 32, 64)

        def forward(self, x):
            return x + self.down_proj(torch.nn.functional.silu(self.gate_proj(x)))

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])

        def forward(self, x):
            for layer in self.layers:
                x = ckpt.checkpoint(layer, x, use_reentrant=False)
            return x

    deferred = {"layers.0.gate_proj", "layers.0.down_proj"}
    seen = {}

    def backend(gm, example_inputs):
        tp_compile.pass_insert_tp_collectives(gm, example_inputs, deferred_names=deferred)
        seen["inserted"] = sum(1 for graph in tp_compile.iter_graphs(gm) for node in graph.nodes
                               if node.op == "call_function" and "autotp" in str(node.target))
        seen["top_level"] = sum(1 for node in gm.graph.nodes
                                if node.op == "call_function" and "autotp" in str(node.target))

        original = tp_compile.iter_graphs
        tp_compile.iter_graphs = lambda module: iter([module.graph])
        try:
            tp_compile.pass_insert_tp_collectives(gm, example_inputs, deferred_names=deferred)
            seen["raised_without_recursion"] = False
        except RuntimeError:
            seen["raised_without_recursion"] = True
        finally:
            tp_compile.iter_graphs = original
        # Only the rewrite is under test. Returning a stub avoids executing the collectives, which
        # would need a real process group and turn this into a distributed test.
        return lambda *args, **kwargs: [torch.zeros(2, 4, 32)]

    torch._dynamo.reset()
    torch.compile(Model(), backend=backend, fullgraph=True)(torch.randn(2, 4, 32))

    assert seen["inserted"] > 0, "no collective was inserted into the checkpointed subgraph"
    assert seen["top_level"] == 0, "the collectives should be inside the subgraph, not the root graph"
    assert seen["raised_without_recursion"], \
        "restricted to the top-level graph the pass inserted nothing and did not raise"


def test_apply_autotp_preserves_the_two_argument_pass_contract():
    """`passes` is an extension point; its contract is (gm, real_inputs).

    Threading extra context to every pass unconditionally raises TypeError on any pass written
    against that contract, before the pass even runs.
    """
    from deepspeed.compile.passes.tp_compile import apply_autotp

    gm = torch.fx.symbolic_trace(torch.nn.Linear(4, 4))
    seen = []

    def two_argument_pass(gm, real_inputs):
        seen.append("two")

    def context_aware_pass(gm, real_inputs, deferred_names=None):
        seen.append(("context", deferred_names))

    def var_keyword_pass(gm, real_inputs, **kwargs):
        seen.append(("kwargs", kwargs.get("deferred_names")))

    apply_autotp(gm, (),
                 passes=[two_argument_pass, context_aware_pass, var_keyword_pass],
                 deferred_names={"layers.0.down_proj"})

    assert seen == ["two", ("context", {"layers.0.down_proj"}), ("kwargs", {"layers.0.down_proj"})]
