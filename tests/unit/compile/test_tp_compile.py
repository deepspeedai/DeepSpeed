# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.compile.init_tp import AUTOTP_MIN_TORCH_VERSION
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
        "patterns": [".*\\.head\\.weight$"],
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


class TestAutoTPCompileEquivalence(DistributedTest):
    """The compile pass must reproduce the module-injection AutoTP path exactly.

    Both paths shard the weights the same way, so the compiled model is compared against the
    module-level collectives it replaces rather than against a single-device run.
    """

    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.sequential
    def test_matches_module_injection(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        device = torch.device(get_accelerator().current_device_name())
        reference_engine = build_engine(self.world_size, use_compile_pass=False)
        compiled_engine = build_engine(self.world_size, use_compile_pass=True)

        # The TP group must see identical inputs on every rank.
        torch.manual_seed(1234)
        x = torch.randn(1, 8, HIDDEN_DIM, device=device, dtype=torch.float32, requires_grad=True)
        compiled_x = x.detach().clone().requires_grad_(True)

        reference_out = reference_engine(x)
        compiled_out = compiled_engine(compiled_x)
        assert torch.allclose(reference_out, compiled_out, atol=1e-5), \
            "AutoTP compile pass changed the forward result"

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


class TestAutoTPCompileRejectsGatherOutput(DistributedTest):
    """gather_output layers must be rejected instead of silently losing a collective.

    Their gather reads shard sizes back into Python, which the full graph the pass needs cannot
    capture, so the pass can neither emit the collectives nor leave them to the module.
    """

    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.sequential
    def test_gather_output_raises(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        with pytest.raises(NotImplementedError, match="gather_output"):
            build_engine(self.world_size, use_compile_pass=True, gather_output_head=True)


class TestAutoTPCompileRejectsUnsupportedCombinations(DistributedTest):

    world_size = 1

    def test_autotp_with_zero_pass_raises(self):
        model = MLPModel()
        config = build_config(tp_size=1, use_compile_pass=True)
        config["compile"]["passes"] = ["autotp", "z1"]
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        with pytest.raises(NotImplementedError, match="cannot yet be combined"):
            engine.compile()
