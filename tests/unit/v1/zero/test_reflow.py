# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Contract tests for Reflow, the asynchronous CPU-offload optimizer for ZeRO-3.

Reflow is meant to be a bit-exact drop-in for ZeRO-Offload (ZeRO-3 + CPU optimizer offload), so every training
test runs the same model and batches through plain ZeRO-Offload as the oracle and compares exactly.
"""

import copy
import os
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from unittest.mock import Mock, patch
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.constants import FP32_FLAT_GROUPS
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.lion import DeepSpeedCPULion
from deepspeed.ops.op_builder import AsyncIOBuilder, CPUAdamBuilder, CPULionBuilder
from deepspeed.ops.op_builder.cpu.reflow_cpu_adam import ReflowCPUAdamBuilder
from deepspeed.ops.op_builder.cpu.reflow_cpu_lion import ReflowCPULionBuilder
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.reflow.reflow_cpu_adam import ReflowCPUAdam
from deepspeed.runtime.reflow.reflow_cpu_lion import ReflowCPULion
from deepspeed.runtime.reflow.reflow_stage3 import ReflowOptimizer_Stage3
from deepspeed.runtime.reflow import reflow_utils
from deepspeed.runtime.utils import combine_grad_norm_groups
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam
from deepspeed.runtime.zero.utils import ZeRORuntimeException
from deepspeed.utils import (safe_get_full_fp32_param, safe_get_full_optimizer_state, safe_get_local_grad,
                             safe_set_local_grad, safe_update_full_grad_vectorized)
from deepspeed.utils.numa import parse_range_list
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


def minimal_reflow_config(reflow=None, **zero_overrides):
    """The smallest config that enables Reflow, for the checks that run when the config is parsed."""
    zero_optimization = {"stage": 3, "offload_optimizer": {"device": "cpu"}, "reflow": reflow or {}}
    zero_optimization.update(zero_overrides)
    return {"train_batch_size": 1, "bf16": {"enabled": True}, "zero_optimization": zero_optimization}


def test_reflow_block_configures_the_optimizer():
    config = DeepSpeedConfig(minimal_reflow_config({"state_update_cores": 4}))
    assert config.zero_config.reflow.state_update_cores == 4
    assert config.zero_config.reflow.bucketwise_cores_per_worker is None
    assert config.zero_config.reflow.h2d_chunk_size_mib == 16
    assert config.zero_config.reflow.bucketwise_worker_affinity == "thread"
    assert config.zero_config.reflow.pin_main_thread is True
    assert DeepSpeedConfig(minimal_reflow_config()).zero_config.reflow.state_update_cores is None

    for cores in (1, 4, 8):
        config = DeepSpeedConfig(minimal_reflow_config({"bucketwise_cores_per_worker": cores}))
        assert config.zero_config.reflow.bucketwise_cores_per_worker == cores

    with pytest.raises(ValueError):
        DeepSpeedConfig(minimal_reflow_config({"state_update_cores": 0}))


@pytest.mark.parametrize("size", [0, -1])
def test_reflow_rejects_nonpositive_h2d_chunks(size):
    with pytest.raises(ValueError, match="h2d_chunk_size_mib"):
        DeepSpeedConfig(minimal_reflow_config({"h2d_chunk_size_mib": size}))


def test_cpu_placement_options_are_validated():
    options = {
        "main_thread_core_type": "physical",
        "worker_core_type": "physical",
        "pin_main_thread": True,
        "bucketwise_worker_affinity": "thread",
        "state_update_backward_cores": 3,
    }
    config = DeepSpeedConfig(minimal_reflow_config(options)).zero_config.reflow
    for key, value in options.items():
        assert getattr(config, key) == value
    config = DeepSpeedConfig(minimal_reflow_config({
        "bucketwise_worker_affinity": "task",
        "pin_main_thread": False
    })).zero_config.reflow
    assert config.bucketwise_worker_affinity == "task"
    assert config.pin_main_thread is False
    for key in ("main_thread_core_type", "worker_core_type", "bucketwise_worker_affinity"):
        with pytest.raises(ValueError):
            DeepSpeedConfig(minimal_reflow_config({key: "unsupported"}))
    with pytest.raises(ValueError):
        DeepSpeedConfig(minimal_reflow_config({"state_update_backward_cores": 0}))


def cpu_siblings(cpu):
    # Use the kernel's sibling list as an oracle independent of the planner's package/core IDs.
    try:
        with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as handle:
            return frozenset(parse_range_list(handle.read().strip()))
    except OSError:
        pytest.skip("requires readable Linux CPU topology")


def available_physical_cpus():
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("requires Linux CPU affinity")
    cpus = sorted(os.sched_getaffinity(0))
    groups = {}
    for cpu in cpus:
        groups.setdefault(cpu_siblings(cpu), []).append(cpu)
    return cpus, list(groups.values())


@pytest.mark.parametrize("main_type", ["logical", "physical"])
@pytest.mark.parametrize("worker_type", ["logical", "physical"])
def test_cpu_split_respects_rank_and_numa_boundaries(main_type, worker_type):
    cpus, groups = available_physical_cpus()
    if len(groups) < 2:
        pytest.skip("requires two physical cores")
    midpoint = len(groups) // 2
    nodes = [[cpu for group in groups[:midpoint] for cpu in group],
             [cpu for group in groups[midpoint:] for cpu in group]]
    main, worker_nodes, workers = reflow_utils.split_cpu_worker_cores(cpus, nodes, 1, main_type, worker_type)
    assert set(main) <= set(cpus)
    assert set(workers) <= set(cpus) - set(main)
    assert workers
    for worker_node in worker_nodes:
        assert any(set(worker_node) <= set(node) for node in nodes)
    if main_type == "physical":
        main_keys = {cpu_siblings(cpu) for cpu in main}
        assert len(main_keys) == 1
        assert all(cpu_siblings(cpu) not in main_keys for cpu in workers)
    if worker_type == "physical":
        assert len(workers) == len({cpu_siblings(cpu) for cpu in workers})


def test_physical_rank_partition_preserves_siblings_and_remainders():
    cpus, groups = available_physical_cpus()
    if len(groups) < 3:
        pytest.skip("requires three physical cores")
    assignments = [reflow_utils.partition_physical_cpus(cpus, 3, rank) for rank in range(3)]
    assert set().union(*map(set, assignments)) == set(cpus)
    physical_assignments = [{cpu_siblings(cpu) for cpu in assignment} for assignment in assignments]
    for rank, physical in enumerate(physical_assignments):
        for other in physical_assignments[rank + 1:]:
            assert physical.isdisjoint(other)
    counts = [len(physical) for physical in physical_assignments]
    assert max(counts) - min(counts) <= 1


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_cpu_split_keeps_a_worker_when_main_reservation_is_too_large(count):
    cpus, _ = available_physical_cpus()
    if len(cpus) < count:
        pytest.skip("requires enough available CPUs")
    rank_cpus = cpus[:count]
    main, _, workers = reflow_utils.split_cpu_worker_cores(rank_cpus, [rank_cpus], 3)
    assert main and workers
    assert len(main) == min(3, max(1, count - 1))
    if count == 1:
        assert main == workers == rank_cpus
    else:
        assert set(main).isdisjoint(workers)
        assert set(main) | set(workers) == set(rank_cpus)


def test_oversubscribed_ranks_share_only_available_cpus():
    cpus, groups = available_physical_cpus()
    if len(groups) < 2:
        pytest.skip("requires two physical cores")
    node = [groups[0][0], groups[1][0]]
    assigned = [reflow_utils.heuristic_rank_cores([node], rank, 4) for rank in range(4)]
    assert assigned == [[node[0]], [node[1]], [node[0]], [node[1]]]
    physical_cpus = groups[0] + groups[1]
    assigned = [reflow_utils.partition_physical_cpus(physical_cpus, 4, rank) for rank in range(4)]
    assert assigned == [groups[0], groups[1], groups[0], groups[1]]


def test_physical_main_reservation_shares_a_single_core():
    _, groups = available_physical_cpus()
    main, _, workers = reflow_utils.split_cpu_worker_cores(groups[0], [groups[0]], 3, "physical", "logical")
    assert set(main) == set(workers) == set(groups[0])


def test_requires_stage3_and_cpu_optimizer_offload():
    # Reflow only extends the stage-3 CPU-offload path; anything else would silently train without it.
    with pytest.raises(ValueError, match="Reflow requires ZeRO stage 3"):
        DeepSpeedConfig(minimal_reflow_config(stage=2))

    with pytest.raises(ValueError, match="Reflow requires optimizer offload"):
        DeepSpeedConfig(minimal_reflow_config(offload_optimizer=None))


@pytest.mark.parametrize("ratio", [0.0, 0.5])
def test_rejects_partial_optimizer_offload(ratio):
    config = minimal_reflow_config(offload_optimizer={"device": "cpu", "ratio": ratio})
    with pytest.raises(ValueError, match="offload_optimizer.ratio must be 1.0"):
        DeepSpeedConfig(config)

    del config["zero_optimization"]["reflow"]
    assert DeepSpeedConfig(config).zero_config.offload_optimizer.ratio == ratio


def test_rejects_deepcompile():
    config = minimal_reflow_config()
    config["compile"] = {"deepcompile": True}
    with pytest.raises(ValueError, match="not supported with DeepCompile"):
        DeepSpeedConfig(config)


def test_rejects_muon_config():
    # Reflow's CPU step would replace Muon's orthogonalized update with Adam, so the combination must fail loudly.
    config = minimal_reflow_config()
    config["optimizer"] = {"type": "Muon", "params": {"lr": 1e-3}}
    with pytest.raises(ValueError, match="does not support the Muon optimizer"):
        DeepSpeedConfig(config)


@pytest.mark.parametrize("precision_config, error", [
    ({
        "bf16": {
            "enabled": False
        }
    }, "requires FP16 or BF16 model parameters"),
    ({
        "torch_autocast": {
            "enabled": True,
            "dtype": "bfloat16"
        }
    }, "requires FP16 or BF16 model parameters"),
    ({
        "bf16": {
            "enabled": True,
            "bf16_master_weights_and_grads": True
        }
    }, "requires FP32 master weights"),
    ({
        "bf16": {
            "enabled": True,
            "bf16_optimizer_states": True
        }
    }, "requires FP32 master weights"),
    ({
        "fp16": {
            "fp16_master_weights_and_grads": True
        }
    }, "requires FP32 master weights"),
])
def test_rejects_unsupported_precision(precision_config, error):
    # These combinations must fail before any worker or native optimizer is created.
    config = minimal_reflow_config()
    config.update(precision_config)
    with pytest.raises(ValueError, match=error):
        DeepSpeedConfig(config)


@pytest.mark.parametrize("model_dtype, grad_dtype", [("bf16", "fp32"), ("bf16", "fp16"), ("fp16", "fp32")])
def test_rejects_mismatched_gradient_accumulation_dtype(model_dtype, grad_dtype):
    config = minimal_reflow_config()
    config["bf16"]["enabled"] = model_dtype == "bf16"
    config["fp16"] = {"enabled": model_dtype == "fp16"}
    config["data_types"] = {"grad_accum_dtype": grad_dtype}
    with pytest.raises(ValueError, match="data_types.grad_accum_dtype to match"):
        DeepSpeedConfig(config)

    config["data_types"]["grad_accum_dtype"] = model_dtype
    assert DeepSpeedConfig(config).grad_accum_dtype == model_dtype

    del config["zero_optimization"]["reflow"]
    config["data_types"]["grad_accum_dtype"] = grad_dtype
    assert DeepSpeedConfig(config).grad_accum_dtype == grad_dtype


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
@pytest.mark.parametrize("device", ["cpu", "nvme"])
def test_accepts_half_precision_and_supported_optimizer_offload(precision, device):
    # FP32 masters/states must not accidentally exclude FP16 compute or NVMe state swapping.
    config = minimal_reflow_config(offload_optimizer={"device": device})
    config["bf16"] = {"enabled": precision == "bf16"}
    config["fp16"] = {"enabled": precision == "fp16"}
    parsed = DeepSpeedConfig(config)
    assert parsed.zero_config.offload_optimizer.device == device
    precision_config = parsed.float16_config if precision == "fp16" else parsed.bfloat16_config
    assert precision_config.enabled


def test_precision_restrictions_do_not_apply_without_reflow():
    config = minimal_reflow_config()
    del config["zero_optimization"]["reflow"]
    config["bf16"] = {"enabled": False}
    assert not DeepSpeedConfig(config).bfloat16_config.enabled
    config["bf16"] = {"enabled": True, "bf16_master_weights_and_grads": True}
    assert DeepSpeedConfig(config).bfloat16_config.bf16_master_weights_and_grads


def test_heuristic_numa_split_with_uneven_rank_count():
    nodes = [list(range(8)), list(range(8, 16))]
    assigned = [set(reflow_utils.heuristic_rank_cores(nodes, rank, 3)) for rank in range(3)]
    assert assigned == [set(range(4)), set(range(8, 16)), set(range(4, 8))]
    assigned = [set(reflow_utils.heuristic_rank_cores([list(range(11))], rank, 3)) for rank in range(3)]
    assert assigned == [set(range(4)), set(range(4, 8)), set(range(8, 11))]


# The oracle runs are bf16 ZeRO-3 on a GPU accelerator; Reflow is not validated on the CPU accelerator.
# Skipping here, rather than inside the test, keeps a CPU-only run from starting the distributed workers.
needs_gpu_accelerator = pytest.mark.skipif(get_accelerator().device_name() == "cpu",
                                           reason="Reflow training tests need a GPU accelerator.")


def skip_if_reflow_training_unsupported(optimizer_kind="adam"):
    builder = ReflowCPULionBuilder if optimizer_kind == "lion" else ReflowCPUAdamBuilder
    if not deepspeed.ops.__compatible_ops__.get(builder.NAME, False):
        pytest.skip(f"{builder.__name__} is not compatible on this system.")
    if not get_accelerator().is_bf16_supported():
        pytest.skip("bf16 is not supported on this accelerator.")


@pytest.mark.parametrize("optimizer_kind", ["adam", "lion"])
def test_training_compatibility_checks_selected_reflow_builder(optimizer_kind, monkeypatch):
    monkeypatch.setitem(deepspeed.ops.__compatible_ops__, ReflowCPUAdamBuilder.NAME, optimizer_kind == "lion")
    monkeypatch.setitem(deepspeed.ops.__compatible_ops__, ReflowCPULionBuilder.NAME, optimizer_kind == "adam")
    with pytest.raises(pytest.skip.Exception):
        skip_if_reflow_training_unsupported(optimizer_kind)


def make_batches(hidden_dim, count, dtype=torch.bfloat16):
    device = get_accelerator().current_device_name()
    batches = []
    for _ in range(count):
        inputs = torch.randn(2, hidden_dim, device=device, dtype=dtype)
        labels = torch.randint(0, hidden_dim, (2, ), device=device)
        batches.append((inputs, labels))
    return batches


def get_offload_clip_config(reflow, gradient_clipping, sub_group_size=None):
    offload_optimizer = {"device": "cpu", "pin_memory": True}
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "bf16": {
            "enabled": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 0.1
            }
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": offload_optimizer,
        },
        "gradient_clipping": gradient_clipping,
    }
    if sub_group_size is not None:
        config["zero_optimization"]["sub_group_size"] = sub_group_size
    if reflow:
        config["zero_optimization"]["reflow"] = {}
    return config


def train_with_offload(model, config, batches):
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    losses = []
    grad_norms = []
    for inputs, labels in batches:
        loss = engine(inputs, labels)
        # Shrink the grads so the 1e-6 term in the clip factor (norm + 1e-6) / clip is not negligible and
        # per-element grads sit near Adam's eps, where the update size depends on the grad scale.
        engine.backward(loss * 1e-7)
        engine.step()
        losses.append(loss.detach().clone())
        grad_norms.append(float(engine.get_global_grad_norm()))
    engine.destroy()
    return losses, grad_norms


def assert_same_training(expected, actual):
    expected_losses, expected_norms = expected
    actual_losses, actual_norms = actual
    assert actual_norms == expected_norms
    for step, (want, got) in enumerate(zip(expected_losses, actual_losses)):
        assert torch.equal(got, want), f"loss differs at step {step}: {got} vs {want}"


@pytest.mark.parametrize("clip_ratio", [0.5, 1.0, 100.0])
@needs_gpu_accelerator
class TestReflowGradClipMatchesZeroOffload(DistributedTest):
    world_size = [1, 2]

    def test_losses_and_grad_norms_match(self, clip_ratio):
        # A clip threshold equal to the grad norm is the boundary case: ZeRO-Offload still clips there
        # because it clips whenever (norm + 1e-6) / clip > 1.
        skip_if_reflow_training_unsupported()

        hidden_dim = 16
        torch.manual_seed(1234)
        initial_model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        batches = make_batches(hidden_dim, count=4)

        # The first step's grad norm does not depend on the clip threshold, so measure it unclipped.
        _, unclipped_norms = train_with_offload(copy.deepcopy(initial_model),
                                                get_offload_clip_config(reflow=False, gradient_clipping=0.0),
                                                batches[:1])
        gradient_clipping = unclipped_norms[0] * clip_ratio

        expected = train_with_offload(copy.deepcopy(initial_model),
                                      get_offload_clip_config(reflow=False, gradient_clipping=gradient_clipping),
                                      batches)
        actual = train_with_offload(copy.deepcopy(initial_model),
                                    get_offload_clip_config(reflow=True, gradient_clipping=gradient_clipping), batches)
        assert_same_training(expected, actual)


@needs_gpu_accelerator
class TestReflowGradNormWithPerParamPartitionGroups(DistributedTest):
    world_size = [1, 2]

    def test_losses_and_grad_norms_match(self):
        # Some params partition over a process group of their own (as AutoEP expert params do), so the grad norm
        # must be reduced per subgroup group. The groups have the same members here, which keeps plain ZeRO-Offload a
        # valid oracle while still exercising the per-group reduction.
        skip_if_reflow_training_unsupported()

        hidden_dim = 16
        torch.manual_seed(1234)
        initial_model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        batches = make_batches(hidden_dim, count=4)

        def model_with_own_partition_group():
            model = copy.deepcopy(initial_model)
            own_group = dist.new_group(ranks=list(range(dist.get_world_size())))
            for param in model.linears[0].parameters():
                param.ds_zero_partition_process_group = own_group
            return model

        # One param per subgroup, so no subgroup mixes partition groups.
        _, unclipped_norms = train_with_offload(
            model_with_own_partition_group(),
            get_offload_clip_config(reflow=False, gradient_clipping=0.0, sub_group_size=1), batches[:1])
        gradient_clipping = unclipped_norms[0] * 0.5

        expected = train_with_offload(
            model_with_own_partition_group(),
            get_offload_clip_config(reflow=False, gradient_clipping=gradient_clipping, sub_group_size=1), batches)
        actual = train_with_offload(
            model_with_own_partition_group(),
            get_offload_clip_config(reflow=True, gradient_clipping=gradient_clipping, sub_group_size=1), batches)
        assert_same_training(expected, actual)


def get_variant_config(reflow, variant):
    offload_optimizer = {"device": "cpu", "pin_memory": True}
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_clipping": 0.0,
        "bf16": {
            "enabled": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-3
            }
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": offload_optimizer,
        },
    }
    if variant == "lion":
        config["optimizer"] = {"type": "Lion", "params": {"lr": 1e-4}}
    elif variant == "offload_param":
        config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}
    elif variant == "client_optimizer":
        # The client passes a DeepSpeedCPUAdam instead, which Reflow remaps to ReflowCPUAdam.
        del config["optimizer"]
    if reflow:
        config["zero_optimization"]["reflow"] = {}
    return config


def train_and_collect(model, config, batches, client_optimizer=None, on_initialize=None, state_keys=()):
    engine, _, _, _ = deepspeed.initialize(model=model,
                                           model_parameters=model.parameters(),
                                           optimizer=client_optimizer,
                                           config=config)
    try:
        if on_initialize is not None:
            on_initialize(engine)
        losses = []
        for inputs, labels in batches:
            loss = engine(inputs, labels)
            engine.backward(loss)
            engine.step()
            losses.append(loss.detach().clone())
        result = {
            "losses": losses,
            "fp32_params": [safe_get_full_fp32_param(param).clone() for param in engine.module.parameters()],
            "skipped_steps": engine.skipped_steps,
            "cpu_optimizer": type(engine.optimizer.optimizer),
        }
        for key in state_keys:
            result[key] = [safe_get_full_optimizer_state(param, key).clone() for param in engine.module.parameters()]
        return result
    finally:
        engine.destroy()


@pytest.mark.parametrize("variant", ["lion", "offload_param", "client_optimizer"])
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
@needs_gpu_accelerator
class TestReflowVariantsMatchZeroOffload(DistributedTest):
    world_size = [1, 2]

    def test_matches_zero_offload(self, variant, precision):
        skip_if_reflow_training_unsupported(variant)
        hidden_dim = 16
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        num_steps = 4
        torch.manual_seed(1234)
        initial_model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        batches = make_batches(hidden_dim, count=num_steps, dtype=dtype)

        def run(reflow):
            model = copy.deepcopy(initial_model)
            client_optimizer = None
            if variant == "client_optimizer":
                client_optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-3)
            config = get_variant_config(reflow, variant)
            if precision == "fp16":
                config["bf16"] = {"enabled": False}
                config["fp16"] = {"enabled": True, "loss_scale": 128}
            return train_and_collect(model, config, batches, client_optimizer)

        expected = run(reflow=False)
        actual = run(reflow=True)

        # Without this check, a run that silently fell back to plain ZeRO-Offload would pass every comparison below.
        expected_cpu_optimizer = ReflowCPULion if variant == "lion" else ReflowCPUAdam
        assert issubclass(actual["cpu_optimizer"], expected_cpu_optimizer)
        assert actual["skipped_steps"] == expected["skipped_steps"]
        for step, (want, got) in enumerate(zip(expected["losses"], actual["losses"])):
            assert torch.equal(got, want), f"loss differs at step {step}: {got} vs {want}"
        for index, (want, got) in enumerate(zip(expected["fp32_params"], actual["fp32_params"])):
            assert torch.equal(got, want), f"fp32 param {index} differs"


@pytest.mark.parametrize("optimizer_kind", ["adam", "lion"])
@needs_gpu_accelerator
class TestReflowSmallLossScaleMatchesZeroOffload(DistributedTest):
    world_size = [1, 2]

    def test_fp16_training_matches_zero_offload(self, optimizer_kind):
        skip_if_reflow_training_unsupported(optimizer_kind)
        torch.manual_seed(1234)
        initial_model = SimpleModel(hidden_dim=16, nlayers=2)
        batches = make_batches(16, count=4, dtype=torch.float16)

        def run(reflow):
            model = copy.deepcopy(initial_model)
            config = get_variant_config(reflow, optimizer_kind)
            config["bf16"] = {"enabled": False}
            config["fp16"] = {"enabled": True, "loss_scale": 0.5}
            engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                           model_parameters=model.parameters(),
                                                           config=config)
            try:
                assert optimizer.loss_scale == 0.5
                losses = []
                for batch in batches:
                    loss = engine(*batch)
                    engine.backward(loss)
                    engine.step()
                    losses.append(loss.detach().clone())
                params = [safe_get_full_fp32_param(param).clone() for param in engine.module.parameters()]
                moments = [
                    safe_get_full_optimizer_state(param, "exp_avg").clone() for param in engine.module.parameters()
                ]
                variances = ([
                    safe_get_full_optimizer_state(param, "exp_avg_sq").clone() for param in engine.module.parameters()
                ] if optimizer_kind == "adam" else [])
                return losses, params, moments, variances, engine.skipped_steps, type(optimizer.optimizer)
            finally:
                engine.destroy()

        expected = run(False)
        actual = run(True)
        assert issubclass(actual[-1], ReflowCPUAdam if optimizer_kind == "adam" else ReflowCPULion)
        assert expected[-2] == actual[-2] == 0
        for expected_group, actual_group in zip(expected[:4], actual[:4]):
            for want, got in zip(expected_group, actual_group):
                assert torch.equal(got, want)


@pytest.mark.parametrize("optimizer_kind", ["adam", "lion"])
@needs_gpu_accelerator
class TestReflowMultipleOptimizerGroupsMatchZeroOffload(DistributedTest):
    world_size = 1

    def test_group_hyperparameters_stay_isolated(self, optimizer_kind):
        # Large groups keep bucket workers active long enough to overlap across optimizer groups.
        skip_if_reflow_training_unsupported(optimizer_kind)

        hidden_dim = 2049
        torch.manual_seed(1234)
        initial_model = SimpleModel(hidden_dim=hidden_dim, nlayers=4)
        batches = make_batches(hidden_dim, count=4)

        def run(reflow):
            model = copy.deepcopy(initial_model)
            first_group = list(model.linears[0].parameters()) + list(model.linears[1].parameters())
            second_group = list(model.linears[2].parameters()) + list(model.linears[3].parameters())
            groups = [{
                "params": first_group,
                "lr": 1e-3,
                "weight_decay": 0.1
            }, {
                "params": second_group,
                "lr": 0.1,
                "weight_decay": 0.0
            }]
            if optimizer_kind == "adam":
                optimizer = DeepSpeedCPUAdam(groups, adamw_mode=True)
            else:
                optimizer = DeepSpeedCPULion(groups)
            config = get_variant_config(reflow, "client_optimizer")
            if reflow:
                config["zero_optimization"]["reflow"]["bucketwise_cores_per_worker"] = 1
            return train_and_collect(model, config, batches, optimizer)

        expected = run(reflow=False)
        actual = run(reflow=True)
        expected_cpu_optimizer = ReflowCPUAdam if optimizer_kind == "adam" else ReflowCPULion
        assert issubclass(actual["cpu_optimizer"], expected_cpu_optimizer)
        for step, (want, got) in enumerate(zip(expected["losses"], actual["losses"])):
            assert torch.equal(got, want), f"loss differs at step {step}: {got} vs {want}"
        for index, (want, got) in enumerate(zip(expected["fp32_params"], actual["fp32_params"])):
            assert torch.equal(got, want), f"fp32 param {index} differs"


@needs_gpu_accelerator
class TestReflowSharedOptimizerWorker(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("optimizer_kind, dtype, clip", [("adam", torch.bfloat16, 0.01),
                                                             ("lion", torch.float16, 0.)])
    def test_training_across_h2d_chunk_boundaries(self, optimizer_kind, dtype, clip):
        skip_if_reflow_training_unsupported(optimizer_kind)
        torch.manual_seed(1234)
        # A single weight is larger than one 1 MiB transfer. Later forwards consume both pieces.
        model = SimpleModel(hidden_dim=768, nlayers=2)
        batches = make_batches(768, count=4, dtype=dtype)

        state_keys = ("exp_avg", "exp_avg_sq") if optimizer_kind == "adam" else ("exp_avg", )

        def run(reflow):
            config = get_variant_config(reflow, optimizer_kind)
            config["gradient_clipping"] = clip
            if dtype == torch.float16:
                config["bf16"]["enabled"] = False
                config["fp16"] = {"enabled": True, "loss_scale": 16}
            if reflow:
                config["zero_optimization"]["reflow"]["h2d_chunk_size_mib"] = 1
            return train_and_collect(copy.deepcopy(model), config, batches, state_keys=state_keys)

        expected, actual = run(False), run(True)
        assert expected["skipped_steps"] == actual["skipped_steps"] == 0
        for key in ("losses", "fp32_params", *state_keys):
            for want, got in zip(expected[key], actual[key]):
                assert torch.equal(want, got)

    def test_failed_ready_restores_cpu_partition(self):
        skip_if_reflow_training_unsupported()
        torch.manual_seed(1234)
        model = torch.nn.Linear(16, 16, bias=False)
        config = get_variant_config(True, "offload_param")
        config["gradient_clipping"] = 0.
        config["optimizer"]["params"]["lr"] = 0.25
        engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        try:
            partition = model.weight.ds_tensor
            expected = partition.clone()
            inputs = torch.randn(2, 16, dtype=torch.bfloat16, device=engine.device)
            native = optimizer.optimizer.ds_opt_adam
            ready = native.reflow_adam_update_params_halfgrad

            def fail_after_write(*args):
                ready(*args)
                raise RuntimeError("injected failure after CPU Ready store")

            with patch.object(native, "reflow_adam_update_params_halfgrad", fail_after_write):
                engine.backward(engine(inputs).sum())
                with pytest.raises(RuntimeError, match="injected failure after CPU Ready store"):
                    engine.step()
            assert torch.equal(partition, expected)
        finally:
            engine.destroy()

    @pytest.mark.parametrize("optimizer_kind,dtype", [("adam", torch.bfloat16), ("lion", torch.float16)])
    @pytest.mark.parametrize("outcome", ["complete", "cancel", "failure"])
    def test_team_publishes_only_finished_ranges(self, optimizer_kind, dtype, outcome):
        # The next Ready can have different scalars and an unrelated destination. Each DMA
        # must see every writer's data and leave gaps and a cancelled successor untouched.
        skip_if_reflow_training_unsupported(optimizer_kind)
        accelerator = get_accelerator()
        if accelerator.device_name() != "cuda":
            pytest.skip("requires CUDA Ready transfers")
        torch.manual_seed(8719)
        model = SimpleModel(hidden_dim=16)
        config = get_variant_config(True, optimizer_kind)
        engine, opt, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        try:
            cpu_opt = opt.optimizer
            native = cpu_opt.ds_opt_adam if optimizer_kind == "adam" else cpu_opt.ds_opt_lion
            update = getattr(native, f"reflow_{optimizer_kind}_update_params_halfgrad")
            stream = accelerator.Stream()
            size, chunk = 262151, 65536
            master = torch.randn(size)
            grad, momentum = torch.randn(size, dtype=dtype), torch.randn(size)
            variance = torch.rand(size)
            jobs, destinations, expected = [], [], []
            for step, start in enumerate((0, 5, 16), 1):
                dst = torch.full((size + 32, ), -77, device=engine.device, dtype=dtype)
                source = torch.empty(size + 32, dtype=dtype, pin_memory=True)[start:start + size]
                reference = torch.empty_like(source, pin_memory=False)
                job = native.ReadyWork(dst, stream, chunk)
                args = [cpu_opt.opt_id, step, 0.1 / step, 0.8, 0.99]
                if optimizer_kind == "adam":
                    args.extend((1e-8, 0.01, True, master, grad, momentum, variance))
                else:
                    args.extend((0.01, master, grad, momentum))
                flags = [0.5 * step, False]
                if optimizer_kind == "adam":
                    flags.append(step == 2)
                update(*args, reference, *flags)
                update(*args, source, *flags, job, start)
                want = torch.full((size + 32, ), -77, dtype=dtype)
                want[start:start + size].copy_(reference)
                jobs.append(job)
                destinations.append(dst)
                expected.append(want)

            stream.wait_stream(accelerator.current_stream())

            def next_job():
                nonlocal admissions
                admissions += 1
                if admissions == 1:
                    return jobs[1]
                if outcome == "failure":
                    raise RuntimeError("injected Ready admission failure")
                if outcome == "cancel":
                    jobs[0].cancel()
                    return jobs[2]
                return None

            admissions = 0
            if outcome == "failure":
                with pytest.raises(RuntimeError, match="injected Ready admission failure"):
                    jobs[0].run(next_job)
            else:
                jobs[0].run(next_job)
            if outcome == "failure":
                assert admissions == 2  # A failed admission must be propagated exactly once.
            stream.synchronize()
            for dst, want, start in zip(destinations[:2], expected[:2], (0, 5)):
                got = dst.cpu()
                if outcome == "complete":
                    assert torch.equal(got, want)
                else:
                    assert (got[:start] == -77).all() and (got[start + size:] == -77).all()
                    for offset in range(start, start + size, chunk):
                        end = min(offset + chunk, start + size)
                        actual = got[offset:end]
                        assert torch.equal(actual, want[offset:end]) or (actual == -77).all()
            assert (destinations[2].cpu() == -77).all()
        finally:
            engine.destroy()

    def test_cancelled_chunk_keeps_prior_h2d_and_range_gaps(self):
        # Cancellation can happen after a prefix has reached the copy stream but before
        # the rest is ready. The returned event must still protect that prefix's source.
        skip_if_reflow_training_unsupported()
        config = get_variant_config(True, "adam")
        config["zero_optimization"]["reflow"]["h2d_chunk_size_mib"] = 1
        model = SimpleModel(hidden_dim=16)
        engine, opt, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        event = None
        try:
            chunk = (1 << 20) // 2
            ranges = [(3, chunk + 11), (chunk + 32, chunk + 7)]
            host = torch.zeros(2 * chunk + 64, dtype=torch.bfloat16, pin_memory=True)
            device = torch.full_like(host, -1, device=engine.device)
            opt._bucketwise_h2d_stream.wait_stream(get_accelerator().current_stream())
            prepared = []

            def prepare(offset, size, source):
                prepared.append((offset, size))
                source.fill_(len(prepared))
                return len(prepared) < 4

            event = opt._copy_ready_ranges(device, host, ranges, prepare=prepare)
            assert event is not None
            assert prepared == [(3, chunk), (3 + chunk, 11), (chunk + 32, chunk), (2 * chunk + 32, 7)]
            expected = torch.full_like(host, -1)
            expected[3:chunk + 3] = 1
            expected[chunk + 3:chunk + 14] = 2
            expected[chunk + 32:2 * chunk + 32] = 3
        finally:
            unpin = get_accelerator().unpin_memory

            def unpin_after_copy(buffer):
                if event is not None:
                    assert event.query(), "teardown released pinned memory before the last Ready copy completed"
                return unpin(buffer)

            with patch.object(get_accelerator(), "unpin_memory", unpin_after_copy):
                engine.destroy()
        assert torch.equal(device.cpu(), expected)

    @pytest.mark.parametrize("invalid", [False, True])
    def test_norm_progresses_while_ready_is_running(self, invalid):
        # A running preview must allow the clipping/overflow decision to proceed, yet its
        # late weight publication must never overwrite the clipped or restored weights.
        skip_if_reflow_training_unsupported()
        torch.manual_seed(1234)
        model = SimpleModel(hidden_dim=16, nlayers=4)
        batches = make_batches(16, count=3)

        def run(reflow):
            config = get_offload_clip_config(reflow, gradient_clipping=0.01, sub_group_size=512)
            config["zero_optimization"]["reduce_bucket_size"] = 128
            net = copy.deepcopy(model)
            engine, opt, _, _ = deepspeed.initialize(model=net, model_parameters=net.parameters(), config=config)
            snapshots = []
            try:
                for index, (inputs, labels) in enumerate(batches):
                    loss = engine(inputs, labels)
                    if index == 1 and invalid:
                        loss = loss * float("inf")
                    if reflow and index == 1:
                        native = opt.optimizer.ds_opt_adam
                        ready = native.reflow_adam_update_params_halfgrad
                        started, norm_finished = threading.Event(), threading.Event()

                        def delayed_ready(*args):
                            started.set()
                            if not norm_finished.wait(10):
                                norm_finished.set()
                                raise RuntimeError("gradient norm waited for a running Ready")
                            return ready(*args)

                        def observe_norm(*args):
                            norm = combine_grad_norm_groups(*args)
                            norm_finished.set()
                            return norm

                        with patch.object(native, "reflow_adam_update_params_halfgrad", delayed_ready), \
                                patch("deepspeed.runtime.reflow.reflow_stage3.combine_grad_norm_groups", observe_norm):
                            engine.backward(loss)
                            assert started.wait(10)
                            engine.step()
                    else:
                        engine.backward(loss)
                        engine.step()
                    for param in net.parameters():
                        snapshots.extend([
                            param.ds_tensor.detach().clone(),
                            safe_get_full_fp32_param(param).clone(),
                            safe_get_full_optimizer_state(param, "exp_avg").clone(),
                            safe_get_full_optimizer_state(param, "exp_avg_sq").clone(),
                        ])
                steps = [int(opt.optimizer.state[p]["step"]) for p in opt.fp32_partitioned_groups_flat]
                return snapshots, steps, engine.skipped_steps
            finally:
                engine.destroy()

        expected, expected_steps, expected_skips = run(False)
        actual, steps, skips = run(True)
        assert steps == expected_steps and skips == expected_skips == int(invalid)
        for want, got in zip(expected, actual):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    @pytest.mark.parametrize("disjoint_teams", [False, True])
    def test_delayed_commit(self, disjoint_teams):
        # Same-core Ready must wait for Persist. A Ready on disjoint CPUs with a
        # committed master must be able to finish while a later subgroup is stalled.
        skip_if_reflow_training_unsupported()
        torch.manual_seed(1234)
        model = SimpleModel(hidden_dim=512, nlayers=3)
        batches = make_batches(512, count=3)
        config = get_variant_config(False, "adam")
        config["zero_optimization"].update(sub_group_size=262144, reduce_bucket_size=262144)
        expected = train_and_collect(copy.deepcopy(model), config, batches)
        cpus, _ = available_physical_cpus()
        if disjoint_teams and len(cpus) < 4:
            pytest.skip("requires a main CPU and three optimizer CPU teams")
        config["zero_optimization"]["reflow"] = {
            "main_thread_cores": 1,
            "bucketwise_cores_per_worker": 1,
            "state_update_cores": 1,
            "state_update_backward_cores": 1,
        }
        last_commit_cpus, conflicting_cpus = set(), set()
        started, disjoint_ready, release = threading.Event(), threading.Event(), threading.Event()
        workers = {"params": set(), "state": set()}
        engine = None
        try:
            os.sched_setaffinity(0, sorted(cpus)[:4 if disjoint_teams else 1])
            engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                           model_parameters=model.parameters(),
                                                           config=config)
            native = optimizer.optimizer.ds_opt_adam
            ready, persist = native.reflow_adam_update_params_halfgrad, native.reflow_adam_update_state_halfgrad
            last_master = optimizer.fp32_partitioned_groups_flat[0].data_ptr()

            def record_ready(*args, **kwargs):
                workers["params"].add(threading.get_native_id())
                if started.is_set() and not release.is_set():
                    mask = os.sched_getaffinity(0)
                    conflicting_cpus.update(mask & last_commit_cpus)
                    if mask.isdisjoint(last_commit_cpus):
                        disjoint_ready.set()
                return ready(*args, **kwargs)

            def delay_last_commit(*args, **kwargs):
                workers["state"].add(threading.get_native_id())
                if args[8].data_ptr() == last_master and not started.is_set():
                    last_commit_cpus.update(os.sched_getaffinity(0))
                    started.set()
                    assert release.wait(30), "next backward did not finish with Persist outstanding"
                return persist(*args, **kwargs)

            with patch.object(native, "reflow_adam_update_params_halfgrad", record_ready), \
                    patch.object(native, "reflow_adam_update_state_halfgrad", delay_last_commit):
                for step, batch in enumerate(batches):
                    if step == 1:
                        assert started.wait(30)
                    loss = engine(*batch)
                    engine.backward(loss)
                    if step == 1:
                        if disjoint_teams:
                            assert disjoint_ready.wait(20), "independent Ready was blocked by the later subgroup"
                        release.set()
                    engine.step()
                    assert torch.equal(loss.detach(), expected["losses"][step])
                for param, want in zip(model.parameters(), expected["fp32_params"]):
                    assert torch.equal(safe_get_full_fp32_param(param), want)
                assert not conflicting_cpus
                if not disjoint_teams:
                    assert workers["params"] == workers["state"]
                    assert len(workers["state"]) == 1
        finally:
            release.set()
            if engine is not None:
                engine.destroy()
            os.sched_setaffinity(0, cpus)


@pytest.mark.parametrize("worker_affinity", ["task", "thread"])
@pytest.mark.parametrize("worker_type", ["logical", "physical"])
@needs_gpu_accelerator
class TestReflowCPUPlacementMatchesZeroOffload(DistributedTest):
    world_size = [1, 2]

    def test_training_and_cpu_placement(self, worker_affinity, worker_type):
        # Catch skipped masks from double round-robin selection, changing masks in thread mode,
        # SMT overlap with the pinned main core, and leaked affinity after engine.destroy().
        skip_if_reflow_training_unsupported()
        original_cpus, groups = available_physical_cpus()
        if len(groups) < 2 * dist.get_world_size():
            pytest.skip("requires a main and a worker physical core per rank")
        torch.manual_seed(1234)
        initial_model = SimpleModel(hidden_dim=1024, nlayers=4)
        batches = make_batches(1024, count=8)
        expected = train_and_collect(copy.deepcopy(initial_model), get_variant_config(False, "adam"), batches)
        config = get_variant_config(True, "adam")
        config["zero_optimization"]["sub_group_size"] = 1_000_000
        config["zero_optimization"]["reflow"] = {
            "main_thread_cores": 1,
            "main_thread_core_type": "physical",
            "worker_core_type": worker_type,
            "bucketwise_cores_per_worker": 4,
            "bucketwise_worker_affinity": worker_affinity,
            "state_update_cores": 2,
            "state_update_backward_cores": 3,
        }
        param_masks = {}
        state_masks = set()
        pinned_cpus = set()
        originals = {}
        native = None

        def observe(engine):
            nonlocal native
            pinned_cpus.update(os.sched_getaffinity(0))
            native = engine.optimizer.optimizer.ds_opt_adam
            for phase in ("params", "state"):
                name = f"reflow_adam_update_{phase}_halfgrad"
                original = getattr(native, name)
                originals[name] = original

                def recorded(*args, _phase=phase, _original=original, **kwargs):
                    mask = frozenset(os.sched_getaffinity(0))
                    if _phase == "params":
                        param_masks.setdefault(threading.get_native_id(), set()).add(mask)
                    else:
                        state_masks.add(mask)
                    return _original(*args, **kwargs)

                setattr(native, name, recorded)

        try:
            actual = train_and_collect(copy.deepcopy(initial_model), config, batches, on_initialize=observe)
        finally:
            if native is not None:
                for name, original in originals.items():
                    setattr(native, name, original)
        assert set(os.sched_getaffinity(0)) == set(original_cpus)
        main_keys = {cpu_siblings(cpu) for cpu in pinned_cpus}
        assert len(main_keys) == 1
        assert pinned_cpus == set().union(*main_keys) & set(original_cpus)
        all_masks = set().union(*param_masks.values())
        used_cpus = set().union(*all_masks)
        assert used_cpus <= set(original_cpus) - pinned_cpus
        for mask in all_masks | state_masks:
            assert {cpu_siblings(cpu) for cpu in mask}.isdisjoint(main_keys)
            if worker_type == "physical":
                assert len(mask) == len({cpu_siblings(cpu) for cpu in mask})
        assert state_masks
        assert all(len(mask) <= 3 for mask in state_masks)
        if worker_affinity == "thread":
            assert all(len(masks) == 1 for masks in param_masks.values())
        placements = [None] * dist.get_world_size()
        dist.all_gather_object(placements, (pinned_cpus, used_cpus))
        all_main = set().union(*(main for main, _ in placements))
        all_used = set().union(*(used for _, used in placements))
        assert {cpu_siblings(cpu) for cpu in all_main}.isdisjoint({cpu_siblings(cpu) for cpu in all_used})
        if worker_affinity == "task":
            remaining = set(original_cpus) - all_main
            if worker_type == "physical":
                remaining = {min(set(group) & remaining) for group in groups if set(group) & remaining}
            assert all_used == remaining
        for want, got in zip(expected["losses"], actual["losses"]):
            assert torch.equal(got, want)
        for want, got in zip(expected["fp32_params"], actual["fp32_params"]):
            assert torch.equal(got, want)


@pytest.mark.parametrize(
    "rank_affinity", ["shared", "launcher", "partial", "insufficient", "small_shared", "one_cpu", "oversubscribed"])
@needs_gpu_accelerator
class TestReflowRankCoreBinding(DistributedTest):
    world_size = 2

    def test_rank_main_and_worker_cpus_are_disjoint(self, rank_affinity):
        # A shared container cpuset must be split, while launcher-assigned rank cpus must be retained.
        skip_if_reflow_training_unsupported()
        if not hasattr(os, "sched_getaffinity"):
            pytest.skip("requires Linux CPU affinity")
        original = set(os.sched_getaffinity(0))
        available = [None] * dist.get_world_size()
        dist.all_gather_object(available, sorted(original))
        shared = sorted(set.intersection(*(set(cpus) for cpus in available)))
        if len(shared) < 16:
            pytest.skip("requires at least 16 CPUs available to both ranks")
        shared = shared[:16]
        rank = dist.get_rank()
        rank_cpus = set(shared[rank * 8:(rank + 1) * 8])
        if rank_affinity == "shared":
            input_cpus = set(shared)
        elif rank_affinity == "partial":
            input_cpus = set(shared[:12] if rank == 0 else shared[4:])
        elif rank_affinity == "insufficient":
            input_cpus = set(shared[rank * 3:(rank + 1) * 3])
            rank_cpus = input_cpus
        elif rank_affinity == "small_shared":
            input_cpus = set(shared[:4])
            rank_cpus = set(shared[rank * 2:(rank + 1) * 2])
        elif rank_affinity == "one_cpu":
            input_cpus = {shared[rank]}
            rank_cpus = input_cpus
        elif rank_affinity == "oversubscribed":
            input_cpus = {shared[0]}
            rank_cpus = input_cpus
        else:
            input_cpus = rank_cpus
        engine = None
        native = None
        originals = {}
        worker_masks = []
        state_masks = []
        try:
            os.sched_setaffinity(0, input_cpus)
            torch.manual_seed(1234)
            model = SimpleModel(hidden_dim=16, nlayers=2)
            config = get_variant_config(reflow=True, variant="adam")
            config["zero_optimization"]["reduce_bucket_size"] = 1_000_000
            if rank_affinity == "partial":
                with pytest.raises(ValueError, match="partially overlapping rank CPU affinities"):
                    deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
                return
            engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                           model_parameters=model.parameters(),
                                                           config=config)
            main_cpus = set(os.sched_getaffinity(0))
            native = optimizer.optimizer.ds_opt_adam
            for phase in ("params", "state"):
                name = f"reflow_adam_update_{phase}_halfgrad"
                originals[name] = getattr(native, name)

                def recorded(*args, _phase=phase, _original=originals[name], **kwargs):
                    mask = set(os.sched_getaffinity(0))
                    if _phase == "params":
                        worker_masks.append(mask)
                    else:
                        state_masks.append(mask)
                    return _original(*args, **kwargs)

                setattr(native, name, recorded)
            batches = make_batches(16, count=2)
            for inputs, labels in batches:
                loss = engine(inputs, labels)
                engine.backward(loss)
                engine.step()
            optimizer.state_dict()
            expected_main = set(sorted(rank_cpus)[:min(3, max(1, len(rank_cpus) - 1))])
            expected_workers = rank_cpus - expected_main or rank_cpus
            assert main_cpus == expected_main
            assert worker_masks
            assert all(mask == expected_workers for mask in worker_masks)
            assert state_masks
            assert all(mask == expected_workers for mask in state_masks)
            placements = [None] * dist.get_world_size()
            dist.all_gather_object(placements, (main_cpus, rank_cpus))
            if rank_affinity == "oversubscribed":
                assert placements[0][1] == placements[1][1]
            else:
                assert placements[0][1].isdisjoint(placements[1][1])
            assert all(main <= cpus for main, cpus in placements)
        finally:
            if native is not None:
                for name, original_fn in originals.items():
                    setattr(native, name, original_fn)
            if engine is not None:
                engine.destroy()
            restored_cpus = set(os.sched_getaffinity(0))
            os.sched_setaffinity(0, original)
            if engine is not None:
                assert restored_cpus == input_cpus


class TestReflowOptimizerRequiresOptIn(DistributedTest):
    world_size = 1
    non_daemonic_procs = True

    def test_regular_optimizer_without_reflow_config(self):
        model = SimpleModel(hidden_dim=16, nlayers=2)
        optimizer = torch.optim.Adam(model.parameters())
        config = {"train_micro_batch_size_per_gpu": 1, "zero_optimization": {"stage": 0}}
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
        try:
            assert engine.basic_optimizer is optimizer
        finally:
            engine.destroy()

    @pytest.mark.parametrize("optimizer_type, builder", [(ReflowCPUAdam, ReflowCPUAdamBuilder),
                                                         (ReflowCPULion, ReflowCPULionBuilder)])
    def test_rejects_reflow_optimizer_without_reflow_config(self, optimizer_type, builder):
        if not deepspeed.ops.__compatible_ops__.get(builder.NAME, False):
            pytest.skip("the CPU optimizer builder is not supported on this accelerator")
        if builder().simd_width() not in ("-D__AVX512__", "-D__AVX256__"):
            pytest.skip("the Reflow CPU optimizer requires AVX2 or AVX-512")
        model = SimpleModel(hidden_dim=16, nlayers=2)
        optimizer = optimizer_type(model.parameters())
        config = {"train_micro_batch_size_per_gpu": 1, "zero_optimization": {"stage": 0}}
        with pytest.raises(ZeRORuntimeException, match="require zero_optimization.reflow"):
            deepspeed.initialize(model=model, optimizer=optimizer, config=config)


class TestReflowPrecisionValidation(DistributedTest):
    world_size = 1
    non_daemonic_procs = True

    @pytest.mark.parametrize("optimizer_type", [DeepSpeedCPUAdam, DeepSpeedCPULion])
    def test_rejects_low_precision_client_states(self, optimizer_type):
        builder = CPULionBuilder if optimizer_type is DeepSpeedCPULion else CPUAdamBuilder
        if not deepspeed.ops.__compatible_ops__.get(builder.NAME, False):
            pytest.skip("the CPU optimizer builder is not supported on this accelerator")
        model = SimpleModel(hidden_dim=16, nlayers=2)
        optimizer = optimizer_type(model.parameters(), fp32_optimizer_states=False)
        config = get_variant_config(reflow=True, variant="client_optimizer")
        with pytest.raises(ZeRORuntimeException, match="requires FP32 optimizer states"):
            deepspeed.initialize(model=model, optimizer=optimizer, config=config)

    @pytest.mark.parametrize("precision", ["bf16", "fp16"])
    def test_initializes_half_precision_zero_init_model(self, precision):
        if precision == "fp16" and not get_accelerator().is_fp16_supported():
            pytest.skip("the accelerator does not support FP16")
        if not deepspeed.ops.__compatible_ops__.get(ReflowCPUAdamBuilder.NAME, False):
            pytest.skip("the ReFlow CPU Adam builder is not supported on this accelerator")
        config = get_variant_config(reflow=True, variant="offload_param")
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        if precision == "fp16":
            config["bf16"] = {"enabled": False}
            config["fp16"] = {"enabled": True, "loss_scale": 128}
        with deepspeed.zero.Init(config_dict_or_path=config, dtype=dtype):
            model = SimpleModel(hidden_dim=16, nlayers=2)
        engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        try:
            assert all(param.dtype == dtype for param in engine.module.parameters())
            assert isinstance(optimizer.optimizer, ReflowCPUAdam)
            assert optimizer.optimizer.fp32_optimizer_states
        finally:
            engine.destroy()

    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("fp16_supported", [False, True])
    def test_rejects_zero_init_model_with_unsupported_dtype(self, dtype, fp16_supported):
        # zero.Init bypasses the engine's model cast, so its actual parameters must still be half precision.
        config = get_variant_config(reflow=True, variant="offload_param")
        with patch.object(get_accelerator(), "is_fp16_supported", return_value=fp16_supported):
            with deepspeed.zero.Init(config_dict_or_path=config, dtype=dtype):
                model = SimpleModel(hidden_dim=16, nlayers=2)
        assert all(param.dtype == dtype for param in model.parameters())
        with pytest.raises(ValueError, match="requires FP16 or BF16 model parameters"):
            deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)


@needs_gpu_accelerator
class TestReflowUnsupportedGradientAccess(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("precision", ["bf16", "fp16"])
    def test_rejects_mixed_fp32_and_half_parameters(self, precision):
        skip_if_reflow_training_unsupported()
        model = SimpleModel(hidden_dim=16, nlayers=2)
        config = get_variant_config(True, "adam")
        if precision == "fp16":
            config["bf16"] = {"enabled": False}
            config["fp16"] = {"enabled": True, "loss_scale": 128}
        config["data_types"] = {"keep_in_fp32_modules": ["linears.0"]}
        with pytest.raises(ValueError, match="requires FP16 or BF16 model parameters"):
            deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)


@needs_gpu_accelerator
class TestReflowLocalGradientAccess(DistributedTest):
    world_size = [1, 2]
    non_daemonic_procs = True

    @pytest.mark.parametrize("precision", ["bf16", "fp16"])
    @pytest.mark.parametrize("optimizer_kind", ["adam", "lion"])
    @pytest.mark.parametrize("accumulation_steps", [1, 2])
    @pytest.mark.parametrize("clip", [0.0, 0.1])
    def test_local_gradients_and_updates_match_zero(self, precision, optimizer_kind, accumulation_steps, clip):
        skip_if_reflow_training_unsupported(optimizer_kind)
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        torch.manual_seed(1234)
        initial_model = SimpleModel(hidden_dim=16, nlayers=2)
        batches = make_batches(16, count=2 * accumulation_steps, dtype=dtype)

        def run(reflow):
            model = copy.deepcopy(initial_model)
            config = get_variant_config(reflow, optimizer_kind)
            config["data_types"] = {"grad_accum_dtype": precision}
            config["gradient_accumulation_steps"] = accumulation_steps
            config["gradient_clipping"] = clip
            config["zero_optimization"]["reduce_bucket_size"] = 128
            if precision == "fp16":
                config["bf16"] = {"enabled": False}
                config["fp16"] = {"enabled": True, "loss_scale": 128}
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            snapshots = []
            try:
                for step in range(2):
                    for micro in range(accumulation_steps):
                        inputs, labels = batches[step * accumulation_steps + micro]
                        engine.backward(engine(inputs, labels))
                        if micro == accumulation_steps - 1:
                            param = next(engine.module.parameters())
                            gradient = safe_get_local_grad(param).clone()
                            safe_set_local_grad(param, (-gradient * 0.5).to(dtype))
                            updated_gradient = safe_get_local_grad(param).clone()
                        engine.step()
                    params = [safe_get_full_fp32_param(p).clone() for p in engine.module.parameters()]
                    states = [safe_get_full_optimizer_state(p, "exp_avg").clone() for p in engine.module.parameters()]
                    variances = (
                        [safe_get_full_optimizer_state(p, "exp_avg_sq").clone()
                         for p in engine.module.parameters()] if optimizer_kind == "adam" else [])
                    steps = [int(state["step"]) for state in engine.optimizer.optimizer.state.values()]
                    snapshots.append(
                        (gradient, updated_gradient, params, states, variances, steps, engine.get_global_grad_norm()))
                return snapshots
            finally:
                engine.destroy()

        expected = run(False)
        actual = run(True)
        for want, got in zip(expected, actual):
            assert torch.equal(want[0], got[0])
            assert torch.equal(want[1], got[1])
            assert want[5:] == got[5:]
            for expected_group, actual_group in zip(want[2:5], got[2:5]):
                for expected_tensor, actual_tensor in zip(expected_group, actual_group):
                    assert torch.equal(expected_tensor, actual_tensor)

    @pytest.mark.parametrize("precision", ["bf16", "fp16"])
    def test_modification_between_microsteps_accumulates(self, precision):
        # The reference zeros this parameter's first gradient in autograd, before ZeRO sees it.
        skip_if_reflow_training_unsupported()
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        torch.manual_seed(1234)
        initial_model = SimpleModel(hidden_dim=16, nlayers=2)
        batches = make_batches(16, count=2, dtype=dtype)

        def run(reflow):
            model = copy.deepcopy(initial_model)
            phase = [0]
            if not reflow:
                model.linears[0].weight.register_hook(lambda grad: torch.zeros_like(grad) if phase[0] == 0 else grad)
            config = get_variant_config(reflow, "adam")
            config["data_types"] = {"grad_accum_dtype": precision}
            config["gradient_accumulation_steps"] = 2
            if precision == "fp16":
                config["bf16"] = {"enabled": False}
                config["fp16"] = {"enabled": True, "loss_scale": 128}
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            try:
                for micro_step, (inputs, labels) in enumerate(batches):
                    phase[0] = micro_step
                    engine.backward(engine(inputs, labels))
                    if reflow and micro_step == 0:
                        param = engine.module.linears[0].weight
                        local_grad = safe_get_local_grad(param)
                        safe_set_local_grad(param, torch.zeros_like(local_grad, dtype=dtype))
                        assert torch.count_nonzero(safe_get_local_grad(param)) == 0
                    engine.step()
                params = [safe_get_full_fp32_param(p).clone() for p in engine.module.parameters()]
                states = [safe_get_full_optimizer_state(p, "exp_avg").clone() for p in engine.module.parameters()]
                return params, states
            finally:
                engine.destroy()

        expected = run(False)
        actual = run(True)
        for expected_group, actual_group in zip(expected, actual):
            for want, got in zip(expected_group, actual_group):
                assert torch.equal(want, got)


@needs_gpu_accelerator
class TestReflowFullGradientUpdate(DistributedTest):
    world_size = [1, 2]
    non_daemonic_procs = True

    @pytest.mark.parametrize("optimizer_kind,precision,clip", [("adam", "bf16", 0.0), ("lion", "fp16", 0.1)])
    def test_edits_drive_ready_persist_and_overflow(self, optimizer_kind, precision, clip):
        skip_if_reflow_training_unsupported(optimizer_kind)
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        torch.manual_seed(8719)
        initial_model = SimpleModel(hidden_dim=16, nlayers=2)
        batches = make_batches(16, count=8, dtype=dtype)

        def run(reflow):
            model = copy.deepcopy(initial_model)
            config = get_variant_config(reflow, optimizer_kind)
            config["gradient_accumulation_steps"] = 2
            config["gradient_clipping"] = clip
            config["zero_optimization"]["reduce_bucket_size"] = 128
            if precision == "fp16":
                config["bf16"] = {"enabled": False}
                config["fp16"] = {"enabled": True, "loss_scale": 128}
            engine, opt, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            params = list(engine.module.parameters())
            results = []
            try:
                for step in range(4):
                    for micro in range(2):
                        inputs, labels = batches[2 * step + micro]
                        loss = engine(inputs, labels)
                        engine.backward(loss)
                        if micro == 1:

                            def edit(grad, param):
                                assert grad.shape == param.ds_shape and grad.dtype == torch.float32
                                if step in (1, 2) and param is params[0]:
                                    return grad.fill_(float("inf"))
                                # Reverse across rank boundaries and return values that need rounding.
                                value = grad.flip((0, )).add_(0.0137)
                                return value if reflow else value.to(dtype)

                            safe_update_full_grad_vectorized(params, edit)
                            if step == 1:
                                safe_update_full_grad_vectorized([params[0]], lambda g, p: torch.zeros_like(g))
                            results.extend(safe_get_local_grad(p).clone() for p in params)
                        engine.step()
                    results.extend([loss.detach(), engine.get_global_grad_norm(), engine.skipped_steps])
                    results.extend(int(opt.optimizer.state[p]["step"]) for p in opt.fp32_partitioned_groups_flat)
                    for param in params:
                        results.append(safe_get_full_fp32_param(param).clone())
                        results.append(safe_get_full_optimizer_state(param, "exp_avg").clone())
                        if optimizer_kind == "adam":
                            results.append(safe_get_full_optimizer_state(param, "exp_avg_sq").clone())
                assert engine.skipped_steps == 1
                return results
            finally:
                engine.destroy()

        torch.testing.assert_close(run(True), run(False), rtol=0, atol=0)

    def test_padded_and_unused_gradients_between_microsteps(self):
        skip_if_reflow_training_unsupported()

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(9, dtype=torch.float32).reshape(3, 3))
                self.bias = torch.nn.Parameter(torch.ones(1))

            def forward(self, inputs, bias_scale):
                loss = (self.weight * inputs).sum()
                return loss if bias_scale is None else loss + self.bias.sum() * bias_scale

        def run(reflow):
            model = Model()
            config = get_variant_config(reflow, "adam")
            config["gradient_accumulation_steps"] = 2
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            params = list(engine.module.parameters())
            try:
                inputs = torch.arange(9, device=engine.device, dtype=torch.bfloat16).reshape(3, 3)
                for step in range(2):
                    for micro in range(2):
                        editing = step == 1 and micro == 0
                        # The ZeRO reference obtains the edited gradients directly from its loss.
                        data = inputs.t() if editing and not reflow else inputs
                        bias_scale = 4 if step == 0 else (1 if editing and not reflow else None)
                        engine.backward(engine(data, bias_scale))
                        if editing and reflow:

                            def edit(grad, param):
                                if param is engine.module.bias:
                                    assert grad.shape == (1, ) and grad.item() == 0
                                    return grad.fill_(0.5)
                                torch.testing.assert_close(grad, inputs.float() / 2, rtol=0, atol=0)
                                return grad.t().cpu()

                            safe_update_full_grad_vectorized(params, edit)
                        engine.step()
                return ([safe_get_full_fp32_param(p).clone()
                         for p in params], [safe_get_full_optimizer_state(p, "exp_avg").clone()
                                            for p in params], engine.get_global_grad_norm())
            finally:
                engine.destroy()

        torch.testing.assert_close(run(True), run(False), rtol=0, atol=0)

    @pytest.mark.world_size(2)
    def test_update_on_owning_partition_group(self):
        # Only rank zero edits its expert shard. A world collective in the API would hang its peer.
        skip_if_reflow_training_unsupported()
        rank = dist.get_rank()
        singleton_groups = [dist.new_group(ranks=[r]) for r in range(dist.get_world_size())]
        torch.manual_seed(8719)
        initial_model = SimpleModel(hidden_dim=16, nlayers=2)
        inputs, labels = make_batches(16, 1)[0]

        def run(reflow):
            model = copy.deepcopy(initial_model)
            weight = model.linears[0].weight
            weight.ds_zero_partition_process_group = singleton_groups[rank]
            config = get_variant_config(reflow, "adam")
            config["zero_optimization"]["sub_group_size"] = 1
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            try:
                engine.backward(engine(inputs, labels))
                if rank == 0:
                    safe_update_full_grad_vectorized([weight], lambda g, p: g.flip((0, )))
                engine.step()
                master = safe_get_full_fp32_param(weight).clone()
                momentum = safe_get_full_optimizer_state(weight, "exp_avg").clone()
                return master, momentum
            finally:
                engine.destroy()

        torch.testing.assert_close(run(True), run(False), rtol=0, atol=0)

    def test_rank_local_callback_failure_can_retry_without_backward(self):
        skip_if_reflow_training_unsupported()
        torch.manual_seed(8719)
        initial_model = SimpleModel(hidden_dim=16, nlayers=2)
        inputs, labels = make_batches(16, 1)[0]

        def run(reflow):
            model = copy.deepcopy(initial_model)
            config = get_variant_config(reflow, "adam")
            config["zero_optimization"]["reduce_bucket_size"] = 128
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            params = list(engine.module.parameters())
            try:
                engine.backward(engine(inputs, labels))
                if reflow:

                    def edit(grad, param):
                        if param is params[1] and dist.get_rank() == 0:
                            raise ValueError("invalid gradient edit")
                        return grad.neg_()

                    with pytest.raises((ValueError, RuntimeError)):
                        safe_update_full_grad_vectorized(params, edit)
                    safe_update_full_grad_vectorized(params[1:], lambda g, p: g.neg_())
                else:
                    safe_update_full_grad_vectorized(params, lambda g, p: g.neg_())
                engine.step()
                return ([safe_get_full_fp32_param(p).clone()
                         for p in params], [safe_get_full_optimizer_state(p, "exp_avg").clone() for p in params])
            finally:
                engine.destroy()

        torch.testing.assert_close(run(True), run(False), rtol=0, atol=0)


@needs_gpu_accelerator
class TestReflowLocalGradientInvalid(DistributedTest):
    world_size = [1, 2]
    non_daemonic_procs = True

    @pytest.mark.parametrize("precision", ["bf16", "fp16"])
    @pytest.mark.parametrize("restore", [False, True])
    def test_replaced_invalid_gradient_matches_zero(self, precision, restore):
        skip_if_reflow_training_unsupported()
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        torch.manual_seed(1234)
        initial_model = SimpleModel(hidden_dim=16, nlayers=2)
        inputs, labels = make_batches(16, count=1, dtype=dtype)[0]

        def run(reflow):
            model = copy.deepcopy(initial_model)
            config = get_variant_config(reflow, "adam")
            config["data_types"] = {"grad_accum_dtype": precision}
            if precision == "fp16":
                config["bf16"] = {"enabled": False}
                config["fp16"] = {"enabled": True, "loss_scale": 128}
            engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                           model_parameters=model.parameters(),
                                                           config=config)
            try:
                engine.backward(engine(inputs, labels))
                engine.step()
                engine.backward(engine(inputs, labels))
                param = next(engine.module.parameters())
                if dist.get_rank() == 0:
                    original = safe_get_local_grad(param).to(dtype)
                    safe_set_local_grad(param, torch.full_like(original, float("inf")))
                    if restore:
                        safe_set_local_grad(param, original)
                engine.step()
                params = [safe_get_full_fp32_param(p).clone() for p in engine.module.parameters()]
                states = [safe_get_full_optimizer_state(p, "exp_avg").clone() for p in engine.module.parameters()]
                return engine.skipped_steps, params, states
            finally:
                engine.destroy()

        expected = run(False)
        actual = run(True)
        assert actual[0] == expected[0] == (0 if restore else 1)
        for expected_group, actual_group in zip(expected[1:], actual[1:]):
            for want, got in zip(expected_group, actual_group):
                assert torch.equal(want, got)


def test_reflow_releases_every_parent_gradient_dtype():
    optimizer = object.__new__(ReflowOptimizer_Stage3)
    optimizer.offload_optimizer_pin_memory = True
    references = []
    released = []

    def create_parent_buffers(parent):
        buffers = {dtype: torch.zeros(4, dtype=dtype) for dtype in (torch.float16, torch.float32)}
        references.extend(weakref.ref(buffer) for buffer in buffers.values())
        parent.grad_partitions_flat_buffers = buffers
        parent.grad_partitions_flat_buffer = buffers[torch.float16]
        parent._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition = {
            i: buffer.narrow(0, 0, 2)
            for i, buffer in enumerate(buffers.values())
        }

    accelerator = SimpleNamespace(unpin_memory=lambda buffer: released.append(id(buffer)))
    with patch.object(DeepSpeedZeroOptimizer_Stage3, "_setup_for_real_optimizer", create_parent_buffers), \
            patch("deepspeed.runtime.reflow.reflow_stage3.get_accelerator", return_value=accelerator):
        optimizer._setup_for_real_optimizer()
    assert len(released) == len(set(released)) == 2
    assert all(reference() is None for reference in references)
    assert not optimizer.grad_partitions_flat_buffers
    assert optimizer.grad_partitions_flat_buffer is None
    assert not optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition


@pytest.mark.parametrize("nvme_parameter", [False, True])
def test_reflow_overflow_keeps_nvme_restore_paths(nvme_parameter):
    # NVMe parameters must be re-swapped from the unchanged master; swapped optimizer
    # subgroups never published a preview and must not attempt a master copy.
    optimizer = SimpleNamespace(fp16_partitioned_groups_flat=[None if nvme_parameter else torch.zeros(4)],
                                fp32_partitioned_groups_flat=[torch.ones(4)],
                                _partitioned_params_swap_out=Mock(),
                                _swappable_optimizer_subgroup=Mock(return_value=not nvme_parameter),
                                _copy_pinned_master_to_device=Mock(),
                                _unflatten_partitioned_parameters=Mock())
    ReflowOptimizer_Stage3._restore_fp16_params_from_master(optimizer, 0)
    if nvme_parameter:
        optimizer._partitioned_params_swap_out.assert_called_once_with(0)
    else:
        optimizer._partitioned_params_swap_out.assert_not_called()
    optimizer._copy_pinned_master_to_device.assert_not_called()
    optimizer._unflatten_partitioned_parameters.assert_not_called()


@needs_gpu_accelerator
class TestReflowReleasesParentGradientBuffers(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("pin_memory", [False, True])
    def test_releases_all_parent_gradient_buffers(self, pin_memory):
        skip_if_reflow_training_unsupported()
        original_setup = DeepSpeedZeroOptimizer_Stage3._setup_for_real_optimizer
        accelerator = get_accelerator()
        original_unpin = accelerator.unpin_memory
        parent_buffers = []
        unpinned = []

        def capture_parent_buffers(optimizer):
            original_setup(optimizer)
            parent_buffers.extend(weakref.ref(buffer) for buffer in optimizer.grad_partitions_flat_buffers.values())

        def record_unpin(buffer):
            if any(reference() is buffer for reference in parent_buffers):
                unpinned.append(id(buffer))
            return original_unpin(buffer)

        config = get_variant_config(reflow=True, variant="adam")
        config["zero_optimization"]["offload_optimizer"]["pin_memory"] = pin_memory
        model = SimpleModel(hidden_dim=16, nlayers=2)
        with patch.object(DeepSpeedZeroOptimizer_Stage3, "_setup_for_real_optimizer", capture_parent_buffers), \
                patch.object(accelerator, "unpin_memory", record_unpin):
            engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                           model_parameters=model.parameters(),
                                                           config=config)
            try:
                assert parent_buffers
                assert all(reference() is None for reference in parent_buffers)
                assert not optimizer.grad_partitions_flat_buffers
                assert optimizer.grad_partitions_flat_buffer is None
            finally:
                engine.destroy()
        assert len(unpinned) == (len(parent_buffers) if pin_memory else 0)
        assert len(set(unpinned)) == len(unpinned)


@needs_gpu_accelerator
class TestReflowRejectsUnsupportedClientOptimizer(DistributedTest):
    world_size = 1

    def test_raises_instead_of_training_with_adam(self):
        # MuonWithAuxAdam's class name contains "Adam", so it used to be rebuilt as ReflowCPUAdam and silently train
        # every parameter with Adam.
        skip_if_reflow_training_unsupported()
        model = SimpleModel(hidden_dim=16, nlayers=2)
        muon_params = [param for param in model.parameters() if param.ndim >= 2]
        adam_params = [param for param in model.parameters() if param.ndim < 2]
        for param in muon_params:
            param.use_muon = True
        for param in adam_params:
            param.use_muon = False
        client_optimizer = MuonWithAuxAdam([
            dict(params=muon_params, use_muon=True, lr=1e-3),
            dict(params=adam_params, use_muon=False, lr=1e-3),
        ])
        config = get_variant_config(reflow=True, variant="client_optimizer")
        # Without this, ZeRO-Offload rejects every non-DeepSpeed CPU optimizer before Reflow sees it.
        config["zero_force_ds_cpu_optimizer"] = False
        with pytest.raises(ZeRORuntimeException, match="does not support the Muon optimizer"):
            deepspeed.initialize(model=model, optimizer=client_optimizer, config=config)

    def test_rejects_radam_instead_of_training_with_adam(self):
        skip_if_reflow_training_unsupported()
        model = SimpleModel(hidden_dim=16, nlayers=2)
        client_optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)
        config = get_variant_config(reflow=True, variant="client_optimizer")
        config["zero_force_ds_cpu_optimizer"] = False
        with pytest.raises(ZeRORuntimeException, match="cannot remap the client optimizer RAdam"):
            deepspeed.initialize(model=model, optimizer=client_optimizer, config=config)

    def test_rejects_scheduler_bound_to_replaced_optimizer(self):
        skip_if_reflow_training_unsupported()
        model = SimpleModel(hidden_dim=16, nlayers=2)
        client_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(client_optimizer, step_size=1)
        config = get_variant_config(reflow=True, variant="client_optimizer")
        config["zero_force_ds_cpu_optimizer"] = False
        with pytest.raises(ZeRORuntimeException, match="already bound to an LR scheduler"):
            deepspeed.initialize(model=model, optimizer=client_optimizer, lr_scheduler=scheduler, config=config)


@pytest.mark.parametrize("failure_source", ["accumulate", "enqueue"])
@needs_gpu_accelerator
class TestReflowAsyncWorkerFailures(DistributedTest):
    world_size = [1, 2]
    non_daemonic_procs = True

    def test_step_reports_worker_error(self, failure_source):
        # A failed CPU accumulation or enqueue must abort the public step on every rank.
        skip_if_reflow_training_unsupported()
        message = f"injected {failure_source} failure"

        def fail(*args):
            raise RuntimeError(message)

        if failure_source == "accumulate":
            failure_patch = patch.object(ReflowCPUAdam, "bf16_accumulate", classmethod(fail))
        else:
            failure_patch = patch.object(reflow_utils, "clip_range_to_valid_partition", fail)

        model = SimpleModel(hidden_dim=16, nlayers=2)
        config = get_offload_clip_config(reflow=True, gradient_clipping=0.0)
        # Three micro-steps exercise the CPU accumulation worker between the first and boundary steps.
        config["gradient_accumulation_steps"] = 3
        batches = make_batches(16, count=3)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        try:
            with failure_patch:
                for step, (inputs, labels) in enumerate(batches):
                    loss = engine(inputs, labels)
                    engine.backward(loss)
                    if step < 2:
                        engine.step()
                    else:
                        with pytest.raises(RuntimeError, match=message):
                            engine.step()
        finally:
            engine.destroy()


@pytest.mark.parametrize("worker_fails, read_before_save", [(True, False), (True, True), (False, True)])
class TestReflowCheckpointWorkerFailures(DistributedTest):
    world_size = [1, 2] if get_accelerator().device_count() > 1 else 1
    non_daemonic_procs = True

    def test_checkpoint_reports_worker_failure(self, worker_fails, read_before_save, tmpdir):
        skip_if_reflow_training_unsupported()
        model = SimpleModel(hidden_dim=16, nlayers=2)
        config = get_offload_clip_config(reflow=True, gradient_clipping=0.0)
        engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        save_dir = tmpdir / f"checkpoint_{dist.get_world_size()}"
        message = "injected asynchronous state commit failure"
        failing_rank = worker_fails and dist.get_rank() == 0

        def commit():
            if failing_rank:
                raise RuntimeError(message)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Inject the real Future at the async commit boundary to reproduce the checkpoint bug;
                # CPU-only execution cannot run Reflow's GPU backward/copy path.
                optimizer._last_state_update_future = executor.submit(commit)
                read_error = None
                if read_before_save and dist.get_rank() == 0:
                    try:
                        optimizer.state_dict()
                    except RuntimeError as error:
                        read_error = error

                if worker_fails:
                    with pytest.raises(RuntimeError) as save_error:
                        engine.save_checkpoint(str(save_dir), tag="test")
                    if failing_rank:
                        assert str(save_error.value) == message
                    assert not save_dir.exists()
                    if read_before_save and failing_rank:
                        assert str(read_error) == message
                    if failing_rank:
                        # A consumed future must not hide its failure from a later serialization attempt.
                        for read in (optimizer.state_dict, optimizer.get_lean_optimizer_state):
                            with pytest.raises(RuntimeError, match=message):
                                read()
                else:
                    assert read_error is None
                    assert engine.save_checkpoint(str(save_dir), tag="test")
                    assert (save_dir / "latest").read_text(encoding="utf-8") == "test"
        finally:
            engine.destroy()


class TestReflowFailedCommitStateAccess(DistributedTest):
    world_size = [1, 2]
    non_daemonic_procs = True

    def test_parameter_and_state_access_report_failed_commit(self):
        skip_if_reflow_training_unsupported()
        model = SimpleModel(hidden_dim=16, nlayers=2)
        engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                       model_parameters=model.parameters(),
                                                       config=get_variant_config(True, "adam"))
        try:
            if dist.get_rank() == 0:
                failed_commit = Future()
                failed_commit.set_exception(RuntimeError("injected asynchronous state commit failure"))
                optimizer._last_state_update_future = failed_commit
            param = next(engine.module.parameters())
            for read in (lambda: safe_get_full_fp32_param(param),
                         lambda: safe_get_full_optimizer_state(param, "exp_avg")):
                with pytest.raises(RuntimeError, match="asynchronous"):
                    read()
        finally:
            engine.destroy()

    @needs_gpu_accelerator
    def test_failed_native_commit_aborts_next_step(self):
        # Fail partway through a shared Persist batch. Readers on every rank, including
        # readers of a later subgroup, must be released and the next step must abort.
        skip_if_reflow_training_unsupported()
        torch.manual_seed(1234)
        model = SimpleModel(hidden_dim=768, nlayers=3)
        config = get_offload_clip_config(True, gradient_clipping=0., sub_group_size=262144)
        config["zero_optimization"]["reduce_bucket_size"] = 262144
        engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        try:
            native = optimizer.optimizer.ds_opt_adam
            persist = native.reflow_adam_update_state_halfgrad
            target = optimizer.fp32_partitioned_groups_flat[1].data_ptr()

            def commit(*args, **kwargs):
                if dist.get_rank() == 0 and args[8].data_ptr() == target:
                    raise RuntimeError("injected subgroup commit failure")
                return persist(*args, **kwargs)

            first, second = make_batches(768, count=2)
            with patch.object(native, "reflow_adam_update_state_halfgrad", commit):
                engine.backward(engine(*first))
                engine.step()
                engine.backward(engine(*second))
                with pytest.raises(RuntimeError,
                                   match="injected subgroup commit failure|worker failed on another rank"):
                    engine.step()
        finally:
            engine.destroy()


@needs_gpu_accelerator
class TestReflowCheckpointResume(DistributedTest):
    world_size = [1, 2]
    non_daemonic_procs = True

    def test_resume_matches_uninterrupted_training(self, tmpdir):
        # A checkpoint immediately after step must include the outstanding Persist update.
        skip_if_reflow_training_unsupported()
        torch.manual_seed(1234)
        model = SimpleModel(hidden_dim=16, nlayers=2)
        batches = make_batches(16, count=3)
        config = get_variant_config(True, "adam")
        config["gradient_clipping"] = 0.1
        save_dir = str(tmpdir / "resume")
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)

        def continue_training(current_engine):
            snapshots = []
            for inputs, labels in batches[1:]:
                loss = current_engine(inputs, labels)
                current_engine.backward(loss)
                current_engine.step()
                params = [safe_get_full_fp32_param(param).clone() for param in current_engine.module.parameters()]
                moments = [
                    safe_get_full_optimizer_state(param, "exp_avg").clone()
                    for param in current_engine.module.parameters()
                ]
                snapshots.append((loss.detach().clone(), params, moments))
            return snapshots

        try:
            inputs, labels = batches[0]
            engine.backward(engine(inputs, labels))
            engine.step()
            assert engine.save_checkpoint(save_dir, tag="first")
            expected = continue_training(engine)
        finally:
            engine.destroy()

        model = SimpleModel(hidden_dim=16, nlayers=2)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        try:
            loaded_path, _ = engine.load_checkpoint(save_dir, tag="first")
            assert loaded_path is not None
            actual = continue_training(engine)
        finally:
            engine.destroy()

        for expected_step, actual_step in zip(expected, actual):
            assert torch.equal(expected_step[0], actual_step[0])
            for expected_group, actual_group in zip(expected_step[1:], actual_step[1:]):
                for want, got in zip(expected_group, actual_group):
                    assert torch.equal(want, got)


def get_state_read_config(reflow, gradient_accumulation_steps, warmup_lr):
    offload_optimizer = {"device": "cpu", "pin_memory": True}
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 0.0,
        "bf16": {
            "enabled": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-3
            }
        },
        "zero_optimization": {
            "stage": 3,
            # Split every weight into its own subgroup, so the chained async commits keep running after step()
            # returns and overlap the next micro-steps and the LR scheduler step.
            "sub_group_size": 2_000_000,
            "offload_optimizer": offload_optimizer,
        },
    }
    if warmup_lr:
        # The LR changes after every optimizer step, so a commit that reads it late would use the next step's LR.
        config["scheduler"] = {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0.0,
                "warmup_max_lr": 1e-3,
                "warmup_num_steps": 10
            }
        }
    if reflow:
        config["zero_optimization"]["reflow"] = {}
    return config


def read_state_right_after_each_step(model, config, batches):
    # Read as soon as an optimizer step returns, which is when a training script would save a checkpoint.
    engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    snapshots = []
    for inputs, labels in batches:
        loss = engine(inputs, labels)
        engine.backward(loss)
        at_boundary = engine.is_gradient_accumulation_boundary()
        engine.step()
        if not at_boundary:
            continue
        flat_groups = [group.clone() for group in optimizer.state_dict()[FP32_FLAT_GROUPS]]
        fp32_params = [safe_get_full_fp32_param(lp).clone() for lp in engine.module.parameters()]
        exp_avgs = [safe_get_full_optimizer_state(lp, "exp_avg").clone() for lp in engine.module.parameters()]
        snapshots.append((flat_groups, fp32_params, exp_avgs))
    engine.destroy()
    return snapshots


@pytest.mark.parametrize("gradient_accumulation_steps, warmup_lr", [(1, False), (2, False), (3, False), (4, False),
                                                                    (8, False), (1, True)])
@needs_gpu_accelerator
class TestReflowStateReadAfterStep(DistributedTest):
    world_size = [1, 2]

    def test_matches_zero_offload(self, gradient_accumulation_steps, warmup_lr):
        # Reflow commits the FP32 master and optimizer state in background workers after step() returns, while the
        # next micro-steps run. Checkpoints and safe_get_* reads right after step() must see the committed values, and
        # the commit must not be disturbed by the next micro-steps' grads or by an LR scheduler step. The model is
        # large enough that the commit outlasts the read.
        skip_if_reflow_training_unsupported()

        hidden_dim = 2048
        torch.manual_seed(1234)
        batches = make_batches(hidden_dim, count=2 * gradient_accumulation_steps)

        expected = read_state_right_after_each_step(
            SimpleModel(hidden_dim=hidden_dim, nlayers=4),
            get_state_read_config(reflow=False,
                                  gradient_accumulation_steps=gradient_accumulation_steps,
                                  warmup_lr=warmup_lr), batches)
        torch.manual_seed(1234)
        actual = read_state_right_after_each_step(
            SimpleModel(hidden_dim=hidden_dim, nlayers=4),
            get_state_read_config(reflow=True,
                                  gradient_accumulation_steps=gradient_accumulation_steps,
                                  warmup_lr=warmup_lr), batches)

        for step, (expected_step, actual_step) in enumerate(zip(expected, actual)):
            for name, expected_tensors, actual_tensors in zip(("state_dict fp32 groups", "fp32 params", "exp_avg"),
                                                              expected_step, actual_step):
                for index, (want, got) in enumerate(zip(expected_tensors, actual_tensors)):
                    assert torch.equal(got, want), f"{name}[{index}] differs after step {step}"


@pytest.mark.parametrize("gradient_accumulation_steps", [4, 8])
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
@needs_gpu_accelerator
class TestReflowAccumulationBufferReuse(DistributedTest):
    world_size = 1

    def test_matches_zero_offload(self, gradient_accumulation_steps, precision):
        # Repeated updates with many parameters in one subgroup catch scratch buffers reused before CPU reads finish.
        skip_if_reflow_training_unsupported()
        hidden_dim = 128
        torch.manual_seed(1234)
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        batches = make_batches(hidden_dim, count=4 * gradient_accumulation_steps, dtype=dtype)

        def config_for(reflow):
            config = get_state_read_config(reflow=reflow,
                                           gradient_accumulation_steps=gradient_accumulation_steps,
                                           warmup_lr=False)
            config["zero_optimization"]["reduce_bucket_size"] = 1_000_000
            if precision == "fp16":
                config["bf16"] = {"enabled": False}
                config["fp16"] = {"enabled": True, "loss_scale": 128}
            return config

        expected = read_state_right_after_each_step(SimpleModel(hidden_dim=hidden_dim, nlayers=8), config_for(False),
                                                    batches)
        torch.manual_seed(1234)
        actual = read_state_right_after_each_step(SimpleModel(hidden_dim=hidden_dim, nlayers=8), config_for(True),
                                                  batches)
        for expected_step, actual_step in zip(expected, actual):
            for expected_tensors, actual_tensors in zip(expected_step, actual_step):
                for want, got in zip(expected_tensors, actual_tensors):
                    assert torch.equal(got, want)


class TestReflowAccumulationBufferReuseDistributed(TestReflowAccumulationBufferReuse):
    world_size = 2


class ConditionalGradientModel(torch.nn.Module):

    def __init__(self, width=16):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.always = torch.nn.Linear(width, width)
        self.optional = torch.nn.Linear(width, width)

    def forward(self, inputs, use_optional):
        result = self.always(inputs)
        if use_optional:
            result = result + self.optional(inputs)
        return self.scale.float() * result.float().square().mean()


@needs_gpu_accelerator
class TestReflowConditionalGradients(DistributedTest):
    world_size = [1, 2]

    @pytest.mark.parametrize("optimizer_kind", ["adam", "lion"])
    @pytest.mark.parametrize("accumulation_steps", [2, 4])
    def test_matches_zero_offload_across_accumulation_windows(self, optimizer_kind, accumulation_steps):
        skip_if_reflow_training_unsupported(optimizer_kind)
        if accumulation_steps == 2:
            schedule = [(True, True), (True, False), (False, False), (False, True), (True, True)]
        else:
            schedule = [(True, True, True, False), (False, False, False, False), (False, False, True, True),
                        (True, True, True, True)]

        def run(reflow):
            torch.manual_seed(1234)
            model = ConditionalGradientModel()
            config = get_variant_config(reflow, optimizer_kind)
            config["data_types"] = {"grad_accum_dtype": "bf16"}
            config["gradient_accumulation_steps"] = accumulation_steps
            config["zero_optimization"]["reduce_bucket_size"] = 1000
            config["zero_optimization"]["sub_group_size"] = 1
            engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                           model_parameters=model.parameters(),
                                                           config=config)
            snapshots = []
            try:
                for step, uses in enumerate(schedule):
                    for micro_step, use_optional in enumerate(uses):
                        torch.manual_seed(900 + step * accumulation_steps + micro_step)
                        inputs = torch.randn(2, 16, device=engine.device, dtype=torch.bfloat16)
                        engine.backward(engine(inputs, use_optional))
                        engine.step()
                    params = [safe_get_full_fp32_param(p).clone() for p in engine.module.parameters()]
                    states = [safe_get_full_optimizer_state(p, "exp_avg").clone() for p in engine.module.parameters()]
                    steps = [int(optimizer.optimizer.state[p]["step"]) for p in optimizer.fp32_partitioned_groups_flat]
                    snapshots.append((params, states, steps))
                return snapshots
            finally:
                engine.destroy()

        expected = run(False)
        actual = run(True)
        for step, (reference, result) in enumerate(zip(expected, actual)):
            assert reference[2] == result[2], (f"optimizer steps differ at step {step} rank {dist.get_rank()}: "
                                               f"expected {reference[2]}, got {result[2]}")
            for expected_group, actual_group in zip(reference[:2], result[:2]):
                for want, got in zip(expected_group, actual_group):
                    assert torch.equal(got, want), f"parameter or optimizer state differs at step {step}"

    def test_norm_after_replacing_an_earlier_microstep_gradient(self):
        skip_if_reflow_training_unsupported()

        def run(reflow):
            torch.manual_seed(1234)
            model = ConditionalGradientModel()
            weight = model.optional.weight
            # Zero the same first contribution in the reference before ZeRO partitions it.
            if not reflow:
                weight.register_hook(lambda grad: torch.zeros_like(grad) if micro_step == 0 else grad)
            config = get_variant_config(reflow, "adam")
            config["data_types"] = {"grad_accum_dtype": "bf16"}
            config["gradient_accumulation_steps"] = 3
            config["gradient_clipping"] = 0.01
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            try:
                for micro_step, use_optional in enumerate((True, True, False)):
                    torch.manual_seed(900 + micro_step)
                    inputs = torch.randn(2, 16, device=engine.device, dtype=torch.bfloat16)
                    engine.backward(engine(inputs, use_optional))
                    if reflow and micro_step == 0:
                        safe_set_local_grad(weight, torch.zeros_like(safe_get_local_grad(weight)))
                    engine.step()
                return engine.get_global_grad_norm(), safe_get_full_optimizer_state(weight, "exp_avg").clone()
            finally:
                engine.destroy()

        expected_norm, expected_moment = run(False)
        actual_norm, actual_moment = run(True)
        assert actual_norm == expected_norm
        assert torch.equal(actual_moment, expected_moment)

    def test_skipped_step_does_not_reuse_invalid_gradient(self):
        skip_if_reflow_training_unsupported()
        schedule = [(True, False), (True, True), (False, False), (False, True)]

        def run(reflow):
            torch.manual_seed(1234)
            model = ConditionalGradientModel()
            config = get_variant_config(reflow, "adam")
            config["bf16"] = {"enabled": False}
            config["fp16"] = {"enabled": True, "loss_scale": 128}
            config["data_types"] = {"grad_accum_dtype": "fp16"}
            config["gradient_accumulation_steps"] = 2
            config["zero_optimization"]["sub_group_size"] = 256
            engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                           model_parameters=model.parameters(),
                                                           config=config)
            snapshots = []
            try:
                for step, uses in enumerate(schedule):
                    for micro_step, use_optional in enumerate(uses):
                        torch.manual_seed(900 + step * 2 + micro_step)
                        inputs = torch.randn(2, 16, device=engine.device, dtype=torch.float16)
                        loss = engine(inputs, use_optional)
                        if step == 1 and micro_step == 0 and dist.get_rank() == 0:
                            loss = loss * float("inf")
                        engine.backward(loss)
                        engine.step()
                    params = [safe_get_full_fp32_param(p).clone() for p in engine.module.parameters()]
                    states = [safe_get_full_optimizer_state(p, "exp_avg").clone() for p in engine.module.parameters()]
                    steps = [int(optimizer.optimizer.state[p]["step"]) for p in optimizer.fp32_partitioned_groups_flat]
                    with deepspeed.zero.GatheredParameters(list(engine.module.parameters())):
                        low_params = [p.detach().clone() for p in engine.module.parameters()]
                    snapshots.append((params, states, steps, engine.skipped_steps, low_params))
                return snapshots
            finally:
                engine.destroy()

        for step, (expected, actual) in enumerate(zip(run(False), run(True))):
            assert expected[2:4] == actual[2:4], f"step counters differ at step {step}"
            for kind, (expected_group,
                       actual_group) in enumerate(zip((expected[4], *expected[:2]), (actual[4], *actual[:2]))):
                for index, (want, got) in enumerate(zip(expected_group, actual_group)):
                    if not torch.equal(got, want):
                        max_index = (got - want).abs().argmax().item()
                        raise AssertionError(
                            f"kind {kind} index {index} differs at step {step} rank {dist.get_rank()}: "
                            f"low={want.flatten()[max_index].item()},{got.flatten()[max_index].item()} "
                            f"master={expected[0][index].flatten()[max_index].item()},"
                            f"{actual[0][index].flatten()[max_index].item()}")

    def test_zero_grad_discards_pending_invalid_gradient(self):
        skip_if_reflow_training_unsupported()

        def run(reflow):
            torch.manual_seed(1234)
            model = ConditionalGradientModel()
            config = get_variant_config(reflow, "adam")
            config["bf16"] = {"enabled": False}
            config["fp16"] = {"enabled": True, "loss_scale": 128}
            config["data_types"] = {"grad_accum_dtype": "fp16"}
            config["zero_optimization"]["reduce_bucket_size"] = 256
            engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                           model_parameters=model.parameters(),
                                                           config=config)
            try:
                torch.manual_seed(900)
                inputs = torch.randn(2, 16, device=engine.device, dtype=torch.float16)
                engine.backward(engine(inputs, True) * float("inf"))
                optimizer.zero_grad()
                engine.backward(engine(inputs, False))
                engine.step()
                params = [safe_get_full_fp32_param(p).clone() for p in engine.module.parameters()]
                states = [safe_get_full_optimizer_state(p, "exp_avg").clone() for p in engine.module.parameters()]
                return params, states, engine.skipped_steps
            finally:
                engine.destroy()

        expected, actual = run(False), run(True)
        assert expected[2] == actual[2] == 0
        for expected_group, actual_group in zip(expected[:2], actual[:2]):
            for want, got in zip(expected_group, actual_group):
                assert torch.equal(got, want)


class ReentrantGradientModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(16, 16)
        self.shared = torch.nn.Linear(16, 16)
        self.output = torch.nn.Linear(16, 1)

    def forward(self, inputs):
        hidden = self.input(inputs).tanh()
        for _ in range(2):
            hidden = checkpoint(self.shared, hidden, use_reentrant=True).tanh()
        return self.output(hidden).float().square().mean()


@needs_gpu_accelerator
class TestReflowReentrantGradients(DistributedTest):
    world_size = [1, 2]

    @pytest.mark.parametrize("optimizer_kind", ["adam", "lion"])
    @pytest.mark.parametrize("gradient_accumulation_steps, clip, precision, offload_params",
                             [(1, 0., "bf16", False), (1, 1000., "bf16", False), (2, 0.1, "bf16", False),
                              (1, 0.0001, "fp16", False), (1, 0., "fp16", False),
                              pytest.param(2, 0.1, "bf16", True, id="cpu-partitions-bf16"),
                              pytest.param(1, 0., "fp16", True, id="cpu-partitions-fp16")])
    def test_matches_zero_offload(self, optimizer_kind, gradient_accumulation_steps, clip, precision, offload_params):
        skip_if_reflow_training_unsupported(optimizer_kind)

        def run(reflow):
            torch.manual_seed(1234)
            model = ReentrantGradientModel()
            config = get_variant_config(reflow, optimizer_kind)
            if offload_params:
                config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}
            config["data_types"] = {"grad_accum_dtype": precision}
            dtype = torch.bfloat16
            if precision == "fp16":
                dtype = torch.float16
                config["bf16"] = {"enabled": False}
                config["fp16"] = {"enabled": True, "initial_scale_power": 3, "loss_scale_window": 1, "hysteresis": 1}
            config["gradient_accumulation_steps"] = gradient_accumulation_steps
            config["gradient_clipping"] = clip
            config["zero_optimization"]["reduce_bucket_size"] = 256
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            snapshots = []
            try:
                for step in range(3):
                    for micro_step in range(gradient_accumulation_steps):
                        torch.manual_seed(900 + step * gradient_accumulation_steps + micro_step)
                        inputs = torch.randn(2, 16, device=engine.device, dtype=dtype)
                        loss = engine(inputs)
                        if precision == "fp16" and step == 1 and dist.get_rank() == 0:
                            loss = loss * float("inf")
                        engine.backward(loss)
                        engine.step()
                    with deepspeed.zero.GatheredParameters(list(engine.module.parameters())):
                        published = [p.detach().clone() for p in engine.module.parameters()]
                    params = [safe_get_full_fp32_param(p).clone() for p in engine.module.parameters()]
                    states = [safe_get_full_optimizer_state(p, "exp_avg").clone() for p in engine.module.parameters()]
                    snapshots.append((published, params, states, engine.get_global_grad_norm(),
                                      engine.optimizer.loss_scale, engine.skipped_steps))
                return snapshots
            finally:
                engine.destroy()

        for step, (expected, actual) in enumerate(zip(run(False), run(True))):
            assert actual[3:] == expected[3:], f"gradient norm, loss scale or skipped steps differ at step {step}"
            for expected_group, actual_group in zip(expected[:3], actual[:3]):
                for want, got in zip(expected_group, actual_group):
                    assert torch.equal(got, want), (f"parameter or optimizer state differs at step {step}: "
                                                    f"max diff {(got - want).abs().max().item()}")

    @pytest.mark.parametrize("precision", ["bf16", "fp16"])
    def test_unclipped_checkpoint_ready_preserves_forward_weights(self, precision):
        # Ready may finish in backward, but recomputation still owns the published weights.
        skip_if_reflow_training_unsupported()

        class CheckpointedLayers(torch.nn.Sequential):

            def forward(self, inputs):
                inputs = checkpoint(self[0], inputs, use_reentrant=True).tanh()
                inputs = checkpoint(self[1], inputs, use_reentrant=True).tanh()
                return checkpoint(self[0], inputs, use_reentrant=True).float().square().mean()

        def run(reflow):
            torch.manual_seed(731)
            model = CheckpointedLayers(torch.nn.Linear(16, 16, bias=False), torch.nn.Linear(16, 16, bias=False))
            config = get_offload_clip_config(reflow, gradient_clipping=0., sub_group_size=1)
            dtype = torch.bfloat16
            if precision == "fp16":
                dtype = torch.float16
                config["bf16"] = {"enabled": False}
                config["fp16"] = {"enabled": True, "loss_scale": 0.5}
            config["zero_optimization"]["reduce_bucket_size"] = 32
            engine, opt, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            snapshots = []
            try:
                for step in range(3):
                    inputs = torch.randn(2, 16, dtype=dtype, device=engine.device, requires_grad=True)
                    loss = engine(inputs)
                    if step == 1 and dist.get_rank() == 0:
                        loss = loss * float("inf")
                    with ExitStack() as patches:
                        if reflow:
                            weights = [param.ds_tensor.clone() for param in model.parameters()]
                            native = opt.optimizer.ds_opt_adam
                            ready = native.reflow_adam_update_params_halfgrad
                            ready_finished = threading.Event()
                            processed = []

                            def observe_ready(*args):
                                result = ready(*args)
                                processed.append(args[8].numel())
                                ready_finished.set()
                                return result

                            patches.enter_context(
                                patch.object(native, "reflow_adam_update_params_halfgrad", observe_ready))
                        engine.backward(loss)
                        if reflow:
                            assert ready_finished.wait(10), "checkpoint Ready was held until step()"
                        if step == 2:
                            # Replace the gradient of the weight that already has a staged Ready.
                            weight = model[1].weight
                            safe_set_local_grad(weight, torch.full_like(safe_get_local_grad(weight), 0.125))
                        if reflow:
                            # The local setter joins outstanding previews. A premature copy
                            # is now visible regardless of worker ordering.
                            get_accelerator().synchronize()
                            for param, original in zip(model.parameters(), weights):
                                assert torch.equal(param.ds_tensor, original)
                        engine.step()
                        if reflow and step == 0:
                            # The tied weight receives two gradients but needs only one Ready.
                            assert sum(processed) == sum(weight.numel() for weight in weights)
                    for param in model.parameters():
                        snapshots.extend([
                            param.ds_tensor.clone(),
                            safe_get_full_fp32_param(param).clone(),
                            safe_get_full_optimizer_state(param, "exp_avg").clone(),
                            safe_get_full_optimizer_state(param, "exp_avg_sq").clone()
                        ])
                steps = [int(opt.optimizer.state[p]["step"]) for p in opt.fp32_partitioned_groups_flat]
                return snapshots, steps, engine.skipped_steps
            finally:
                engine.destroy()

        expected, expected_steps, expected_skips = run(False)
        actual, steps, skips = run(True)
        assert steps == expected_steps and skips == expected_skips == 1
        for want, got in zip(expected, actual):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    @pytest.mark.parametrize("dynamic, failure", [(False, "kernel"), (True, "kernel"), (True, "executor")])
    def test_deferred_ready_failure_aborts_before_persist(self, dynamic, failure):
        # A rank-local error published after the norm decision must abort all ranks
        # before any state, master weight or optimizer step counter is committed.
        skip_if_reflow_training_unsupported()
        torch.manual_seed(1234)
        model = ReentrantGradientModel()
        config = get_offload_clip_config(True, gradient_clipping=0.0001)
        if dynamic:
            config["bf16"] = {"enabled": False}
            config["fp16"] = {"enabled": True, "initial_scale_power": 3, "loss_scale_window": 1}
        # Keep repeated uses in separate reduction buckets, as in the ZeRO-3 oracle above.
        config["zero_optimization"]["reduce_bucket_size"] = 256
        engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        try:
            dtype = torch.float16 if dynamic else torch.bfloat16
            inputs = torch.randn(2, 16, device=engine.device, dtype=dtype)
            engine.backward(engine(inputs))
            engine.step()
            optimizer.state_dict()
            masters = optimizer.fp32_partitioned_groups_flat
            snapshots = [(p.detach().clone(), copy.deepcopy(optimizer.optimizer.state[p])) for p in masters]
            engine.backward(engine(inputs))
            native = optimizer.optimizer.ds_opt_adam
            ready = native.reflow_adam_update_params_halfgrad

            scale_before = optimizer.loss_scale
            if dynamic:
                iteration_before = optimizer.loss_scaler.cur_iter

            def fail_on_one_rank(*args, **kwargs):
                if dist.get_rank() == 0 and float(args[13]) > optimizer.loss_scale:
                    raise RuntimeError("injected final Ready failure")
                return ready(*args, **kwargs)

            if failure == "executor" and dist.get_rank() == 0:
                # An executor can fail a Future before the Ready kernel gets a chance to report it.
                failed = Future()
                failed.set_exception(RuntimeError("injected final Ready failure"))
                failure_patch = patch.object(optimizer._bucketwise_update_executor, "submit", return_value=failed)
            else:
                failure_patch = patch.object(native, "reflow_adam_update_params_halfgrad", fail_on_one_rank)
            with failure_patch:
                with pytest.raises(RuntimeError, match="injected final Ready failure|worker failed on another rank"):
                    engine.step()
            assert optimizer.loss_scale == scale_before
            if dynamic:
                assert optimizer.loss_scaler.cur_iter == iteration_before
            for param, (master, state) in zip(masters, snapshots):
                assert torch.equal(param, master)
                now = optimizer.optimizer.state[param]
                assert now["step"] == state["step"]
                for key in ("exp_avg", "exp_avg_sq"):
                    assert torch.equal(now[key], state[key])
        finally:
            engine.destroy()


@needs_gpu_accelerator
class TestReflowNVMeConditionalGradients(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("optimizer_kind", ["adam", "lion"])
    @pytest.mark.parametrize("nvme_target", ["offload_optimizer", "offload_param"])
    def test_matches_zero_offload(self, tmpdir, optimizer_kind, nvme_target):
        skip_if_reflow_training_unsupported(optimizer_kind)
        if not deepspeed.ops.__compatible_ops__.get(AsyncIOBuilder.NAME, False):
            pytest.skip("NVMe offload requires the async_io extension")
        # Parameter I/O requires at least 1 MiB per BF16 weight. Optimizer I/O uses the FP32 subgroup.
        width = 768 if nvme_target == "offload_param" else 384

        def run(reflow):
            torch.manual_seed(1234)
            model = ConditionalGradientModel(width)
            config = get_variant_config(reflow, optimizer_kind)
            config["data_types"] = {"grad_accum_dtype": "bf16"}
            config["gradient_accumulation_steps"] = 2
            config["zero_optimization"]["reduce_bucket_size"] = 1_000_000
            config["zero_optimization"][nvme_target] = {
                "device": "nvme",
                "nvme_path": str(tmpdir.join("reflow" if reflow else "zero"))
            }
            if nvme_target == "offload_param":
                # One large weight per subgroup keeps the pinned swap pools small for both runs.
                zero_config = config["zero_optimization"]
                zero_config["sub_group_size"] = width * width
                zero_config[nvme_target].update(max_in_cpu=0, buffer_size=width * width, buffer_count=2)
                zero_config["offload_optimizer"]["pin_memory"] = False
            engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                           model_parameters=model.parameters(),
                                                           config=config)
            snapshots = []
            try:
                if nvme_target == "offload_param":
                    assert any(p.ds_tensor.final_location == "nvme" for p in engine.module.parameters())
                else:
                    assert any(optimizer._swappable_optimizer_subgroup(i) for i in range(len(optimizer.fp16_groups)))
                for step, uses in enumerate(((True, True), (True, False), (False, False), (False, True))):
                    for micro_step, use_optional in enumerate(uses):
                        torch.manual_seed(900 + step * 2 + micro_step)
                        inputs = torch.randn(2, width, device=engine.device, dtype=torch.bfloat16)
                        engine.backward(engine(inputs, use_optional))
                        engine.step()
                    params = [safe_get_full_fp32_param(p).clone() for p in engine.module.parameters()]
                    states = [safe_get_full_optimizer_state(p, "exp_avg").clone() for p in engine.module.parameters()]
                    # Reading through the public gather also checks the weights Ready published to the swap file.
                    weights = []
                    for param in engine.module.parameters():
                        with deepspeed.zero.GatheredParameters([param]):
                            weights.append(param.detach().cpu().clone())
                    snapshots.append((params, states, weights))
                return snapshots
            finally:
                engine.destroy()

        for step, (expected, actual) in enumerate(zip(run(False), run(True))):
            for expected_group, actual_group in zip(expected, actual):
                for want, got in zip(expected_group, actual_group):
                    assert torch.equal(got, want), f"NVMe parameter or optimizer state differs at step {step}"

    @pytest.mark.parametrize("optimizer_kind", ["adam", "lion"])
    @pytest.mark.parametrize("vectorized", [False, True], ids=["local", "full"])
    def test_local_gradient_access_matches_zero(self, tmpdir, optimizer_kind, vectorized):
        skip_if_reflow_training_unsupported(optimizer_kind)
        if not deepspeed.ops.__compatible_ops__.get(AsyncIOBuilder.NAME, False):
            pytest.skip("NVMe offload requires the async_io extension")
        width = 384

        def run(reflow):
            torch.manual_seed(1234)
            model = ConditionalGradientModel(width)
            config = get_variant_config(reflow, optimizer_kind)
            config["data_types"] = {"grad_accum_dtype": "bf16"}
            config["gradient_clipping"] = 0.1
            config["zero_optimization"]["offload_optimizer"] = {
                "device": "nvme",
                "nvme_path": str(tmpdir.join("local_reflow" if reflow else "local_zero"))
            }
            engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                           model_parameters=model.parameters(),
                                                           config=config)
            try:
                param = engine.module.always.weight
                group = optimizer.grad_position[optimizer.get_param_id(param)][0]
                assert optimizer._swappable_optimizer_subgroup(group)
                torch.manual_seed(900)
                inputs = torch.randn(2, width, device=engine.device, dtype=torch.bfloat16)
                engine.backward(engine(inputs, True))
                gradient = safe_get_local_grad(param).clone()
                if vectorized:
                    safe_update_full_grad_vectorized([param], lambda g, p: (-0.5 * g).to(torch.bfloat16))
                else:
                    safe_set_local_grad(param, (-0.5 * gradient).to(torch.bfloat16))
                engine.step()
                params = [safe_get_full_fp32_param(p).clone() for p in engine.module.parameters()]
                states = [safe_get_full_optimizer_state(p, "exp_avg").clone() for p in engine.module.parameters()]
                return gradient, params, states
            finally:
                engine.destroy()

        expected = run(False)
        actual = run(True)
        assert torch.equal(expected[0], actual[0])
        for expected_group, actual_group in zip(expected[1:], actual[1:]):
            for want, got in zip(expected_group, actual_group):
                assert torch.equal(want, got)


class TestReflowMaximizeClientInitialization(DistributedTest):
    world_size = 1
    non_daemonic_procs = True

    @pytest.mark.parametrize('optimizer_type', [torch.optim.Adam, torch.optim.AdamW])
    def test_preserves_group_directions(self, optimizer_type):
        skip_if_reflow_training_unsupported()
        model = SimpleModel(hidden_dim=16, nlayers=2)
        params = list(model.parameters())
        optimizer = optimizer_type([{'params': params[:2]}, {'params': params[2:], 'maximize': False}], maximize=True)
        config = get_variant_config(reflow=True, variant='client_optimizer')
        config['zero_force_ds_cpu_optimizer'] = False
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
        try:
            actual = engine.optimizer.optimizer
            assert isinstance(actual, ReflowCPUAdam)
            assert [group['maximize'] for group in actual.param_groups] == [True, False]
        finally:
            engine.destroy()


@needs_gpu_accelerator
class TestReflowMaximizeTraining(DistributedTest):
    world_size = [1, 2]

    @pytest.mark.parametrize('adamw_mode', [False, True])
    def test_matches_minimizing_negative_loss(self, adamw_mode):
        # Negating the loss supplies an independent ascent oracle using the existing descent optimizer.
        skip_if_reflow_training_unsupported()
        torch.manual_seed(1234)
        initial = SimpleModel(hidden_dim=16, nlayers=2)
        batches = make_batches(16, count=5)
        results = []
        for reflow in (False, True):
            model = copy.deepcopy(initial)
            config = get_variant_config(reflow, 'client_optimizer')
            config['zero_force_ds_cpu_optimizer'] = False
            if reflow:
                optimizer_type = torch.optim.AdamW if adamw_mode else torch.optim.Adam
                optimizer = optimizer_type(model.parameters(), lr=0.01, weight_decay=0.1, maximize=True)
            else:
                optimizer = DeepSpeedCPUAdam(model.parameters(), lr=0.01, weight_decay=0.1, adamw_mode=adamw_mode)
            engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
            try:
                losses = []
                for inputs, labels in batches:
                    loss = engine(inputs, labels)
                    engine.backward(loss if reflow else -loss)
                    engine.step()
                    losses.append(loss.detach().clone())
                results.append((losses, [safe_get_full_fp32_param(p).clone() for p in engine.module.parameters()]))
            finally:
                engine.destroy()
        for expected, actual in zip(results[0][0], results[1][0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for expected, actual in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("offload_param", [False, True])
@needs_gpu_accelerator
class TestReflowInvalidGradient(DistributedTest):
    world_size = [1, 2]

    def test_invalid_step_preserves_parameters_and_state(self, precision, invalid_value, pin_memory, offload_param):
        skip_if_reflow_training_unsupported()
        torch.manual_seed(1234)
        config = get_variant_config(reflow=True, variant="offload_param" if offload_param else "adam")
        config["zero_optimization"]["offload_optimizer"]["pin_memory"] = pin_memory
        # Submit Ready during backward so overflow must restore an actual parameter preview.
        config["zero_optimization"]["reduce_bucket_size"] = 128
        if precision == "fp16":
            config["bf16"] = {"enabled": False}
            config["fp16"] = {"enabled": True, "loss_scale": 128}
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        batches = make_batches(16, count=2, dtype=dtype)
        model = SimpleModel(hidden_dim=16, nlayers=2)
        engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        try:
            loss = engine(*batches[0])
            engine.backward(loss)
            engine.step()
            before_params = [safe_get_full_fp32_param(p).clone() for p in engine.module.parameters()]
            before_states = [safe_get_full_optimizer_state(p, "exp_avg").clone() for p in engine.module.parameters()]
            before_steps = [int(state["step"]) for state in optimizer.optimizer.state.values()]
            with deepspeed.zero.GatheredParameters(list(engine.module.parameters())):
                before_low_precision = [p.detach().clone() for p in engine.module.parameters()]
            skipped_steps = engine.skipped_steps
            loss = engine(*batches[1])
            if dist.get_rank() == 0:
                loss = loss * invalid_value
            engine.backward(loss)
            with patch.object(optimizer,
                              "_copy_pinned_master_to_device",
                              wraps=optimizer._copy_pinned_master_to_device) as pinned_copy:
                engine.step()
            assert bool(pinned_copy.call_count) == (pin_memory and not offload_param)
            assert engine.skipped_steps == skipped_steps + 1
            assert [int(state["step"]) for state in optimizer.optimizer.state.values()] == before_steps
            for param, before_param, before_state in zip(engine.module.parameters(), before_params, before_states):
                assert torch.equal(safe_get_full_fp32_param(param), before_param)
                assert torch.equal(safe_get_full_optimizer_state(param, "exp_avg"), before_state)
            with deepspeed.zero.GatheredParameters(list(engine.module.parameters())):
                for param, before_param in zip(engine.module.parameters(), before_low_precision):
                    assert torch.equal(param, before_param)
        finally:
            engine.destroy()
