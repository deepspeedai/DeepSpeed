# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Checkpoint metadata tests for AutoEP + AutoTP folding."""

import glob
import os

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.checkpoint.autoep_universal import (
    consolidate_autoep_expert_files,
    make_folding_metadata,
    validate_folding_metadata,
)
from deepspeed.checkpoint.constants import (
    FOLDING_DISPATCH_STRATEGY,
    FOLDING_EP_SIZE,
    FOLDING_ETP_RANK,
    FOLDING_ETP_SIZE,
    FOLDING_FAMILY,
    FOLDING_METADATA_KEY,
    FOLDING_PARAM_FAMILIES,
    FOLDING_SHARED_EXPERT_PLACEMENT,
    FOLDING_TP_SIZE,
    FOLDING_ZERO_PARTITION_GROUP,
    FOLDING_ZERO_PARTITION_COUNT,
)
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.utils import groups
from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import (
    MockMoEOnlyTransformer,
    engine_input_dtype,
    make_autoep_config,
    run_cpu_gloo_test,
    seed_everything,
    skip_unless_h100_tests_enabled,
)


def test_folding_metadata_records_all_mvp_axes():
    metadata = make_folding_metadata(tp_size=2,
                                     tp_rank=1,
                                     ep_size=4,
                                     ep_rank=3,
                                     zero_partition_group="edp",
                                     zero_partition_rank=1,
                                     zero_partition_count=2,
                                     family="routed_expert")
    folding = metadata

    assert folding["version"] == 1
    assert folding[FOLDING_TP_SIZE] == 2
    assert folding[FOLDING_EP_SIZE] == 4
    assert folding[FOLDING_ETP_SIZE] == 1
    assert folding[FOLDING_ETP_RANK] == 0
    assert folding[FOLDING_ZERO_PARTITION_GROUP] == "edp"
    assert folding[FOLDING_DISPATCH_STRATEGY] == "route_full_partition_dispatch"
    assert folding[FOLDING_SHARED_EXPERT_PLACEMENT] == "tp_sharded"
    assert folding[FOLDING_FAMILY] == "routed_expert"


def test_validate_folding_metadata_rejects_missing_and_mismatched_topology():
    folding = make_folding_metadata(tp_size=2,
                                    tp_rank=0,
                                    ep_size=4,
                                    ep_rank=0,
                                    zero_partition_group="dense_dp",
                                    zero_partition_rank=0,
                                    zero_partition_count=4,
                                    family="dense")
    wrapped = {FOLDING_METADATA_KEY: folding}

    assert validate_folding_metadata(wrapped,
                                     tp_size=2,
                                     ep_size=4,
                                     zero_partition_group="dense_dp",
                                     zero_partition_count=4,
                                     family="dense",
                                     shared_expert_placement="tp_sharded")[FOLDING_TP_SIZE] == 2
    with pytest.raises(RuntimeError, match="Missing AutoEP\\+AutoTP folding metadata"):
        validate_folding_metadata({}, tp_size=2, ep_size=4)
    with pytest.raises(RuntimeError, match="tp_size"):
        validate_folding_metadata(wrapped, tp_size=1, ep_size=4)
    with pytest.raises(RuntimeError, match="ep_size"):
        validate_folding_metadata(wrapped, tp_size=2, ep_size=2)
    wrapped[FOLDING_METADATA_KEY][FOLDING_ETP_SIZE] = 2
    with pytest.raises(RuntimeError, match="etp_size"):
        validate_folding_metadata(wrapped, tp_size=2, ep_size=4)
    wrapped[FOLDING_METADATA_KEY][FOLDING_ETP_SIZE] = 1
    wrapped[FOLDING_METADATA_KEY][FOLDING_ZERO_PARTITION_COUNT] = 2
    with pytest.raises(RuntimeError, match="zero_partition_count"):
        validate_folding_metadata(wrapped, tp_size=2, ep_size=4, zero_partition_count=4)
    wrapped[FOLDING_METADATA_KEY][FOLDING_ZERO_PARTITION_COUNT] = 4
    wrapped[FOLDING_METADATA_KEY][FOLDING_FAMILY] = "routed_expert"
    with pytest.raises(RuntimeError, match="family"):
        validate_folding_metadata(wrapped, tp_size=2, ep_size=4, family="dense")
    wrapped[FOLDING_METADATA_KEY][FOLDING_FAMILY] = "dense"
    wrapped[FOLDING_METADATA_KEY][FOLDING_SHARED_EXPERT_PLACEMENT] = "replicated"
    with pytest.raises(RuntimeError, match="shared_expert_placement"):
        validate_folding_metadata(wrapped, tp_size=2, ep_size=4, shared_expert_placement="tp_sharded")
    wrapped[FOLDING_METADATA_KEY][FOLDING_SHARED_EXPERT_PLACEMENT] = "tp_sharded"
    wrapped[FOLDING_METADATA_KEY][FOLDING_ETP_RANK] = 1
    with pytest.raises(RuntimeError, match="etp_rank"):
        validate_folding_metadata(wrapped, tp_size=2, ep_size=4, etp_rank=0)


def _folded_checkpoint_config(*, ep_size=2, mixed_precision=True):
    config = make_autoep_config(zero_stage=0, ep_size=ep_size, mixed_precision=mixed_precision)
    if not mixed_precision:
        config["optimizer"]["params"]["torch_adam"] = True
    config["tensor_parallel"] = {
        "autotp_size": 2,
        "partition_config": {
            "use_default_specs": False,
            "layer_specs": [{
                "patterns": [".*\\.weight$"],
                "partition_type": "skip",
            }],
        },
    }
    return config


def test_validate_routed_expert_metadata_accepts_edp_replica_readers():
    folding = make_folding_metadata(tp_size=2,
                                    tp_rank=0,
                                    ep_size=2,
                                    ep_rank=1,
                                    zero_partition_group="edp",
                                    zero_partition_rank=0,
                                    zero_partition_count=4,
                                    family="routed_expert")
    wrapped = {FOLDING_METADATA_KEY: folding}

    assert validate_folding_metadata(wrapped,
                                     tp_size=2,
                                     ep_size=2,
                                     tp_rank=0,
                                     ep_rank=1,
                                     zero_partition_group="edp",
                                     zero_partition_count=4,
                                     family="routed_expert")[FOLDING_EP_SIZE] == 2
    with pytest.raises(RuntimeError, match="zero_partition_rank"):
        validate_folding_metadata(wrapped,
                                  tp_size=2,
                                  ep_size=2,
                                  tp_rank=0,
                                  ep_rank=1,
                                  zero_partition_group="edp",
                                  zero_partition_rank=1,
                                  zero_partition_count=4,
                                  family="routed_expert")


def test_folded_checkpoint_metadata_rejects_unfolded_runtime():
    state = {
        FOLDING_METADATA_KEY:
        make_folding_metadata(tp_size=2,
                              tp_rank=0,
                              ep_size=2,
                              ep_rank=0,
                              zero_partition_group="dense_dp",
                              zero_partition_rank=0,
                              zero_partition_count=2,
                              family="dense")
    }

    with pytest.raises(RuntimeError, match="requires a folded runtime"):
        DeepSpeedEngine._validate_autoep_folding_checkpoint_metadata(state,
                                                                     folding_spec=None,
                                                                     family="dense",
                                                                     zero_partition_group="dense_dp",
                                                                     zero_partition_count=2)


def test_universal_conversion_rejects_folded_tp_expert_shards(tmpdir):
    checkpoint_dir = tmpdir.mkdir("folded")
    output_dir = tmpdir.mkdir("universal")
    for tp_rank in (0, 1):
        torch.save(
            {
                FOLDING_METADATA_KEY:
                make_folding_metadata(tp_size=2,
                                      tp_rank=tp_rank,
                                      ep_size=2,
                                      ep_rank=0,
                                      zero_partition_group="edp",
                                      zero_partition_rank=0,
                                      zero_partition_count=1,
                                      family="routed_expert"),
                "experts.w1.0":
                torch.ones(2, 2),
            },
            os.path.join(str(checkpoint_dir), f"layer_0_expert_0_mp_rank_0{tp_rank}_model_states.pt"),
        )

    with pytest.raises(NotImplementedError, match="folded AutoEP\\+AutoTP expert shards"):
        consolidate_autoep_expert_files(str(checkpoint_dir), str(output_dir), [{
            "moe_layer_id": 0,
            "num_experts": 1,
            "expert_key_prefix": "experts",
        }])


def _load_torch_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _drop_folding_metadata_from_model_checkpoints(checkpoint_dir):
    model_paths = sorted(glob.glob(os.path.join(str(checkpoint_dir), "folded", "mp_rank_*_model_states.pt")))
    assert model_paths
    if dist.get_rank() == 0:
        for path in model_paths:
            state = _load_torch_checkpoint(path)
            state.pop(FOLDING_METADATA_KEY, None)
            torch.save(state, path)
    dist.barrier()


def _assert_saved_checkpoint_metadata(checkpoint_dir, *, ep_size=2):
    model_paths = sorted(glob.glob(os.path.join(str(checkpoint_dir), "folded", "mp_rank_*_model_states.pt")))
    expert_paths = sorted(
        glob.glob(os.path.join(str(checkpoint_dir), "folded", "layer_*_expert_*_mp_rank_*_model_states.pt")))
    optim_paths = sorted(glob.glob(os.path.join(str(checkpoint_dir), "folded", "*_optim_states.pt")))

    assert model_paths
    assert expert_paths
    assert optim_paths

    for path in model_paths:
        state = _load_torch_checkpoint(path)
        folding = validate_folding_metadata(state, tp_size=2, ep_size=ep_size, etp_rank=0)
        assert folding[FOLDING_FAMILY] == "dense"
        assert folding[FOLDING_ZERO_PARTITION_GROUP] == "dense_dp"
        param_families = folding[FOLDING_PARAM_FAMILIES]
        assert param_families["model.layers.0.mlp.router.gate.weight"] == "router_gate_replicated"
        assert param_families["model.layers.0.dense.weight"] == "dense"
        assert all(not key.endswith("experts.w1") for key in param_families)

    for path in expert_paths:
        state = _load_torch_checkpoint(path)
        folding = validate_folding_metadata(state, tp_size=2, ep_size=ep_size)
        assert folding[FOLDING_FAMILY] == "routed_expert"
        assert folding[FOLDING_ZERO_PARTITION_GROUP] == "edp"

    assert any(FOLDING_METADATA_KEY in _load_torch_checkpoint(path) for path in optim_paths)


def _run_folded_checkpoint_same_topology_resume(checkpoint_dir, *, ep_size=2, mixed_precision=True):
    config = _folded_checkpoint_config(ep_size=ep_size, mixed_precision=mixed_precision)
    seed_everything(1234)
    engine, _, _, _ = deepspeed.initialize(model=MockMoEOnlyTransformer(), config=config)
    folded_layers = [module for module in engine.module.modules() if isinstance(module, AutoEPMoELayer)]
    assert folded_layers
    torch.manual_seed(1234)
    x = torch.randn(1, 4, 64, device=engine.device, dtype=engine_input_dtype(engine))
    dist.broadcast(x, groups.get_tensor_model_parallel_src_rank(), group=groups.get_tensor_model_parallel_group())
    loss = engine(x).float().mean()
    engine.backward(loss)
    engine.step()
    engine.save_checkpoint(str(checkpoint_dir), tag="folded")
    _assert_saved_checkpoint_metadata(checkpoint_dir, ep_size=ep_size)

    seed_everything(1234)
    reloaded, _, _, _ = deepspeed.initialize(model=MockMoEOnlyTransformer(), config=config)
    reloaded.load_checkpoint(str(checkpoint_dir), tag="folded")
    resumed_loss = reloaded(x).float().mean()
    assert torch.isfinite(resumed_loss.detach()).item()

    _drop_folding_metadata_from_model_checkpoints(checkpoint_dir)
    seed_everything(1234)
    missing_metadata, _, _, _ = deepspeed.initialize(model=MockMoEOnlyTransformer(), config=config)
    with pytest.raises(RuntimeError, match="Missing AutoEP\\+AutoTP folding metadata"):
        missing_metadata.load_checkpoint(str(checkpoint_dir), tag="folded")


def _cpu_folded_checkpoint_worker(_rank, _world_size, shared_tmpdir):
    _run_folded_checkpoint_same_topology_resume(shared_tmpdir, mixed_precision=False)


def test_cpu_gloo_folded_checkpoint_edp_replica_resume(tmpdir):
    run_cpu_gloo_test(_cpu_folded_checkpoint_worker, tmpdir, world_size=8)


class TestH100FoldedCheckpoint(DistributedTest):
    world_size = 4
    reuse_dist_env = False

    def test_h100_folded_checkpoint_same_topology_resume(self, tmpdir):
        skip_unless_h100_tests_enabled("H100 checkpoint resume node")

        _run_folded_checkpoint_same_topology_resume(tmpdir)


class TestH100FoldedCheckpointTP2EP4(DistributedTest):
    world_size = 8
    reuse_dist_env = False

    def test_h100_folded_tp2_ep4_checkpoint_same_topology_resume(self, tmpdir):
        skip_unless_h100_tests_enabled("H100 TP2-EP4 checkpoint resume node")

        _run_folded_checkpoint_same_topology_resume(tmpdir, ep_size=4)
