# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import glob
import os

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.constants import UNIVERSAL_CHECKPOINT_INFO
from deepspeed.runtime.zero.utils import ZeRORuntimeException
from unit.common import DistributedTest, get_master_port

TAG = "fp32-masters"


class _OneParam(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 4))

    def forward(self, x):
        return (x @ self.weight).sum()


def _config(*, load_universal=False, world_size=1):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": world_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.1
            },
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
        },
    }
    if load_universal:
        config["checkpoint"] = {"load_universal": True}
    return config


def _init_engine(*, load_universal=False, world_size=1):
    model = _OneParam()
    engine, _, _, _ = deepspeed.initialize(config=_config(load_universal=load_universal, world_size=world_size),
                                           model=model,
                                           model_parameters=model.parameters())
    return engine


def _unpadded_master(engine):
    master = engine.optimizer.single_partition_of_fp32_groups[0].detach().cpu()
    padding = engine.optimizer.groups_padding[0]
    return master[:-padding] if padding else master


def _train_and_detach_masters(engine):
    data = torch.ones(1, 4, device=engine.device, dtype=torch.bfloat16)
    engine.backward(engine(data))
    engine.step()
    # A sub-bf16 offset keeps the saved master off the bf16 grid without another step.
    with torch.no_grad():
        engine.optimizer.single_partition_of_fp32_groups[0].add_(2**-12)
    assert _has_moments(engine)
    saved = engine.optimizer.single_partition_of_fp32_groups[0].detach().clone()
    assert not torch.equal(saved, saved.to(torch.bfloat16).to(saved.dtype))
    return saved


def _has_moments(engine):
    return any("exp_avg" in state for state in engine.optimizer.optimizer.state.values())


def _masters(engine):
    return engine.optimizer.single_partition_of_fp32_groups[0].detach()


def _init_dist(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_SIZE"] = str(world_size)
    deepspeed.init_distributed(dist_backend="gloo", auto_mpi_discovery=False, rank=rank, world_size=world_size)


def _save_dp2(rank, world_size, tmpdir, port):
    _init_dist(rank, world_size, port)
    torch.manual_seed(0)
    engine = _init_engine(world_size=world_size)
    _train_and_detach_masters(engine)
    torch.save(_unpadded_master(engine), os.path.join(tmpdir, f"part_{rank}.pt"))
    legacy_dir = os.path.join(tmpdir, "dp2")
    engine.save_checkpoint(legacy_dir, tag=TAG, client_state={UNIVERSAL_CHECKPOINT_INFO: {}})
    # The same masters saved in the elastic layout, which stores per-group partitions without optimizer ids.
    engine.optimizer.elastic_checkpoint = True
    engine.save_checkpoint(os.path.join(tmpdir, "dp2_elastic"), tag=TAG)
    engine.optimizer.elastic_checkpoint = False
    dist.barrier()
    if rank == 0:
        from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal

        convert_to_universal(
            argparse.Namespace(input_folder=os.path.join(legacy_dir, TAG),
                               output_folder=os.path.join(tmpdir, "universal", TAG),
                               num_extract_workers=1,
                               num_merge_workers=1,
                               keep_temp_folder=False,
                               strict=True,
                               inject_missing_state=False))
    dist.barrier()
    dist.destroy_process_group()


def _saved_dp2_masters(tmpdir):
    return torch.cat([torch.load(os.path.join(tmpdir, f"part_{rank}.pt"), weights_only=True) for rank in range(2)])


def _load_dp1(tmpdir, port):
    _init_dist(0, 1, port)
    loaded = _init_engine(load_universal=True, world_size=1)
    loaded.load_checkpoint(os.path.join(tmpdir, "universal"),
                           tag=TAG,
                           load_module_only=False,
                           load_optimizer_states=False,
                           load_lr_scheduler_states=False)
    expected = _saved_dp2_masters(tmpdir)
    assert torch.equal(_unpadded_master(loaded), expected)
    assert not torch.equal(expected, expected.to(torch.bfloat16).to(expected.dtype))
    assert not _has_moments(loaded)
    dist.destroy_process_group()


def _load_dp1_legacy(tmpdir, port):
    _init_dist(0, 1, port)
    loaded = _init_engine(world_size=1)
    with pytest.raises(ZeRORuntimeException, match="DP world size"):
        loaded.load_checkpoint(os.path.join(tmpdir, "dp2"),
                               tag=TAG,
                               load_module_only=False,
                               load_optimizer_states=False,
                               load_lr_scheduler_states=False)
    dist.destroy_process_group()


def _load_dp1_elastic(tmpdir, port):
    _init_dist(0, 1, port)
    loaded = _init_engine(world_size=1)
    loaded.load_checkpoint(os.path.join(tmpdir, "dp2_elastic"),
                           tag=TAG,
                           load_module_only=False,
                           load_optimizer_states=False,
                           load_lr_scheduler_states=False)
    expected = _saved_dp2_masters(tmpdir)
    assert torch.equal(_unpadded_master(loaded), expected)
    assert not torch.equal(expected, expected.to(torch.bfloat16).to(expected.dtype))
    assert not _has_moments(loaded)

    with_optimizer = _init_engine(world_size=1)
    with pytest.raises(ZeRORuntimeException, match="DP world size"):
        with_optimizer.load_checkpoint(os.path.join(tmpdir, "dp2_elastic"), tag=TAG, load_optimizer_states=True)
    dist.destroy_process_group()


def _join(processes):
    failed = []
    for rank, process in enumerate(processes):
        process.join(180)
        if process.exitcode != 0:
            failed.append((rank, process.exitcode))
    assert not failed, failed


def _spawn(target, *args):
    import torch.multiprocessing as mp

    process = mp.get_context("spawn").Process(target=target, args=args)
    process.start()
    _join([process])


@pytest.fixture(scope="module")
def dp2_checkpoint(tmp_path_factory):
    if not get_accelerator().is_bf16_supported():
        pytest.skip("bf16 is not supported")
    import torch.multiprocessing as mp

    tmpdir = str(tmp_path_factory.mktemp("dp2"))
    ctx = mp.get_context("spawn")
    port = get_master_port()
    savers = [ctx.Process(target=_save_dp2, args=(rank, 2, tmpdir, port)) for rank in range(2)]
    for process in savers:
        process.start()
    _join(savers)
    return tmpdir


def test_universal_dp_change_keeps_masters_and_drops_moments(dp2_checkpoint):
    _spawn(_load_dp1, dp2_checkpoint, get_master_port())


def test_legacy_dp_change_rejects_incompatible_fp32_partitions(dp2_checkpoint):
    _spawn(_load_dp1_legacy, dp2_checkpoint, get_master_port())


def test_elastic_dp_change_repartitions_masters_and_drops_moments(dp2_checkpoint):
    _spawn(_load_dp1_elastic, dp2_checkpoint, get_master_port())


class TestFp32MasterOnlyLoad(DistributedTest):
    world_size = 1
    backend = "gloo"
    requires_cuda_env = False

    def test_legacy_load_keeps_masters_and_drops_moments(self, tmpdir):
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported")

        engine = _init_engine()
        saved = _train_and_detach_masters(engine)
        load_dir = os.path.join(str(tmpdir), "legacy")
        engine.save_checkpoint(load_dir, tag=TAG)

        module_only = _init_engine()
        module_only.load_checkpoint(load_dir, tag=TAG, load_module_only=True)
        rebuilt = _masters(module_only).cpu()
        expected = saved.to(torch.bfloat16).to(saved.dtype).cpu()
        assert torch.equal(rebuilt, expected)
        assert not _has_moments(module_only)

        loaded = _init_engine()
        loaded.load_checkpoint(load_dir,
                               tag=TAG,
                               load_module_only=False,
                               load_optimizer_states=False,
                               load_lr_scheduler_states=False)
        assert torch.equal(_masters(loaded).cpu(), saved.cpu())
        assert not _has_moments(loaded)

    def test_universal_load_keeps_masters_and_drops_moments(self, tmpdir):
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported")

        engine = _init_engine()
        saved = _train_and_detach_masters(engine)
        legacy_dir = os.path.join(str(tmpdir), "legacy")
        engine.save_checkpoint(legacy_dir, tag=TAG, client_state={UNIVERSAL_CHECKPOINT_INFO: {}})

        from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal

        source = os.path.join(legacy_dir, TAG)
        universal_tag_dir = os.path.join(str(tmpdir), "universal", TAG)
        convert_to_universal(
            argparse.Namespace(input_folder=source,
                               output_folder=universal_tag_dir,
                               num_extract_workers=1,
                               num_merge_workers=1,
                               keep_temp_folder=False,
                               strict=True,
                               inject_missing_state=False))

        loaded = _init_engine(load_universal=True)
        loaded.load_checkpoint(os.path.join(str(tmpdir), "universal"),
                               tag=TAG,
                               load_module_only=False,
                               load_optimizer_states=False,
                               load_lr_scheduler_states=False)
        assert torch.equal(_masters(loaded).cpu(), saved.cpu())
        assert not _has_moments(loaded)

        module_only = _init_engine(load_universal=True)
        module_only.load_checkpoint(os.path.join(str(tmpdir), "universal"), tag=TAG, load_module_only=True)
        rebuilt = _masters(module_only).cpu()
        expected = saved.to(torch.bfloat16).to(saved.dtype).cpu()
        assert torch.equal(rebuilt, expected)
        assert not _has_moments(module_only)

    def test_missing_fp32_shards_only_allows_module_only_load(self, tmpdir):
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported")

        engine = _init_engine()
        saved = _train_and_detach_masters(engine)
        load_dir = os.path.join(str(tmpdir), "legacy")
        engine.save_checkpoint(load_dir, tag=TAG)

        shard_paths = glob.glob(os.path.join(load_dir, TAG, "*optim_states.pt"))
        assert shard_paths
        for path in shard_paths:
            os.rename(path, path + ".absent")

        module_only = _init_engine()
        module_only.load_checkpoint(load_dir, tag=TAG, load_module_only=True)
        rebuilt = _masters(module_only).cpu()
        expected = saved.detach().to(torch.bfloat16).to(saved.dtype).cpu()
        assert torch.equal(rebuilt, expected)
        assert not _has_moments(module_only)

        loaded = _init_engine()
        with pytest.raises(ZeRORuntimeException, match="Saved fp32 weights are required"):
            loaded.load_checkpoint(load_dir,
                                   tag=TAG,
                                   load_module_only=False,
                                   load_optimizer_states=False,
                                   load_lr_scheduler_states=False)
