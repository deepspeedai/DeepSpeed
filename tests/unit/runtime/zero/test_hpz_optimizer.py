# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from contextlib import nullcontext
from datetime import timedelta
import math
import os
import signal
from unittest.mock import patch

import pytest
import torch

import deepspeed
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.adam import FusedAdam
from deepspeed.utils import safe_get_full_fp32_param, safe_get_local_fp32_param
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from unit.common import get_master_port


def _hpz_config(enabled):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "gradient_clipping": 0.4,
        "bf16": {
            "enabled": True
        },
        "communication_data_type": "fp32",
        "zero_optimization": {
            "stage": 3,
            "zero_hpz_partition_size": 2,
            "stage3_param_persistence_threshold": 0,
        },
    }
    if enabled:
        config["zero_optimization"]["zero_hpz_optimizer_sharding"] = True
    return config


def _input(step, microbatch, rank, size):
    factors = ((-5, -1, 3, 11), (7, -3, -11, 3))
    return (torch.arange(1, size + 1) * (factors[step][rank] + microbatch)).float() / 8


class _HpZModel(torch.nn.Module):

    def __init__(self, rank, frozen_value=1.0):
        super().__init__()
        self.first = torch.nn.Parameter(torch.arange(1, 18).float() / 16 + rank)
        self.second = torch.nn.Parameter(torch.arange(1, 4).float() / 16 + rank)
        self.frozen = torch.nn.Parameter(torch.full((5, ), frozen_value), requires_grad=False)

    @property
    def weights(self):
        return (self.first, self.second)

    def forward(self, inputs):
        return self.frozen.sum() + sum((weight * batch).sum() for weight, batch in zip(self.weights, inputs))


def _adam_param_groups(model):
    return [{"params": [model.first]}, {"params": [model.second], "lr": 0.03, "weight_decay": 0.1}]


def _bits(tensor):
    assert tensor.dtype == torch.float32
    return tensor.detach().cpu().contiguous().view(torch.int32).flatten().tolist()


def _step_bits(step):
    if isinstance(step, torch.Tensor):
        return int(step), str(step.dtype), step.detach().cpu().reshape(1).view(torch.uint8).tolist()
    return step, type(step).__name__, None


def _local_state(engine):
    payload = []
    for param in engine.module.weights:
        shard = engine.optimizer.get_local_hpz_adam_state(param)
        for tensor in (shard.fp32_master, shard.exp_avg, shard.exp_avg_sq):
            assert tensor.shape == (2, math.ceil(param.ds_numel / 4))
        payload.append((tuple(shard.replica_ranks), _bits(shard.fp32_master), _bits(shard.exp_avg),
                        _bits(shard.exp_avg_sq), _step_bits(shard.step)))
    return payload


def _resident_state(engine, optimizer, rank):
    masters = engine.optimizer.fp32_partitioned_groups_flat
    expected_widths = [2 * math.ceil(param.ds_numel / 4) for param in engine.module.weights]
    assert [master.numel() for master in masters] == expected_widths, "Adam shards must span both replica lanes"
    replicas = (rank % 2, rank % 2 + 2)
    payload = []
    for master in masters:
        state = optimizer.state[master]
        assert state["exp_avg"].shape == state["exp_avg_sq"].shape == master.shape
        payload.append(
            (replicas, _bits(master), _bits(state["exp_avg"]), _bits(state["exp_avg_sq"]), _step_bits(state["step"])))
    return payload


def _assert_model_writeback(params):
    masters = [safe_get_full_fp32_param(param) for param in params]
    with deepspeed.zero.GatheredParameters(params):
        rounded_masters = [master.to(param.dtype).float() for param, master in zip(params, masters)]
        assert any(_bits(master) != _bits(rounded) for master, rounded in zip(masters, rounded_masters))
        for param, expected in zip(params, rounded_masters):
            assert _bits(param.float()) == _bits(expected)


def _new_engine(rank, enabled, zero_init, fused, frozen_value=1.0):
    config = _hpz_config(enabled)
    if fused:
        del config["bf16"]
        config["fp16"] = {"enabled": True, "initial_scale_power": 7, "hysteresis": 1}
    if zero_init:
        config["zero_optimization"]["reduce_scatter"] = False
        config["zero_optimization"]["pipeline_loading_checkpoint"] = True
    with deepspeed.zero.Init(config_dict_or_path=config) if zero_init else nullcontext():
        model = _HpZModel(rank, frozen_value)
    if fused:
        optimizer = FusedAdam(_adam_param_groups(model), lr=0.01, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(_adam_param_groups(model),
                                      lr=0.01,
                                      weight_decay=0.01,
                                      foreach=False,
                                      fused=False)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
    return engine


def _train_step(engine, rank, step, gather=None):
    losses = []
    for microbatch in range(2):
        inputs = [
            _input(step, microbatch, rank, size).to(engine.device, engine.module.first.dtype) for size in (17, 3)
        ]
        with patch.object(dist, "allgather_fn", new=gather) if gather is not None else nullcontext():
            loss = engine(inputs)
            losses.append(loss.item())
            engine.backward(loss)
        engine.step()
    assert engine.was_step_applied()
    return losses


def _overflow_step(engine, rank):
    for _ in range(2):
        value = float("inf") if rank == 1 else 1.0
        inputs = [torch.full((size, ), value, device=engine.device, dtype=torch.float16) for size in (17, 3)]
        engine.backward(engine(inputs))
        engine.step()
    assert not engine.was_step_applied()


def _checkpoint_metadata(engine):
    groups = [{k: v for k, v in group.items() if k != "params"} for group in engine.optimizer.optimizer.param_groups]
    return (engine.global_steps, engine.global_samples, engine.skipped_steps, groups,
            vars(engine.optimizer.loss_scaler).copy(), engine.optimizer.dynamic_loss_scale, engine.optimizer.overflow)


def _check_load_options(rank, checkpoint_dir, saved):
    for kwargs in ({"load_module_only": True}, {"load_optimizer_states": False}):
        fresh = _new_engine(rank, enabled=True, zero_init=False, fused=False)
        fresh.load_checkpoint(checkpoint_dir, **kwargs)
        assert fresh.global_steps == (0 if kwargs.get("load_module_only") else 1)
        assert not fresh.optimizer.optimizer.state
        for master, (_, master_bits, *_) in zip(fresh.optimizer.fp32_partitioned_groups_flat, saved):
            assert _bits(master) == master_bits
        with pytest.raises(RuntimeError):
            fresh.optimizer.get_local_hpz_adam_state(fresh.module.first)
        _assert_model_writeback(fresh.module.weights)
        fresh.destroy()


def _resave_without_optimizer_sharding(rank, checkpoint_dir, saved):
    ordinary = _new_engine(rank, enabled=False, zero_init=False, fused=False)
    _, client_state = ordinary.load_checkpoint(checkpoint_dir)
    for master, (replicas, *bits, step) in zip(ordinary.optimizer.fp32_partitioned_groups_flat, saved):
        state = ordinary.optimizer.optimizer.state[master]
        lane = replicas.index(rank)
        for actual, expected in zip((master, state["exp_avg"], state["exp_avg_sq"]), bits):
            width = actual.numel()
            assert len(expected) == 2 * width
            assert _bits(actual) == expected[lane * width:(lane + 1) * width]
        assert _step_bits(state["step"]) == step
    full_masters = {
        name: safe_get_full_fp32_param(p)
        for name, p in ordinary.module.named_parameters() if p.requires_grad
    }
    if rank == 0:
        consolidated = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag="first")
        for name, master in full_masters.items():
            assert _bits(consolidated[name]) == _bits(master)

    shard = torch.load(ordinary._get_zero_ckpt_name(checkpoint_dir, "first"), map_location="cpu", weights_only=False)
    saved_optimizer = shard["optimizer_state_dict"]
    tensors = list(saved_optimizer["fp32_flat_groups"])
    for state in saved_optimizer["optimizer_state_dict"]["state"].values():
        tensors.extend((state["exp_avg"], state["exp_avg_sq"]))
    for tensor in tensors:
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()

    ordinary.save_checkpoint(checkpoint_dir, tag="ordinary", client_state={"epoch": client_state["epoch"]})
    ordinary.destroy()


def _hpz_rank(rank, port, conn, enabled, zero_init, fused, checkpoint_dir):
    os.environ.update(MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=port,
                      WORLD_SIZE="4",
                      LOCAL_SIZE="4",
                      RANK=str(rank),
                      LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    if enabled:
        torch.use_deterministic_algorithms(True)
    accelerator = get_accelerator()
    accelerator.set_device(rank)
    deepspeed.init_distributed(dist_backend=accelerator.communication_backend_name(),
                               timeout=timedelta(seconds=45),
                               auto_mpi_discovery=False)
    engine = _new_engine(rank, enabled, zero_init, fused)
    if enabled and not zero_init and not fused:
        engine.save_checkpoint(checkpoint_dir, tag="initial")
        engine.destroy()
        engine = _new_engine(rank, enabled, zero_init, fused)
        engine.load_checkpoint(checkpoint_dir, tag="initial")
    if enabled:
        with pytest.raises(RuntimeError):
            engine.optimizer.get_local_hpz_adam_state(engine.module.first)

    gather_groups = []
    gather = dist.allgather_fn

    def record_gather(output, tensor, group=None, **kwargs):
        ranks_group = dist.get_world_group() if group is None else group
        gather_groups.append(tuple(dist.get_global_rank(ranks_group, i) for i in range(dist.get_world_size(group))))
        return gather(output, tensor, group=group, **kwargs)

    states = []
    for step in range(2):
        uninterrupted_losses = _train_step(engine, rank, step, gather=record_gather)
        if enabled:
            resident = _resident_state(engine, engine.optimizer.optimizer, rank)
            assert _local_state(engine) == resident
            states.append(resident)
            if step == 0:
                if fused:
                    _overflow_step(engine, rank)
                    assert _local_state(engine) == resident
                saved_metadata = _checkpoint_metadata(engine)
                engine.save_checkpoint(checkpoint_dir, tag="first", client_state={"epoch": 7})
    if enabled:
        _assert_model_writeback(engine.module.weights)
        engine.destroy()
        if not fused and not zero_init:
            _check_load_options(rank, checkpoint_dir, states[0])
            _resave_without_optimizer_sharding(rank, checkpoint_dir, states[0])
        engine = _new_engine(rank, enabled, zero_init, fused, frozen_value=4.0)
        for group in engine.optimizer.optimizer.param_groups:
            group["lr"] = 0.5
        load_path, client_state = engine.load_checkpoint(checkpoint_dir)
        assert load_path and client_state["epoch"] == 7
        assert _checkpoint_metadata(engine) == saved_metadata
        assert _resident_state(engine, engine.optimizer.optimizer, rank) == states[0]
        assert _local_state(engine) == states[0]
        _assert_model_writeback(engine.module.weights)
        resumed_losses = _train_step(engine, rank, 1)
        assert resumed_losses == uninterrupted_losses, (resumed_losses, uninterrupted_losses)
        assert _resident_state(engine, engine.optimizer.optimizer, rank) == states[1]
        assert _local_state(engine) == states[1]
        _assert_model_writeback(engine.module.weights)
        if fused:
            _overflow_step(engine, rank)
            assert _resident_state(engine, engine.optimizer.optimizer, rank) == states[-1]
            assert _local_state(engine) == states[-1]
        if zero_init:
            engine.destroy()
            engine = _new_engine(rank, enabled, zero_init, fused)
            engine.load_checkpoint(checkpoint_dir)
    else:
        with pytest.raises(RuntimeError):
            engine.optimizer.get_local_hpz_adam_state(engine.module.first)
        states = [_bits(safe_get_local_fp32_param(param)) for param in engine.module.weights]
    dist.barrier()
    conn.send((states, gather_groups))
    if conn.recv() == "read":
        conn.send(_local_state(engine))
    conn.close()


def _recv(conn, timeout=180):
    assert conn.poll(timeout), "rank did not reply"
    return conn.recv()


def _assert_adam_reference(states):
    reference = _HpZModel(0)
    optimizer = torch.optim.AdamW(_adam_param_groups(reference),
                                  lr=0.01,
                                  weight_decay=0.01,
                                  foreach=False,
                                  fused=False)
    for step in range(2):
        for weight in reference.weights:
            weight.grad = torch.stack([
                _input(step, microbatch, rank, weight.numel()) for microbatch in range(2) for rank in range(4)
            ]).mean(0)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.4)
        optimizer.step()
        for index, weight in enumerate(reference.weights):
            expected = (weight, optimizer.state[weight]["exp_avg"], optimizer.state[weight]["exp_avg_sq"])
            for rank in (0, 1):
                replicas, *shard, adam_step = states[rank][step][index]
                assert replicas == (rank, rank + 2)
                assert adam_step[0] == step + 1
                width = math.ceil(weight.numel() / 4)
                for bits, full_tensor in zip(shard, expected):
                    assert len(bits) == 2 * width
                    actual = torch.tensor(bits, dtype=torch.int32).view(torch.float32).reshape(2, width)
                    padded = torch.nn.functional.pad(full_tensor.detach(), (0, 4 * width - weight.numel()))
                    expected_lanes = padded.reshape(4, width)[list(replicas)]
                    torch.testing.assert_close(actual, expected_lanes, rtol=5e-5, atol=1e-7)


@pytest.mark.parametrize("enabled,zero_init,fused", [(False, False, False), (True, False, False), (True, True, False),
                                                     (True, False, True)],
                         ids=["legacy-hpz", "hierarchical", "hierarchical-zero-init", "hierarchical-fused-adam"])
def test_hpz_adam_recovery(enabled, zero_init, fused, tmp_path):
    if get_accelerator().device_name() not in ("cpu", "cuda"):
        pytest.skip("requires Gloo or NCCL")
    if fused and get_accelerator().device_name() == "cpu":
        pytest.skip("requires CUDA FusedAdam")
    if get_accelerator().device_name() == "cuda" and get_accelerator().device_count() < 4:
        pytest.skip("requires four GPUs")
    if fused and not get_accelerator().is_fp16_supported():
        pytest.skip("requires FP16 model weights")
    if not fused and not get_accelerator().is_bf16_supported():
        pytest.skip("requires BF16 model weights")

    context = torch.multiprocessing.get_context("spawn")
    workers = []
    port = get_master_port(base_port=29650)
    try:
        for rank in range(4):
            parent, child = context.Pipe()
            process = context.Process(target=_hpz_rank, args=(rank, port, child, enabled, zero_init, fused, tmp_path))
            process.start()
            child.close()
            workers.append((process, parent))

        states = []
        for rank, (_, conn) in enumerate(workers):
            state, gather_groups = _recv(conn)
            subgroup = (0, 1) if rank < 2 else (2, 3)
            assert subgroup in gather_groups
            if not enabled:
                assert tuple(map(len, state)) == (5, 1)
            states.append(state)

        if enabled:
            _assert_adam_reference(states)
            for rank in (0, 1):
                assert states[rank] == states[rank + 2]

            owner, _ = workers[0]
            owner.kill()
            owner.join(10)
            assert owner.exitcode == -signal.SIGKILL
            _, peer = workers[2]
            peer.send("read")
            expected = states[0][0] if zero_init else states[0][-1]
            assert _recv(peer, timeout=20) == expected
    finally:
        for process, conn in workers:
            if process.is_alive():
                process.kill()
            process.join(10)
            conn.close()
