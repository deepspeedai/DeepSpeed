# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Regression tests for ZeRO-1/2 + cpu_offload with multiple engine.backward()
calls per optimizer step (ga_steps=1, driven via set_gradient_accumulation_boundary).
"""

import os
from datetime import timedelta

import pytest
import torch
from torch.utils.checkpoint import checkpoint
import deepspeed
import deepspeed.comm as dist

from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader
from deepspeed.accelerator import get_accelerator
from deepspeed.ops import __compatible_ops__
from deepspeed.utils import safe_get_full_grad


def _base_config(zero_stage, gradient_accumulation_steps=1, cpu_offload=False):
    config_dict = {
        "train_batch_size": gradient_accumulation_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage,
        },
        "zero_force_ds_cpu_optimizer": False,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3,
            },
        },
    }
    if cpu_offload:
        config_dict["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
    if get_accelerator().is_bf16_supported():
        config_dict["bf16"] = {"enabled": True}
    elif get_accelerator().is_fp16_supported():
        config_dict["fp16"] = {"enabled": True}
    return config_dict


def _init_engine(config_dict, hidden_dim, seed=42):
    torch.manual_seed(seed)
    model = SimpleModel(hidden_dim, nlayers=2)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=config_dict,
    )
    return engine


def _capture_params(engine):
    return {name: p.detach().float().cpu().clone() for name, p in engine.module.named_parameters()}


def _assert_params_match(ref, test, label, tol=5e-5):
    for name in ref:
        max_diff = (ref[name] - test[name]).abs().max().item()
        assert max_diff < tol, f"{label}: {name} differs by {max_diff:.3e}"


def _run_multi_backward(config_dict, hidden_dim, num_chunks, num_steps=1, seed=42):
    engine = _init_engine(config_dict, hidden_dim, seed=seed)
    data_loader = random_dataloader(
        model=engine,
        total_samples=num_chunks * num_steps,
        hidden_dim=hidden_dim,
        device=engine.device,
    )
    batches = list(data_loader)
    for step_idx in range(num_steps):
        step_batches = batches[step_idx * num_chunks:(step_idx + 1) * num_chunks]
        for i, batch in enumerate(step_batches):
            loss = engine(batch[0], batch[1])
            engine.set_gradient_accumulation_boundary(i == num_chunks - 1)
            engine.backward(loss)
        engine.step()
    params = _capture_params(engine)
    engine.destroy()
    return params


def _run_ga_microsteps(config_dict, hidden_dim, total_microsteps, seed=42):
    engine = _init_engine(config_dict, hidden_dim, seed=seed)
    data_loader = random_dataloader(
        model=engine,
        total_samples=total_microsteps,
        hidden_dim=hidden_dim,
        device=engine.device,
    )
    for batch in data_loader:
        loss = engine(batch[0], batch[1])
        engine.backward(loss)
        engine.step()
    params = _capture_params(engine)
    engine.destroy()
    return params


@pytest.mark.parametrize("zero_stage", [1, 2])
class TestZeroOffloadMultiBackward(DistributedTest):
    world_size = 1

    def test_multi_backward_matches_no_offload(self, zero_stage):
        hidden_dim = 8
        num_chunks = 4
        ref = _run_multi_backward(_base_config(zero_stage, cpu_offload=False), hidden_dim, num_chunks)
        test = _run_multi_backward(_base_config(zero_stage, cpu_offload=True), hidden_dim, num_chunks)
        _assert_params_match(ref, test, label=f"ZeRO-{zero_stage} N=4")

    def test_single_backward_unchanged(self, zero_stage):
        hidden_dim = 8
        ref = _run_multi_backward(_base_config(zero_stage, cpu_offload=False), hidden_dim, num_chunks=1)
        test = _run_multi_backward(_base_config(zero_stage, cpu_offload=True), hidden_dim, num_chunks=1)
        _assert_params_match(ref, test, label=f"ZeRO-{zero_stage} N=1")

    def test_multi_backward_across_multiple_steps(self, zero_stage):
        hidden_dim = 8
        ref = _run_multi_backward(_base_config(zero_stage, cpu_offload=False), hidden_dim, num_chunks=3, num_steps=3)
        test = _run_multi_backward(_base_config(zero_stage, cpu_offload=True), hidden_dim, num_chunks=3, num_steps=3)
        _assert_params_match(ref, test, label=f"ZeRO-{zero_stage} 3x3")

    def test_single_backward_across_multiple_steps(self, zero_stage):
        hidden_dim = 8
        ref = _run_multi_backward(_base_config(zero_stage, cpu_offload=False), hidden_dim, num_chunks=1, num_steps=3)
        test = _run_multi_backward(_base_config(zero_stage, cpu_offload=True), hidden_dim, num_chunks=1, num_steps=3)
        _assert_params_match(ref, test, label=f"ZeRO-{zero_stage} 3x1")

    def test_ga_greater_than_one_offload_unchanged(self, zero_stage):
        hidden_dim = 8
        ga = 4
        ref = _run_ga_microsteps(_base_config(zero_stage, gradient_accumulation_steps=ga, cpu_offload=False),
                                 hidden_dim,
                                 total_microsteps=ga)
        test = _run_ga_microsteps(_base_config(zero_stage, gradient_accumulation_steps=ga, cpu_offload=True),
                                  hidden_dim,
                                  total_microsteps=ga)
        _assert_params_match(ref, test, label=f"ZeRO-{zero_stage} ga=4")


class RequestBoundaryModel(torch.nn.Module):

    def __init__(self, checkpointed):
        super().__init__()
        self.projections = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 2, bias=False)])
        self.checkpointed = checkpointed

    def forward(self, inputs, projection_indices=(0, 1)):
        loss = inputs.new_zeros(())
        for projection_index in projection_indices:
            projection = self.projections[projection_index]
            if self.checkpointed:
                outputs = checkpoint(projection, inputs, use_reentrant=True)
            else:
                outputs = projection(inputs)
            loss = loss + outputs.sum()
        return loss


def _run_request_boundaries(rank, rendezvous, zero_stage, cpu_offload, accumulation_dtype, checkpointed):
    os.environ["LOCAL_RANK"] = str(rank)
    accelerator = get_accelerator()
    accelerator.set_device(rank)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(__compatible_ops__, "deepspeed_shm_comm", False)
        dist.init_distributed(accelerator.communication_backend_name(),
                              auto_mpi_discovery=False,
                              init_method=rendezvous,
                              rank=rank,
                              world_size=2,
                              timeout=timedelta(seconds=60))
    engine = None
    try:
        model = RequestBoundaryModel(checkpointed).to(device=accelerator.device_name(), dtype=torch.bfloat16)
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(0.5)
        references = {
            name: torch.nn.Parameter(param.detach().float().cpu().clone())
            for name, param in model.named_parameters()
        }
        reference_optimizer = torch.optim.SGD(references.values(), lr=1 / 32)
        config = {
            "train_micro_batch_size_per_gpu": 4,
            "gradient_accumulation_steps": 1,
            "gradient_clipping": 0,
            "bf16": {
                "enabled": True,
                "bf16_master_weights_and_grads": False,
                "bf16_optimizer_states": False,
            },
            "data_types": {
                "grad_accum_dtype": accumulation_dtype
            },
            "zero_allow_untested_optimizer": True,
            "zero_force_ds_cpu_optimizer": False,
            "zero_optimization": {
                "stage": zero_stage,
                "reduce_bucket_size": 8
            },
        }
        if cpu_offload:
            config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": False}
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=torch.optim.SGD(model.parameters(), lr=1 / 32),
                                               config=config,
                                               dist_init_required=False)
        calls = ["c0", "c1", "c2", "c3"]
        windows = [
            (requests, {})
            for requests in [[calls], [["ping"], calls], [["c0"], ["c1"]], [["c0"], ["c1"], ["c2", "c3"]], [["c0"]]]
        ]
        # Each projection first joins a later backward after contributing to the preceding update.
        for early, late in [(0, 1), (1, 0)]:
            windows.append(([["c0"], ["c1", "c2"], ["c3"]], {"c0": (early, ), "c1": (late, ), "c2": (late, )}))
        reference_names = list(references)
        for step, (requests, selections) in enumerate(windows):
            expected = {name: torch.zeros_like(reference) for name, reference in references.items()}
            for request in requests:
                for index, name in enumerate(request):
                    offset = 0 if name == "ping" else calls.index(name)
                    scale = 0 if name == "ping" else 1
                    inputs = (torch.arange(16).reshape(4, 4) + step + rank + offset) / 16
                    engine.set_gradient_accumulation_boundary(index == len(request) - 1)
                    device_inputs = inputs.to(device=engine.device, dtype=torch.bfloat16).requires_grad_(checkpointed)
                    projection_indices = selections.get(name, (0, 1))
                    engine.backward(engine(device_inputs, projection_indices).float() * scale)
                    # Linear sum-loss gradients are input column sums, averaged over the two ranks.
                    rank_mean_inputs = inputs + (0.5 - rank) / 16
                    for projection_index in projection_indices:
                        grad = expected[reference_names[projection_index]]
                        grad.add_(rank_mean_inputs.sum(dim=0).expand_as(grad) * scale)
            for name, param in model.named_parameters():
                torch.testing.assert_close(safe_get_full_grad(param).float().cpu(),
                                           expected[name],
                                           rtol=0,
                                           atol=0,
                                           msg=f"{name}: step={step}, requests={requests}")
                references[name].grad = expected[name]
            reference_optimizer.step()
            engine.step()
            for name, param in model.named_parameters():
                torch.testing.assert_close(param.detach().cpu(),
                                           references[name].detach().to(torch.bfloat16),
                                           rtol=0,
                                           atol=0,
                                           msg=f"{name}: update at step={step}, requests={requests}")
    finally:
        if engine is not None:
            engine.destroy()
        dist.destroy_process_group()


@pytest.mark.skipif(get_accelerator().device_name() not in ("cpu", "cuda")
                    or (get_accelerator().device_name() == "cuda" and get_accelerator().device_count() < 2),
                    reason="Requires CPU or two CUDA devices")
@pytest.mark.parametrize("zero_stage", [1, 2])
@pytest.mark.parametrize("cpu_offload", [False, True], ids=["device-optimizer", "cpu-offload"])
@pytest.mark.parametrize("accumulation_dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("checkpointed", [False, True], ids=["plain", "reentrant"])
def test_multiple_request_boundaries_before_step(tmp_path, zero_stage, cpu_offload, accumulation_dtype, checkpointed):
    torch.multiprocessing.spawn(_run_request_boundaries,
                                args=(f"file://{tmp_path / 'rendezvous'}", zero_stage, cpu_offload, accumulation_dtype,
                                      checkpointed),
                                nprocs=2,
                                join=True)
