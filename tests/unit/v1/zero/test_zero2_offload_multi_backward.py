# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Regression tests for ZeRO-1/2 + cpu_offload with multiple engine.backward()
calls per optimizer step (ga_steps=1, driven via set_gradient_accumulation_boundary).
"""

import pytest
import torch
import deepspeed

from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import CPUAdamBuilder


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

    def test_single_backward_allocates_no_cpu_accumulator(self, zero_stage):
        hidden_dim = 8
        engine = _init_engine(_base_config(zero_stage, cpu_offload=True), hidden_dim)
        batch = next(
            iter(random_dataloader(model=engine, total_samples=1, hidden_dim=hidden_dim, device=engine.device)))
        loss = engine(batch[0], batch[1])
        engine.set_gradient_accumulation_boundary(True)
        engine.backward(loss)
        engine.step()
        populated = len(engine.optimizer.accumulated_grads_in_cpu)
        engine.destroy()
        assert populated == 0, f"ZeRO-{zero_stage}: ga=1+N=1 populated accumulated_grads_in_cpu ({populated} entries)"

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


class TestZero2OffloadOverflowRecovery(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("precision", ["bf16", "fp16"])
    @pytest.mark.parametrize("low_precision_master", [True, False])
    def test_finite_step_after_overflow(self, precision, low_precision_master):
        if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("CPUAdam is required for low-precision offloaded master weights")
        if precision == "bf16" and not get_accelerator().is_bf16_supported():
            pytest.skip("BF16 is not supported")

        class TwoGroups(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.large = torch.nn.Parameter(torch.ones(12))
                self.small = torch.nn.Parameter(torch.ones(8))

            def forward(self, overflow):
                return self.large.sum() + self.small.sum() * (float("nan") if overflow else 1.0)

        config = _base_config(2, cpu_offload=True)
        config.pop("bf16", None)
        config.pop("fp16", None)
        config[precision] = {"enabled": True, f"{precision}_master_weights_and_grads": low_precision_master}
        if precision == "fp16":
            config[precision]["loss_scale"] = 1.0
        config["train_batch_size"] = 2
        config["managed_gradient_accumulation"] = False
        config["optimizer"]["params"]["lr"] = 0.1
        config["zero_optimization"].update({"reduce_bucket_size": 32, "allgather_bucket_size": 32})
        model = TwoGroups()
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=[{
                                                   "params": [model.large]
                                               }, {
                                                   "params": [model.small]
                                               }],
                                               config=config)
        try:
            initial = _capture_params(engine)
            engine.backward(engine(True))
            engine.step()
            _assert_params_match(initial, _capture_params(engine), "overflow must skip the update", tol=1e-8)

            # Rank 1 owns [6:12] of large and [4:8] of small. The small
            # group's previous NaNs must not make a later finite step skip.
            for _ in range(2):
                before = _capture_params(engine)
                engine.backward(engine(False))
                engine.step()
                after = _capture_params(engine)
                for name in before:
                    assert torch.isfinite(after[name]).all()
                    assert not torch.equal(before[name], after[name]), f"finite step did not update {name}"
        finally:
            engine.destroy()
