# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Execution-policy contract and independent-process CUDA training tests."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.determinism import _is_nvidia_cuda, apply_determinism


def test_runtime_rejects_rocm(monkeypatch):

    class MockAccelerator:
        _name = "cuda"

        @staticmethod
        def device_name():
            return "cuda"

    monkeypatch.setattr("deepspeed.accelerator.get_accelerator", lambda: MockAccelerator())
    monkeypatch.setattr(torch.version, "hip", "6.0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        apply_determinism(DeepSpeedConfig({"train_batch_size": 1, "deterministic_mode": True}))


def test_runtime_rejects_cuda_compatible_accelerator(monkeypatch):

    class MockAccelerator:
        _name = "klxpu"

        @staticmethod
        def device_name():
            return "cuda"

    monkeypatch.setattr("deepspeed.accelerator.get_accelerator", lambda: MockAccelerator())
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    before = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        apply_determinism(DeepSpeedConfig({"train_batch_size": 1, "deterministic_mode": True}))
    assert torch.are_deterministic_algorithms_enabled() == before


@pytest.mark.parametrize("workspace", [None, "", "invalid"])
def test_runtime_requires_workspace(monkeypatch, workspace):
    if not _is_nvidia_cuda(get_accelerator()):
        pytest.skip("NVIDIA CUDA policy")
    if workspace is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        apply_determinism(DeepSpeedConfig({"train_batch_size": 1, "deterministic_mode": True}))


def test_disabled_runtime_is_noop(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    before = torch.are_deterministic_algorithms_enabled()
    apply_determinism(DeepSpeedConfig({"train_batch_size": 1}))
    assert torch.are_deterministic_algorithms_enabled() == before


def assert_equal(left, right):
    if torch.is_tensor(left):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                           right.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert type(left) is type(right) and left == right


@pytest.mark.parametrize("stage", [0, 2, 3])
def test_independent_cuda_training(stage, tmp_path):
    if not _is_nvidia_cuda(get_accelerator()) or not get_accelerator().is_available():
        pytest.skip("NVIDIA CUDA training")
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1")
    results = []
    for repeat in range(2):
        output = tmp_path / f"repeat{repeat}.pt"
        subprocess.run([
            sys.executable,
            str(Path(torch.__file__).parent / "distributed" / "run.py"), "--standalone", "--nproc_per_node=1",
            str(Path(__file__).resolve()), "--stage",
            str(stage), "--output",
            str(output)
        ],
                       env=env,
                       check=True,
                       timeout=180)
        results.append(torch.load(output, map_location="cpu", weights_only=False))
    assert_equal(*results)


def worker(stage, output):
    import deepspeed
    from deepspeed import comm as dist
    from deepspeed.utils import safe_get_full_grad, safe_get_full_fp32_param, safe_get_full_optimizer_state

    get_accelerator().set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(1234)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    engine, _, _, _ = deepspeed.initialize(model=model,
                                           optimizer=optimizer,
                                           config={
                                               "train_micro_batch_size_per_gpu": 2,
                                               "gradient_accumulation_steps": 2,
                                               "deterministic_mode": True,
                                               "zero_allow_untested_optimizer": True,
                                               "zero_optimization": {
                                                   "stage": stage,
                                                   "overlap_comm": False
                                               },
                                           })
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert not torch.backends.cudnn.benchmark
    assert torch.backends.cudnn.deterministic
    snapshots = []

    def collect(kind):
        values = []
        for parameter in engine.module.parameters():
            if stage == 0:
                if kind == "gradient":
                    value = parameter.grad
                elif kind == "master":
                    value = parameter
                else:
                    value = optimizer.state[parameter][kind]
            elif kind == "gradient":
                value = safe_get_full_grad(parameter)
            elif kind == "master":
                value = safe_get_full_fp32_param(parameter)
            else:
                value = safe_get_full_optimizer_state(parameter, kind)
            assert value is not None
            values.append(value.detach().cpu().clone())
        return values

    for microstep in range(6):
        value = engine(torch.randn(2, 8, device=engine.device))
        engine.backward(value.square().mean())
        boundary = engine.is_gradient_accumulation_boundary()
        snapshot = {"output": value.detach().cpu().clone()}
        if boundary:
            snapshot["gradient"] = collect("gradient")
        engine.step()
        if boundary:
            snapshot.update({kind: collect(kind) for kind in ("master", "exp_avg", "exp_avg_sq")})
            snapshot.update(global_steps=engine.global_steps, micro_steps=engine.micro_steps)
        snapshots.append(snapshot)
    torch.save(snapshots, output)
    engine.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worker(args.stage, args.output)
