# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from copy import deepcopy

import pytest

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from unit.common import DistributedTest, enable_determinism, allclose_on_all_ranks
from unit.simple_model import SimpleModel
from unit.util import bf16_required_version_check

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero import GatheredParameters
from deepspeed.runtime.torch_autocast import PARAM_COMM_DTYPE_ATTR_NAME, get_comm_dtype


def cls_to_qualname(cls):
    return f"{cls.__module__}.{cls.__name__}"


class SimpleModelWithLayerNorm(torch.nn.Module):

    def __init__(self, hidden_dim, nlayers=1):
        super(SimpleModelWithLayerNorm, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for i in range(nlayers)])
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.linears[0](x)
        x = self.norm(x)
        return self.cross_entropy_loss(x, y)


def step_amp(enabled, baseline_model, baseline_optimizer, target_engine, dtype, enable_autocast_outside,
             baseline_scaler, step, x, y, expect_match):
    device_type = get_accelerator().device_name()

    # Runs the forward pass with autocasting.
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled):
        baseline_optimizer.zero_grad()
        baseline_loss = baseline_model(x, y)

    baseline_scaler.scale(baseline_loss).backward()
    baseline_scaler.step(baseline_optimizer)
    baseline_scaler.update()

    # We don't need torch.autocast here in real applications, but want to test the behavior of nested autocast.
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=enable_autocast_outside):
        target_loss = target_engine(x, y)

    # reduce-scatter in `dtype` makes a difference in the loss.
    if step <= 1 and expect_match:
        allclose_on_all_ranks(baseline_loss, target_loss)

    target_engine.backward(target_loss)
    target_engine.step()


@enable_determinism(123)
def compare_loss(model_cls,
                 enable,
                 zero_stage,
                 model_dtype,
                 dtype,
                 autocast_conf,
                 enable_autocast_outside,
                 lower_precision_safe_modules,
                 expect_match=True):
    iteration = 5
    hidden_dim = 10
    lr = 0.001

    if dtype == torch.bfloat16 and not bf16_required_version_check():
        raise ValueError(
            "DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
        )

    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage,
        },
        "torch_autocast": autocast_conf,
    }

    model = model_cls(hidden_dim)
    model.to(model_dtype)

    deepspeed.init_distributed(dist_backend='nccl')

    i = get_accelerator().current_device()
    device = get_accelerator().current_device_name()
    baseline_model = DDP(deepcopy(model).to(device=device, dtype=torch.float32), device_ids=[i], output_device=i)
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=lr, weight_decay=0.0)
    baseline_scaler = torch.amp.GradScaler()

    stage_3_enabled = config_dict["zero_optimization"]["stage"] == 3
    if stage_3_enabled:
        # Trick to avoid conversion to fp32 in Init() while also avoiding deepspeed's mixed precision
        # Ideally Init() should have a flag to avoid conversion to fp32
        import copy
        config_for_init = copy.deepcopy(config_dict)
        if model_dtype == torch.float16:
            config_for_init["fp16"] = {"enabled": True}
        elif model_dtype == torch.bfloat16:
            config_for_init["bf16"] = {"enabled": True}

        with deepspeed.zero.Init(config_dict_or_path=config_for_init):
            target_model = model_cls(hidden_dim)
        with GatheredParameters(target_model.parameters(), modifier_rank=0):
            for p1, p2 in zip(target_model.parameters(), model.parameters()):
                p1.data.copy_(p2.data)
    else:
        target_model = deepcopy(model)

    ds_optimizer = torch.optim.Adam(target_model.parameters(), lr=lr)
    target_engine, _, _, _ = deepspeed.initialize(config=config_dict, model=target_model, optimizer=ds_optimizer)
    train_batch_size = config_dict["train_micro_batch_size_per_gpu"]

    xs = [torch.randn(train_batch_size, hidden_dim, device=device, dtype=torch.float32) for _ in range(iteration)]
    ys = [torch.randn_like(x) for x in xs]

    for i, (x, y) in enumerate(zip(xs, ys)):
        step_amp(enable, baseline_model, baseline_optimizer, target_engine, dtype, enable_autocast_outside,
                 baseline_scaler, i, x, y, expect_match)

    for module in target_engine.modules():
        for p in module.parameters(recurse=False):
            if module.__class__ in lower_precision_safe_modules and autocast_conf["enabled"]:
                assert hasattr(
                    p, PARAM_COMM_DTYPE_ATTR_NAME
                ), f"A module is in the lower precision safe list, but param does not have autocast_dtype: {module.__class__.__name__}"
                assert get_comm_dtype(
                    p
                ) == dtype, f"dtype of a module in the lower precision safe list is not set to {dtype}: {module.__class__.__name__}"
            else:
                assert not hasattr(
                    p, PARAM_COMM_DTYPE_ATTR_NAME
                ), f"A module is not in the lower precision safe list, but param has autocast_dtype: {module.__class__.__name__}"
                assert get_comm_dtype(
                    p
                ) == model_dtype, f"comm dtype doesn't match module dtype though the module is not in lower precision list"
    target_engine.destroy()


@pytest.mark.parametrize("enable", [True])
class TestZeroAutoCast(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("zero_stage", [0, 1, 2, 3])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test(self, enable, zero_stage, dtype):
        lower_precision_safe_modules = [torch.nn.Linear]
        autocast_conf = {"enabled": enable, "dtype": str(dtype)}

        compare_loss(SimpleModel, enable, zero_stage, torch.float32, dtype, autocast_conf, False,
                     lower_precision_safe_modules)

    @pytest.mark.parametrize("zero_stage", [0, 1, 2, 3])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_safe_modules_conf(self, enable, zero_stage, dtype):
        lower_precision_safe_modules = [torch.nn.Linear]
        autocast_conf = {
            "enabled": enable,
            "dtype": str(dtype),
            "lower_precision_safe_modules": [cls_to_qualname(cls) for cls in lower_precision_safe_modules]
        }

        # The model has both lower precision safe and unsafe modules.
        compare_loss(SimpleModelWithLayerNorm, enable, zero_stage, torch.float32, dtype, autocast_conf, False,
                     lower_precision_safe_modules)

    @pytest.mark.parametrize("zero_stage", [0, 1, 2, 3])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_nested_autocast(self, enable, zero_stage, dtype):
        lower_precision_safe_modules = [torch.nn.Linear]
        autocast_conf = {
            "enabled": False,
            "dtype": str(dtype),
        }

        # torch.autocast is disabled in DeepSpeed engine
        compare_loss(SimpleModelWithLayerNorm,
                     enable,
                     zero_stage,
                     torch.float32,
                     dtype,
                     autocast_conf,
                     True,
                     lower_precision_safe_modules,
                     expect_match=False)

    @pytest.mark.parametrize("zero_stage", [0, 1, 2, 3])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_lower_precision_model(self, enable, zero_stage, dtype):
        lower_precision_safe_modules = [torch.nn.Linear]
        autocast_conf = {
            "enabled": enable,
            "dtype": str(dtype),
        }

        # Use the same dtype for model as autocast dtype
        compare_loss(SimpleModelWithLayerNorm, enable, zero_stage, dtype, dtype, autocast_conf, True,
                     lower_precision_safe_modules, False)
