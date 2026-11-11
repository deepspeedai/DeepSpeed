# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

import pytest
import torch
import torch.nn.functional as F

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import QuantizerBuilder

from unit.common import DistributedTest
from unit.util import torch_assert_close

_QUANTIZER_AVAILABLE = deepspeed.ops.__compatible_ops__[QuantizerBuilder.NAME]
_FORWARD_TOL = 5e-2


class _QuantizedLeaf(torch.nn.Module):
    """Two params of different sizes plus one trainable param."""

    def __init__(self, freeze=True):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(1024, 768) * 0.02)
        self.w2 = torch.nn.Parameter(torch.randn(512, 1024) * 0.02)
        self.h = torch.nn.Parameter(torch.randn(32, 512) * 0.02)
        if freeze:
            self.w1.requires_grad_(False)
            self.w2.requires_grad_(False)

    def forward(self, x):
        return F.linear(F.linear(x, self.w1), self.w2) @ self.h.t()


def _reference(w1, w2, h, x):
    return F.linear(F.linear(x, w1), w2) @ h.t()


def _config(world_size, hpz_partition_size=None):
    zero_config = {
        "stage": 3,
        "zero_quantized_nontrainable_weights": True,
        "stage3_prefetch_bucket_size": 1e9,
    }
    if hpz_partition_size is not None:
        zero_config["zero_hpz_partition_size"] = hpz_partition_size
    return {
        "train_batch_size": 8 * world_size,
        "train_micro_batch_size_per_gpu": 8,
        "steps_per_print": 1,
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            },
        },
        "zero_optimization": zero_config,
    }


def _check_quantized_partition(param, full_weight, rank):
    q = param.ds_tensor.data
    scale = param.ds_tensor.ds_quant_scale
    assert q.dtype == torch.int8
    assert (q != 0).any(), "quantized storage is all zeros"

    groups = scale.numel()
    elems = q.numel() // groups
    dequantized = q.view(groups, elems).float() * scale.view(groups, 1)
    start = rank * q.numel()
    expected = full_weight.reshape(-1)[start:start + q.numel()].float().view(groups, elems)
    # One stored scale is one quantization step; allow one step of slack.
    bound = scale.view(groups, 1).expand(groups, elems) * 1.05
    assert torch.all((dequantized - expected).abs() <= bound)


def test_nvme_partition_is_not_quantized():
    from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
    from deepspeed.runtime.zero.utils import should_quantize_nontrainable_partition

    class _Param:
        requires_grad = False

    tensor = torch.empty(1)
    tensor.final_location = OffloadDeviceEnum.nvme
    assert not should_quantize_nontrainable_partition(_Param(), tensor, True)
    tensor.final_location = None
    assert should_quantize_nontrainable_partition(_Param(), tensor, True)


class TestZeroQuantizedNontrainableWeights(DistributedTest):
    world_size = [1, 2]

    def test_mixed_storage_status_is_rejected(self):
        if not _QUANTIZER_AVAILABLE:
            pytest.skip("quantizer op is not available")
        if not get_accelerator().is_fp16_supported():
            pytest.skip("requires fp16 support")
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size < 2:
            return
        device = get_accelerator().current_device_name()
        model = _QuantizedLeaf().half().to(device)

        engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=_config(world_size))
        try:
            with pytest.raises(RuntimeError, match="quantized coalesced bucket"):
                model.w1.all_gather_coalesced([model.w1, model.h], quantize=True)
        finally:
            engine.destroy()

    def test_default_gather_dequantizes_quantized_params(self):
        if not _QUANTIZER_AVAILABLE:
            pytest.skip("quantizer op is not available")
        if not get_accelerator().is_fp16_supported():
            pytest.skip("requires fp16 support")
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size < 2:
            return
        device = get_accelerator().current_device_name()
        torch.manual_seed(0)
        model = _QuantizedLeaf().half().to(device)
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}

        engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=_config(world_size))
        try:
            # No quantize flag: an all-quantized bucket must select the quantized
            # gather path itself rather than returning the int8 codes as weights.
            handle = model.w1.all_gather_coalesced([model.w1, model.w2])
            handle.wait()
            for name in ("w1", "w2"):
                gathered = getattr(model, name)
                assert gathered.dtype == torch.float16
                assert (gathered.data - initial[name]).abs().max() < 1e-2
            model.w1.partition()
            model.w2.partition()
        finally:
            engine.destroy()

    def test_quantize_nontrainable_params_after_init(self):
        if not _QUANTIZER_AVAILABLE:
            pytest.skip("quantizer op is not available")
        if not get_accelerator().is_fp16_supported():
            pytest.skip("requires fp16 support")

        world_size = int(os.environ["WORLD_SIZE"])
        rank = dist.get_rank()
        device = get_accelerator().current_device_name()
        torch.manual_seed(0)
        model = _QuantizedLeaf(freeze=False).half().to(device)
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}
        inputs = torch.randn(8, 768, device=device).half()

        engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=_config(world_size))
        try:
            for name in ("w1", "w2"):
                assert not hasattr(getattr(model, name).ds_tensor, "ds_quant_scale")

            # The documented LoRA flow freezes after init and re-quantizes.
            model.w1.requires_grad_(False)
            model.w2.requires_grad_(False)
            engine.optimizer.quantize_nontrainable_params()

            for name in ("w1", "w2"):
                _check_quantized_partition(getattr(model, name), initial[name], rank)

            out = engine(inputs)
            ref_out = _reference(initial["w1"], initial["w2"], initial["h"], inputs)
            torch_assert_close(out, ref_out, rtol=1e-2, atol=_FORWARD_TOL)
        finally:
            engine.destroy()

    def test_quantize_nontrainable_params_skips_nvme(self):
        from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum

        if not _QUANTIZER_AVAILABLE:
            pytest.skip("quantizer op is not available")
        if not get_accelerator().is_fp16_supported():
            pytest.skip("requires fp16 support")

        world_size = int(os.environ["WORLD_SIZE"])
        device = get_accelerator().current_device_name()
        torch.manual_seed(0)
        model = _QuantizedLeaf(freeze=False).half().to(device)
        engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=_config(world_size))
        try:
            model.w1.requires_grad_(False)
            model.w2.requires_grad_(False)
            original_location = model.w1.ds_tensor.final_location
            model.w1.ds_tensor.final_location = OffloadDeviceEnum.nvme
            try:
                engine.optimizer.quantize_nontrainable_params()
            finally:
                model.w1.ds_tensor.final_location = original_location

            assert not hasattr(model.w1.ds_tensor, "ds_quant_scale")
            assert hasattr(model.w2.ds_tensor, "ds_quant_scale")
        finally:
            engine.destroy()

    def test_quantize_nontrainable_params_after_init_with_hpz(self):
        if not _QUANTIZER_AVAILABLE:
            pytest.skip("quantizer op is not available")
        if not get_accelerator().is_fp16_supported():
            pytest.skip("requires fp16 support")
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size < 2:
            return

        rank = dist.get_rank()
        device = get_accelerator().current_device_name()
        torch.manual_seed(0)
        model = _QuantizedLeaf(freeze=False).half().to(device)
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}
        inputs = torch.randn(8, 768, device=device).half()

        engine, *_ = deepspeed.initialize(model=model,
                                          model_parameters=model.parameters(),
                                          config=_config(world_size, hpz_partition_size=world_size))
        try:
            model.w1.requires_grad_(False)
            model.w2.requires_grad_(False)
            engine.optimizer.quantize_nontrainable_params()

            for name in ("w1", "w2"):
                _check_quantized_partition(getattr(model, name), initial[name], rank)
                assert hasattr(getattr(model, name).ds_secondary_tensor, "ds_quant_scale")

            out = engine(inputs)
            ref_out = _reference(initial["w1"], initial["w2"], initial["h"], inputs)
            torch_assert_close(out, ref_out, rtol=1e-2, atol=_FORWARD_TOL)
        finally:
            engine.destroy()

    def test_storage_forward_and_repartition(self):
        if not _QUANTIZER_AVAILABLE:
            pytest.skip("quantizer op is not available")
        if not get_accelerator().is_fp16_supported():
            pytest.skip("requires fp16 support")

        world_size = int(os.environ["WORLD_SIZE"])
        rank = dist.get_rank()
        device = get_accelerator().current_device_name()
        torch.manual_seed(0)
        model = _QuantizedLeaf().half().to(device)
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}
        inputs = torch.randn(8, 768, device=device).half()
        label = torch.randn(8, 32, device=device).half()

        engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=_config(world_size))
        try:
            for name in ("w1", "w2"):
                _check_quantized_partition(getattr(model, name), initial[name], rank)

            # Load while NOT_AVAILABLE to exercise fetch -> load -> repartition.
            torch.manual_seed(42)
            new_state = {
                "w1": (torch.randn(1024, 768, device=device) * 0.02).half(),
                "w2": (torch.randn(512, 1024, device=device) * 0.02).half(),
                "h": (torch.randn(32, 512, device=device) * 0.02).half(),
            }
            engine.load_module_state_dict({"module": new_state}, fetch_z3_params=True)
            for name in ("w1", "w2"):
                _check_quantized_partition(getattr(model, name), new_state[name], rank)

            ref_out = _reference(new_state["w1"], new_state["w2"], new_state["h"], inputs)
            out = engine(inputs)
            torch_assert_close(out, ref_out, rtol=1e-2, atol=_FORWARD_TOL)

            loss = F.mse_loss(out, label)
            engine.backward(loss)

            # A fetch after backward goes through the coalesced bucket path.
            out_after_backward = engine(inputs)
            torch_assert_close(out_after_backward, ref_out, rtol=1e-2, atol=_FORWARD_TOL)

            h_before = model.h.detach().clone()
            engine.step()
            out_after_step = engine(inputs)
            updated_h = model.h.detach().clone()
            assert not torch.equal(updated_h, h_before)
            torch_assert_close(out_after_step,
                               _reference(new_state["w1"], new_state["w2"], updated_h, inputs),
                               rtol=1e-2,
                               atol=_FORWARD_TOL)
        finally:
            engine.destroy()
