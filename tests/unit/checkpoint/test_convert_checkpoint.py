# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
import shutil
import subprocess
import sys

import torch
import torch.nn as nn
import pytest

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
import deepspeed.utils.zero_to_fp32 as zero_to_fp32
import deepspeed.utils.zero_to_torch as zero_to_torch
from deepspeed.utils.zero_to_fp32 import (convert_zero_checkpoint_to_fp32_state_dict,
                                          convert_zero_checkpoint_to_state_dict, to_torch_tensor)
from deepspeed.utils.zero_to_torch import main as zero_to_torch_main
from unit.common import DistributedTest


def test_output_dtype_conversion_preserves_shared_tensors():
    tensor = torch.arange(16, dtype=torch.float32)
    state_dict = {"weight": tensor, "shared_weight": tensor}

    converted = to_torch_tensor(state_dict, dtype="bf16")
    assert converted["weight"].dtype == torch.bfloat16
    assert id(converted["weight"]) == id(converted["shared_weight"])

    empty = to_torch_tensor(state_dict, return_empty_tensor=True, dtype="fp16")
    assert empty["weight"].dtype == torch.float16
    assert id(empty["weight"]) == id(empty["shared_weight"])

    with pytest.raises(ValueError, match="Unsupported output dtype"):
        to_torch_tensor(state_dict, dtype="int8")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("return_empty_tensor", [False, True])
def test_output_dtype_preserves_nonfloating_buffers(dtype, return_empty_tensor):
    offsets = torch.tensor([2**24 + 1, 2**24 + 3], dtype=torch.int64)
    state_dict = {
        "weight": torch.tensor([0.5, 1.5]),
        "offsets": offsets,
        "shared_offsets": offsets,
        "mask": torch.tensor([True, False]),
        "phase": torch.tensor([1 + 2j, 3 - 4j]),
    }

    converted = to_torch_tensor(state_dict, dtype=dtype, return_empty_tensor=return_empty_tensor)

    assert converted["weight"].dtype == dtype
    assert converted["offsets"] is converted["shared_offsets"]
    for name in ("offsets", "mask", "phase"):
        assert converted[name].dtype == state_dict[name].dtype
        assert converted[name].shape == state_dict[name].shape
        if not return_empty_tensor:
            torch.testing.assert_close(converted[name], state_dict[name], rtol=0, atol=0)
    if not return_empty_tensor:
        torch.testing.assert_close(converted["weight"], state_dict["weight"].to(dtype))


def test_checkpoint_file_output_dtype(monkeypatch, tmp_path):

    def make_state_dict(*args, **kwargs):
        weight = torch.linspace(-1, 1, 4096, dtype=torch.float32)
        return {"weight": weight, "shared_weight": weight}

    monkeypatch.setattr(zero_to_fp32, "get_fp32_state_dict_from_zero_checkpoint", make_state_dict)
    fp32_dir = tmp_path / "fp32"
    bf16_dir = tmp_path / "bf16"

    convert_zero_checkpoint_to_fp32_state_dict("unused", fp32_dir, max_shard_size=None)
    convert_zero_checkpoint_to_state_dict("unused", bf16_dir, dtype="bf16", max_shard_size=None)

    fp32_state_dict = torch.load(fp32_dir / "pytorch_model.bin")
    bf16_state_dict = torch.load(bf16_dir / "pytorch_model.bin")
    assert bf16_state_dict["weight"].dtype == torch.bfloat16
    assert id(bf16_state_dict["weight"]) == id(bf16_state_dict["shared_weight"])
    torch.testing.assert_close(bf16_state_dict["weight"].float(), fp32_state_dict["weight"], rtol=5e-3, atol=5e-3)
    assert (bf16_dir / "pytorch_model.bin").stat().st_size < (fp32_dir / "pytorch_model.bin").stat().st_size * 0.6


def test_zero_to_torch_cli_passes_dtype(monkeypatch, tmp_path):
    call = {}

    def record_conversion(checkpoint_dir, output_dir, **kwargs):
        call.update(checkpoint_dir=checkpoint_dir, output_dir=output_dir, **kwargs)

    monkeypatch.setattr(zero_to_fp32, "convert_zero_checkpoint_to_state_dict", record_conversion)
    zero_to_torch_main(["checkpoint", str(tmp_path), "--dtype", "fp16", "--max_shard_size", "1GB"])

    assert call["checkpoint_dir"] == "checkpoint"
    assert call["output_dir"] == str(tmp_path)
    assert call["dtype"] == "fp16"
    assert call["max_shard_size"] == "1GB"


def test_zero_to_torch_standalone_uses_local_converter(tmp_path):
    script_path = tmp_path / "zero_to_torch.py"
    shutil.copyfile(zero_to_torch.__file__, script_path)
    (tmp_path / "zero_to_fp32.py").write_text(
        "from pathlib import Path\n"
        "OUTPUT_DTYPE_NAMES = {'fp16': None}\n"
        "debug = False\n"
        "def convert_zero_checkpoint_to_state_dict(checkpoint_dir, output_dir, **kwargs):\n"
        "    Path(output_dir).write_text(kwargs['dtype'], encoding='utf-8')\n",
        encoding="utf-8")
    marker_path = tmp_path / "converter.txt"

    subprocess.run([
        sys.executable,
        str(script_path),
        "checkpoint",
        str(marker_path),
        "--dtype",
        "fp16",
    ], check=True)

    assert script_path.read_text(encoding="utf-8").startswith("#!/usr/bin/env python\n")
    assert marker_path.read_text(encoding="utf-8") == "fp16"


class ModelWithSharedWeights(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(100, 100)
        self.layer1 = nn.Linear(200, 200)
        self.layer2 = nn.Linear(300, 300)
        # tie layer 1 and layer 2
        self.layer1.weight = self.layer2.weight


class TestCheckpointConvert(DistributedTest):
    world_size = 2

    def test_convert_zero_checkpoint_to_fp32_state_dict(self, tmp_path):
        config = {
            "train_micro_batch_size_per_gpu": 2,
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": 3
            },
        }
        model = ModelWithSharedWeights()
        optimizer = torch.optim.Adam(model.parameters())

        deepspeed_engine, _, _, _ = deepspeed.initialize(
            config=config,
            model=model,
            optimizer=optimizer,
        )
        ds_save_dir = tmp_path / "checkpoint_ds"
        deepspeed_engine.save_checkpoint(ds_save_dir, tag="checkpoint")
        assert (ds_save_dir / "zero_to_fp32.py").exists()
        assert (ds_save_dir / "zero_to_torch.py").exists()

        model = ModelWithSharedWeights()

        # save checkpoint
        fp32_save_dir = tmp_path / "checkpoint_fp32"
        convert_zero_checkpoint_to_fp32_state_dict(ds_save_dir, fp32_save_dir)

        # load state_dict from fp32 checkpoint
        state_dict = torch.load(fp32_save_dir / 'pytorch_model.bin')

        # check shared tensor
        assert id(state_dict['layer1.weight']) == id(state_dict['layer2.weight'])

        # load state_dict into model
        model.load_state_dict(state_dict, strict=True)

        # Exporting in bfloat16 uses the target dtype for both shard planning
        # and serialization. At 300KB this model fits in one bf16 shard but
        # would be split if shard planning still counted fp32 bytes.
        bf16_save_dir = tmp_path / "checkpoint_bf16"
        convert_zero_checkpoint_to_state_dict(ds_save_dir, bf16_save_dir, dtype="bfloat16", max_shard_size="300KB")
        bf16_state_dict = torch.load(bf16_save_dir / 'pytorch_model.bin')
        assert not (bf16_save_dir / 'pytorch_model.bin.index.json').exists()

        assert id(bf16_state_dict['layer1.weight']) == id(bf16_state_dict['layer2.weight'])
        for name, tensor in bf16_state_dict.items():
            assert tensor.dtype == torch.bfloat16
            torch.testing.assert_close(tensor.float(), state_dict[name], rtol=5e-3, atol=5e-3)

        fp32_size = (fp32_save_dir / 'pytorch_model.bin').stat().st_size
        bf16_size = (bf16_save_dir / 'pytorch_model.bin').stat().st_size
        assert bf16_size < fp32_size * 0.6


class ModelWithBuffers(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(1., 5.))
        self.register_buffer("offsets", torch.tensor([2**24 + 1, 2**24 + 3, 2**24 + 5, 2**24 + 7]))
        self.register_buffer("mask", torch.tensor([True, False, True, False]))
        self.register_buffer("scale", torch.tensor(0.5, dtype=torch.float16))

    def forward(self, inputs):
        indices = (self.offsets % inputs.numel())[self.mask]
        return (inputs[indices] * self.weight[self.mask] * self.scale).sum()


class TestCheckpointConvertBuffers(DistributedTest):
    world_size = [1, 2]

    @pytest.mark.parametrize("zero_stage", [2, 3])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_buffer_round_trip(self, tmp_path, zero_stage, dtype):
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": zero_stage
            },
        }
        model = ModelWithBuffers()
        expected_buffers = {
            name: value.float() if value.is_floating_point() else value.clone()
            for name, value in model.named_buffers()
        }
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
                                               config=config)
        inputs = torch.arange(1., 5., device=get_accelerator().device_name(engine.local_rank))
        engine.backward(engine(inputs))
        engine.step()
        expected_output = engine(inputs).detach().cpu()
        checkpoint_dir = tmp_path / "checkpoint"
        engine.save_checkpoint(checkpoint_dir, tag="step1")

        if dist.get_rank() == 0:
            # load_state_dict casts into the destination buffer dtype, but cannot recover
            # integer precision already lost during checkpoint reconstruction.
            state_dict = zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
            restored = ModelWithBuffers()
            restored.load_state_dict(state_dict)
            torch.testing.assert_close(restored(inputs.cpu()), expected_output)
            for name, expected in expected_buffers.items():
                torch.testing.assert_close(state_dict[name], expected, rtol=0, atol=0)

            output_dir = tmp_path / "converted"
            convert_zero_checkpoint_to_state_dict(checkpoint_dir, output_dir, dtype=dtype, max_shard_size=32)
            index = json.loads((output_dir / "pytorch_model.bin.index.json").read_text())
            exported = {}
            for filename in set(index["weight_map"].values()):
                exported.update(torch.load(output_dir / filename, weights_only=True))

            # The shard plan must count integer/bool buffers at their actual storage size.
            expected_size = sum(value.numel() * value.element_size() for value in exported.values())
            assert index["metadata"]["total_size"] == expected_size
            for name in ("offsets", "mask"):
                torch.testing.assert_close(exported[name], expected_buffers[name], rtol=0, atol=0)
            for name in ("weight", "scale"):
                assert exported[name].dtype == dtype
                torch.testing.assert_close(exported[name], state_dict[name].to(dtype))
            restored.load_state_dict(exported)
            reference = ModelWithBuffers()
            reference.load_state_dict(state_dict)
            reference.to(dtype).float()
            torch.testing.assert_close(restored(inputs.cpu()), reference(inputs.cpu()), rtol=0, atol=0)
        dist.barrier()
        engine.destroy()
