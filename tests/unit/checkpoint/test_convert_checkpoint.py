# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import shutil
import subprocess
import sys

import torch
import torch.nn as nn
import pytest

import deepspeed
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

    def test_convert_zero_checkpoint_to_fp32_state_dict(self, tmpdir):
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
        ds_save_dir = tmpdir / "checkpoint_ds"
        deepspeed_engine.save_checkpoint(ds_save_dir, tag="checkpoint")
        assert (ds_save_dir / "zero_to_fp32.py").exists()
        assert (ds_save_dir / "zero_to_torch.py").exists()

        model = ModelWithSharedWeights()

        # save checkpoint
        fp32_save_dir = tmpdir / "checkpoint_fp32"
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
        bf16_save_dir = tmpdir / "checkpoint_bf16"
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
