# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import pytest
import torch
import deepspeed
from deepspeed.model_implementations import DeepSpeedTransformerInference
from unit.common import DistributedTest, DistributedFixture
from transformers import AutoConfig, AutoModelForCausalLM
import deepspeed.comm as dist
from huggingface_hub import snapshot_download
from transformers.utils import is_offline_mode
from deepspeed.ops.op_builder import InferenceBuilder

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("This op had not been implemented on this system.", allow_module_level=True)


def check_dtype(model, expected_dtype):

    def find_dtype(module):
        for child in module.children():
            if isinstance(child, DeepSpeedTransformerInference):
                return child.attention.attn_qkvw.dtype
            else:
                found_dtype = find_dtype(child)
                if found_dtype:
                    return found_dtype

    found_dtype = find_dtype(model)
    assert found_dtype, "Did not find DeepSpeedTransformerInference in model"
    assert (found_dtype == expected_dtype), f"Expected transformer dtype {expected_dtype}, but found {found_dtype}"


@pytest.fixture(params=[
    "bigscience/bloom-560m", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neo-125M", "facebook/opt-350m", "facebook/opt-125m"
])
def model_name(request):
    return request.param


@pytest.fixture(params=[torch.float16, torch.int8], ids=["fp16", "int8"])
def dtype(request):
    return request.param


class save_shard(DistributedFixture):
    world_size = 2

    def run(self, model_name, class_tmpdir):
        # Only write a checkpoint if one does not exist
        if not os.path.isdir(os.path.join(class_tmpdir, model_name)):
            world_size = int(os.getenv("WORLD_SIZE", "1"))
            inf_config = {
                "replace_with_kernel_inject": True,
                "dtype": torch.float16,
                "enable_cuda_graph": False,
                "tensor_parallel": {
                    "tp_size": world_size
                },
                "save_mp_checkpoint_path": os.path.join(str(class_tmpdir), model_name),
            }

            # Load model and save sharded checkpoint
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
            model = deepspeed.init_inference(model, config=inf_config)


@pytest.mark.seq_inference
class TestCheckpointShard(DistributedTest):
    world_size = 2

    def test(self, model_name, dtype, class_tmpdir, save_shard):
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        inf_config = {
            "replace_with_kernel_inject": True,
            "dtype": dtype,
            "enable_cuda_graph": False,
            "tensor_parallel": {
                "tp_size": world_size
            },
            "checkpoint": os.path.join(class_tmpdir, model_name, "ds_inference_config.json"),
        }

        # Load model on meta tensors
        model_config = AutoConfig.from_pretrained(model_name)
        # Note that we use half precision to load initially, even for int8
        with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
            model = AutoModelForCausalLM.from_config(model_config, torch_dtype=torch.bfloat16)
        model = model.eval()
        model = deepspeed.init_inference(model, config=inf_config)
        check_dtype(model, dtype)


@pytest.mark.seq_inference
class TestCheckpointShardinAutoTP(DistributedTest):
    world_size = 2

    def test(self, model_name, class_tmpdir):

        def write_checkpoints_json(model_name, class_tmpdir):
            import json
            from pathlib import Path
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            if local_rank == 0:
                # download only on first process
                cached_repo_dir = snapshot_download(
                    model_name,
                    local_files_only=is_offline_mode(),
                    cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                    ignore_patterns=["*.safetensors", "*.msgpack", "*.h5"],
                )
                file_list = [str(entry) for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]") if entry.is_file()]
                data = {"type": "ds_model", "checkpoints": file_list, "version": 1.0}
                os.makedirs(os.path.join(class_tmpdir, model_name), exist_ok=True)
                json.dump(data, open(os.path.join(class_tmpdir, model_name, "ds_inference_config.json"), "w"))
            dist.barrier()

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        inf_config = {
            "replace_with_kernel_inject": False,
            "tensor_parallel": {
                "tp_size": world_size
            },
            "checkpoint": os.path.join(class_tmpdir, model_name, "ds_inference_config.json"),
        }

        write_checkpoints_json(model_name, class_tmpdir)

        # Load model on meta tensors
        model_config = AutoConfig.from_pretrained(model_name)
        # Note that we use half precision to load initially, even for int8
        with deepspeed.OnDevice(dtype=torch.bfloat16, device="meta"):
            model = AutoModelForCausalLM.from_config(model_config, torch_dtype=torch.bfloat16)
        model = model.eval()
        model = deepspeed.init_inference(model, config=inf_config)
