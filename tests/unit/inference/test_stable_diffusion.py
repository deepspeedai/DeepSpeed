# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
import pytest
import deepspeed
import numpy
from unit.common import DistributedTest
from deepspeed.accelerator import get_accelerator


# Setup for these models is different from other pipelines, so we add a separate test
@pytest.mark.stable_diffusion
class TestStableDiffusion(DistributedTest):
    world_size = 1

    def test(self):
        from diffusers import DiffusionPipeline
        from image_similarity_measures.quality_metrics import rmse
        dev = get_accelerator().device_name()
        generator = torch.Generator(device=dev)
        seed = 0xABEDABE7
        generator.manual_seed(seed)
        prompt = "a dog on a rocket"
        model = "prompthero/midjourney-v4-diffusion"
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = torch.device(f"{dev}:{local_rank}")
        pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.half)
        pipe = pipe.to(device)
        baseline_image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]

        pipe = deepspeed.init_inference(
            pipe,
            mp_size=1,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            enable_cuda_graph=True,
        )
        generator.manual_seed(seed)
        deepspeed_image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]

        rmse_value = rmse(org_img=numpy.asarray(baseline_image), pred_img=numpy.asarray(deepspeed_image))

        # RMSE threshold value is arbitrary, may need to adjust as needed
        assert rmse_value <= 0.01


class _RecordingUNet(torch.nn.Module):
    """Records what it is called with, in UNet2DConditionModel.forward's parameter order."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.in_channels = 4
        self.config = {}
        self.seen = None

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    def forward(self,
                sample,
                timestep,
                encoder_hidden_states,
                class_labels=None,
                timestep_cond=None,
                attention_mask=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=True):
        self.seen = {
            "class_labels": class_labels,
            "timestep_cond": timestep_cond,
            "cross_attention_kwargs": cross_attention_kwargs,
            "added_cond_kwargs": added_cond_kwargs,
            "return_dict": return_dict,
        }
        return sample


def test_ds_unet_forwards_the_inputs_it_accepts():
    # DSUNet._forward accepts timestep_cond and added_cond_kwargs, and passed neither on;
    # return_dict went in positionally, where UNet2DConditionModel.forward reads class_labels.
    from deepspeed.model_implementations.diffusers.unet import DSUNet

    unet = _RecordingUNet()
    ds_unet = DSUNet(unet, enable_cuda_graph=False)

    timestep_cond = torch.ones(1, 1)
    added_cond_kwargs = {"text_embeds": torch.zeros(1, 2)}
    ds_unet(torch.zeros(1, 2),
            0,
            torch.zeros(1, 2),
            return_dict=False,
            timestep_cond=timestep_cond,
            cross_attention_kwargs={"scale": 0.5},
            added_cond_kwargs=added_cond_kwargs)

    seen = unet.seen
    assert seen["return_dict"] is False
    assert seen["class_labels"] is None
    assert seen["timestep_cond"] is timestep_cond
    assert seen["added_cond_kwargs"] is added_cond_kwargs
    assert seen["cross_attention_kwargs"] == {"scale": 0.5}
