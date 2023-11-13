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
        generator = torch.Generator(device=get_accelerator().current_device())
        seed = 0xABEDABE7
        generator.manual_seed(seed)
        prompt = "a dog on a rocket"
        model = "prompthero/midjourney-v4-diffusion"
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")

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
