# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import time
import pytest
import torch
import deepspeed
from transformers import pipeline
from unit.common import DistributedTest
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("This op had not been implemented on this system.", allow_module_level=True)


@pytest.mark.inference
@pytest.mark.parametrize("use_cuda_events", [True, False])
@pytest.mark.parametrize("enable_cuda_graph", [True, False])
class TestModelProfiling(DistributedTest):
    world_size = 1

    def test(self, enable_cuda_graph, use_cuda_events):
        task = "fill-mask"
        model = "bert-base-cased"
        dtype = torch.float16
        query = "I am a [MASK] model"

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))

        pipe = pipeline(task, model, framework="pt", device=get_accelerator().device_name(local_rank))
        pipe.model = deepspeed.init_inference(pipe.model,
                                              dtype=dtype,
                                              mp_size=world_size,
                                              replace_with_kernel_inject=True,
                                              enable_cuda_graph=enable_cuda_graph)
        pipe.model.profile_model_time(use_cuda_events=use_cuda_events)

        e2e_times = []
        model_times = []
        for _ in range(10):
            get_accelerator().synchronize()
            start = time.perf_counter_ns()

            r = pipe(query)

            get_accelerator().synchronize()
            end = time.perf_counter_ns()

            e2e_times.append((end - start) / 1e6)  # convert ns to ms
            model_times.extend(pipe.model.model_times())

        for e2e_t, model_t in zip(e2e_times, model_times):
            assert e2e_t >= model_t
