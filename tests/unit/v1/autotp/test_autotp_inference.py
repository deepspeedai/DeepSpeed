# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from copy import deepcopy
import os

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest


def skip_on_device():
    if get_accelerator().device_name() == 'xpu':
        pytest.skip("XPU requires a higher version for test")


class TestGPTBigCodeMultiQueryInference(DistributedTest):
    """Inference of multi-query GPTBigCode must match the unsharded model under AutoTP.

    The inference injection path builds the same AutoTP instance as training, so it inherits the
    query-head split for the fused c_attn and the replicated key/value block. This is a smoke
    test that the sharded model still produces the unsharded logits.
    """
    world_size = 2
    reuse_dist_env = False

    def test_matches_unsharded_logits(self):
        skip_on_device()
        transformers = pytest.importorskip("transformers")
        world_size = int(os.getenv("WORLD_SIZE", "2"))

        torch.manual_seed(0)
        config = transformers.GPTBigCodeConfig(vocab_size=64,
                                               n_positions=32,
                                               n_embd=32,
                                               n_layer=2,
                                               n_head=4,
                                               multi_query=True)
        model = transformers.GPTBigCodeForCausalLM(config).eval()
        reference = deepcopy(model)

        device = torch.device(get_accelerator().current_device_name())
        reference = reference.to(device)
        input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)
        with torch.no_grad():
            expected = reference(input_ids).logits

        engine = deepspeed.init_inference(model,
                                          tensor_parallel={"tp_size": world_size},
                                          dtype=torch.float32,
                                          replace_with_kernel_inject=False)
        with torch.no_grad():
            actual = engine(input_ids).logits

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
