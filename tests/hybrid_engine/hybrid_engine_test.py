# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from transformers import AutoModelForCausalLM
import deepspeed
import argparse
from deepspeed.accelerator import get_accelerator

deepspeed.runtime.utils.see_memory_usage('pre test', force=True)

model = AutoModelForCausalLM.from_pretrained('facebook/opt-350M').half().to(get_accelerator().device_name())
parser = argparse.ArgumentParser()
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

deepspeed.runtime.utils.see_memory_usage('post test', force=True)

m, _, _, _ = deepspeed.initialize(model=model, args=args, enable_hybrid_engine=True)

m.eval()
input = torch.ones(1, 16, device='cuda', dtype=torch.long)
out = m(input)

m.train()
out = m(input)
print(out['logits'], out['logits'].norm())
