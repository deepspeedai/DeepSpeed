# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Diagnostic probe for the old-KI numerics bisection (see qwen-he-kernel-inject-proto.md)."""

import torch
from deepspeed.accelerator import get_accelerator  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct',
                                             dtype=torch.bfloat16).to(get_accelerator().device_name()).eval()
ref = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct',
                                           dtype=torch.bfloat16).to(get_accelerator().device_name()).eval()
deepspeed.init_distributed(dist_backend='nccl')
engine, _, _, _ = deepspeed.initialize(model=model,
                                       config={
                                           'train_micro_batch_size_per_gpu': 1,
                                           'bf16': {
                                               'enabled': True
                                           },
                                           'hybrid_engine': {
                                               'enabled': True,
                                               'inference_tp_size': 1,
                                               'max_out_tokens': 64
                                           }
                                       })
engine.eval()

ids = tok('The capital of France is', return_tensors='pt').input_ids.to(get_accelerator().device_name())
with torch.no_grad():
    e_ref = ref.model.embed_tokens(ids)
    e_w = engine.module.model.embed_tokens(ids)
    print('EMBED diff', (e_ref.float() - e_w.float()).abs().max().item(), flush=True)

    x = torch.randn(1, 5, 896, device=get_accelerator().device_name(), dtype=torch.bfloat16)
    l_ref = ref.lm_head(x)
    l_w = engine.module.lm_head(x)
    print('LMHEAD diff', (l_ref.float() - l_w.float()).abs().max().item(), flush=True)
    print('LMHEAD shape', tuple(l_ref.shape), tuple(l_w.shape), flush=True)

    # full forward logits comparison (single shot, no cache)
    o_ref = ref(ids).logits
    o_ds = engine.module(ids).logits
    d = (o_ref.float() - o_ds.float()).abs()
    print('FULLLOGITS max',
          d.max().item(),
          'mean',
          d.mean().item(),
          'refmax',
          o_ref.float().abs().max().item(),
          flush=True)
    print('argmax agree', (o_ref.argmax(-1) == o_ds.argmax(-1)).float().mean().item(), flush=True)
