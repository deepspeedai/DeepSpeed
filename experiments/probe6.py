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

i = 5
cont = engine._inference_containers[i].module
ref_layer = ref.model.layers[i]
grab = {}
orig_fwd = cont.forward


def spy(*a, **k):
    grab['in'] = a[0].detach().clone()
    grab['kw'] = {n: (v.detach().clone() if torch.is_tensor(v) else v) for n, v in k.items()}
    out = orig_fwd(*a, **k)
    grab['out'] = out[0].detach().clone() if isinstance(out, tuple) else out.detach().clone()
    return out


cont.forward = spy

ids = tok('The capital of France is', return_tensors='pt').input_ids.to(get_accelerator().device_name())
with torch.no_grad():
    engine.module.generate(ids,
                           attention_mask=torch.ones_like(ids),
                           max_new_tokens=2,
                           do_sample=False,
                           pad_token_id=tok.pad_token_id,
                           use_cache=True)
    hs = grab['in']
    kw = grab['kw']
    pos_ids = kw.get('position_ids')
    pe = None
    # rebuild position embeddings like the model does
    pe = ref.model.rotary_emb(hs, pos_ids)
    out_ref = ref_layer(hs.clone(),
                        attention_mask=None,
                        position_ids=pos_ids,
                        past_key_values=None,
                        use_cache=False,
                        position_embeddings=pe)
    if isinstance(out_ref, tuple): out_ref = out_ref[0]
d = (out_ref.float() - grab['out'].float()).abs()
print('LAYER5 max_diff', d.max().item(), 'mean', d.mean().item(), flush=True)
print('ref[0,:2,:4]', out_ref[0, :2, :4].float().tolist(), flush=True)
print('ds [0,:2,:4]', grab['out'][0, :2, :4].float().tolist(), flush=True)
