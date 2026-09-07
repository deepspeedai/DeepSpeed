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

cont = engine._inference_containers[5].module
att = cont.attention
grab = {}
orig_qkv = att.qkv_func.forward


def spy_qkv(*a, **k):
    out = orig_qkv(*a, **k)
    if 'qkv_in' not in grab:
        xin = a[0] if a else k.get('input')
        grab['qkv_in'] = xin.detach().clone()
        grab['qkv_out'] = out[0].detach().clone()
        grab['norm_out'] = out[1].detach().clone()
    return out


att.qkv_func.forward = spy_qkv
ids = tok('The capital of France is', return_tensors='pt').input_ids.to(get_accelerator().device_name())
with torch.no_grad():
    engine.module.generate(ids,
                           attention_mask=torch.ones_like(ids),
                           max_new_tokens=2,
                           do_sample=False,
                           pad_token_id=tok.pad_token_id,
                           use_cache=True)

    hs = grab['qkv_in'].float()
    norm_ds = grab['norm_out'].float()
    nw = cont.norm_w.detach().float()
    m = hs * torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) + 1e-6) * nw

    def stats(t, name):
        print(
            f'{name} mean {t.mean().item():.4f} std {t.std().item():.4f} min {t.min().item():.4f} max {t.max().item():.4f}',
            flush=True)

    stats(m, 'manual_norm')
    stats(norm_ds, 'ds_norm    ')
    corr = torch.corrcoef(torch.stack([m.flatten(), norm_ds.flatten()]))[0, 1].item()
    print('NORM corr', corr, 'maxdiff', (m - norm_ds).abs().max().item(), flush=True)

    # qkv output check with DS's own norm
    Wq = att.attn_qkvw.detach().float()
    if Wq.shape[0] != 1152: Wq = Wq.t()
    qkv_m = norm_ds @ Wq.t()
    print('QKVOUT diff', (qkv_m - grab['qkv_out'].float()).abs().max().item(), flush=True)
