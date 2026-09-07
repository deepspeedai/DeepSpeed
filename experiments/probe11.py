# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Diagnostic probe for the old-KI numerics bisection (see qwen-he-kernel-inject-proto.md)."""

import torch
from deepspeed.accelerator import get_accelerator  # noqa: E402
import torch.nn.functional as F
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
mlp = cont.mlp
grab = {}
orig_layer = cont.forward
orig_att = att.forward


def spy_layer(*a, **k):
    out = orig_layer(*a, **k)
    if 'lin' not in grab:
        grab['lin'] = a[0].detach().clone()
        grab['lout'] = out[0].detach().clone() if isinstance(out, tuple) else out.detach().clone()
    return out


def spy_att(*a, **k):
    out = orig_att(*a, **k)
    if 'att_out' not in grab:
        grab['att_out'] = out[0].detach().clone()
    return out


cont.forward = spy_layer
att.forward = spy_att
ids = tok('The capital of France is', return_tensors='pt').input_ids.to(get_accelerator().device_name())
with torch.no_grad():
    engine.module.generate(ids,
                           attention_mask=torch.ones_like(ids),
                           max_new_tokens=2,
                           do_sample=False,
                           pad_token_id=tok.pad_token_id,
                           use_cache=True)

    hf_layer = model.model.layers[5]
    for name, w in [('cont.norm_w', cont.norm_w), ('mlp.inter_w', mlp.inter_w), ('mlp.output_w', mlp.output_w)]:
        pass
    # identify norm weight roles
    print('norm_w vs input_ln',
          (cont.norm_w.detach().float() - hf_layer.input_layernorm.weight.float()).abs().max().item(),
          flush=True)
    print('norm_w vs post_ln ',
          (cont.norm_w.detach().float() - hf_layer.post_attention_layernorm.weight.float()).abs().max().item(),
          flush=True)

    # manual tail: residual1 -> post norm -> gated mlp -> residual2
    res1 = grab['lin'].float() + grab['att_out'].float()
    pn = res1 * torch.rsqrt(res1.pow(2).mean(-1, keepdim=True) +
                            1e-6) * hf_layer.post_attention_layernorm.weight.float()
    g = F.silu(pn @ hf_layer.mlp.gate_proj.weight.float().t())
    u = pn @ hf_layer.mlp.up_proj.weight.float().t()
    out_m = res1 + (g * u) @ hf_layer.mlp.down_proj.weight.float().t()

    lo = grab['lout'].float()
    corr = torch.corrcoef(torch.stack([out_m.flatten(), lo.flatten()]))[0, 1].item()
    print('LAYER max', (out_m - lo).abs().max().item(),
          'mean', (out_m - lo).abs().mean().item(),
          'corr',
          corr,
          'outmax',
          lo.abs().max().item(),
          flush=True)
