# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Step 0 (v2): attribute the elementwise soup via an EAGER decode loop.

Graph replay hides per-op scopes, so we replicate the graph path's decode
step eagerly (same injected kernels) and profile that."""

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StaticCache

import deepspeed.module_inject.kernel_reference as kref
import transformers.models.qwen3_5.modeling_qwen3_5 as m5
from deepspeed.module_inject.segment_ki import (GDNInputProj, apply_segment_ki, install_decode_attention,
                                                install_fused_norm)
from deepspeed.ops.module_inject import get_fused_glu_op
from deepspeed.utils.static_cache import DSStaticGDNSlot, DeepSpeedStaticCache

MODEL = "Qwen/Qwen3.5-4B-Base"
BS, N, dev = 8, 64, "cuda"


def scoped(fn, name):

    def wrapper(*a, **k):
        with record_function(name):
            return fn(*a, **k)

    return wrapper


m5.apply_rotary_pos_emb = scoped(m5.apply_rotary_pos_emb, "ZZ_rope")
GDNInputProj.apply = staticmethod(scoped(GDNInputProj.apply, "ZZ_gdn_proj"))
kref.dual_gemv_silu_mul = scoped(kref.dual_gemv_silu_mul, "ZZ_mlp_act")

tok = AutoTokenizer.from_pretrained(MODEL)
object.__setattr__(tok, "eos_token_id", -1)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
para = ("The Eiffel Tower is located in Paris, France. It was built in 1889 "
        "and stands 330 meters tall. The Louvre museum houses the Mona Lisa. ")
ids = tok(para * 120, return_tensors="pt").input_ids[:, :2048].to(dev)
prompt_ids = ids.repeat(BS, 1)
prompt_len = prompt_ids.shape[1]
max_len = prompt_len + N + 8
dtype = next(m.parameters()).dtype

# inject
op = get_fused_glu_op()
apply_segment_ki(m)

# prefill with HF StaticCache, copy into DeepSpeedStaticCache (mirrors rollout)
prefill_cache = StaticCache(config=m.config, batch_size=BS, max_cache_len=max_len, device=dev, dtype=dtype)
attn = torch.ones_like(prompt_ids)
with torch.no_grad():
    prefill_out = m(prompt_ids,
                    attention_mask={"full_attention": None, "linear_attention": None},
                    past_key_values=prefill_cache,
                    use_cache=True)
next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

write_pos = torch.full((BS, ), prompt_len, dtype=torch.long, device=dev)
ds_cache = DeepSpeedStaticCache(m.config, batch_size=BS, max_cache_len=max_len, device=dev, dtype=dtype)
ds_cache.set_write_position(write_pos)
for li in range(len(ds_cache.layers)):
    ds_layer = ds_cache.layers[li]
    hf_layer = prefill_cache.layers[li]
    if isinstance(ds_layer, DSStaticGDNSlot):
        ds_layer.bind(hf_layer)
        continue
    if not ds_layer.is_initialized:
        ds_layer.lazy_initialization(hf_layer.keys, hf_layer.values)
    ds_layer.keys[:, :, :prompt_len, :].copy_(hf_layer.keys[:, :, :prompt_len, :])
    ds_layer.values[:, :, :prompt_len, :].copy_(hf_layer.values[:, :, :prompt_len, :])

install_decode_attention(m, write_pos, op)
install_fused_norm(m, op)
for name, mod in m.named_modules():
    base = name.split(".")[-1]
    if base == "norm" and "linear_attn" in name:
        mod.forward = scoped(mod.forward, "ZZ_gdn_norm")
    if base in ("q_norm", "k_norm"):
        mod.forward = scoped(mod.forward, "ZZ_qk_norm")

static_attn = torch.zeros(BS, 1, 1, max_len, dtype=torch.bool, device=dev)
static_attn[:, :, :, :prompt_len] = attn.unsqueeze(1).unsqueeze(1).bool()
static_attn[:, :, :, prompt_len] = True
static_token = next_token.clone()


@torch.no_grad()
def decode_step(pos):
    write_pos.fill_(pos)
    static_attn[:, :, :, pos] = True
    out = m(static_token,
            attention_mask={"full_attention": static_attn, "linear_attention": None},
            past_key_values=ds_cache,
            use_cache=True)
    static_token.copy_(out.logits[:, -1, :].argmax(dim=-1, keepdim=True))


for step in range(4):  # warmup (jit kernels)
    decode_step(prompt_len + step)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for step in range(N):
        decode_step(prompt_len + step)
    torch.cuda.synchronize()

rows = [(e.key, e.device_time_total / 1000 / N, e.count / N) for e in prof.key_averages()
        if e.key.startswith("ZZ_") and e.device_time_total > 0]
rows.sort(key=lambda r: -r[1])
print("ATTR2 ms/step  calls/step  scope")
for k, v, c in rows:
    print(f"ATTR2 {v:7.3f} {c:9.1f}  {k}")
tot = sum(e.device_time_total for e in prof.key_averages() if e.device_time_total > 0)
print(f"ATTR2 TOTAL {tot / 1000 / N:.2f} ms/step (eager, launch overhead incl.)")
print("ATTR2-DONE")
