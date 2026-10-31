# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Step 0: attribute the elementwise soup to ops via record_function scopes."""

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

import deepspeed.module_inject.kernel_reference as kref
import transformers.models.qwen3_5.modeling_qwen3_5 as m5
from deepspeed.module_inject.segment_ki import GDNInputProj
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig

MODEL = "Qwen/Qwen3.5-4B-Base"
BS, N = 8, 96
dev = "cuda"


class Shim:

    def __init__(self, m):
        self.module = m


def scoped(fn, name):

    def wrapper(*a, **k):
        with record_function(name):
            return fn(*a, **k)

    return wrapper


# patch the ops we want attributed
m5.apply_rotary_pos_emb = scoped(m5.apply_rotary_pos_emb, "ZZ_rope")
GDNInputProj.apply = staticmethod(scoped(GDNInputProj.__func__.apply if hasattr(GDNInputProj, "__func__") else
                                         GDNInputProj.apply, "ZZ_gdn_proj"))
kref.dual_gemv_silu_mul = scoped(kref.dual_gemv_silu_mul, "ZZ_mlp_act")

tok = AutoTokenizer.from_pretrained(MODEL)
object.__setattr__(tok, "eos_token_id", -1)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
para = ("The Eiffel Tower is located in Paris, France. It was built in 1889 "
        "and stands 330 meters tall. The Louvre museum houses the Mona Lisa. ")
ids = tok(para * 120, return_tensors="pt").input_ids[:, :2048].to(dev)
req = RolloutRequest(prompt_ids=ids.repeat(BS, 1), prompt_attention_mask=torch.ones_like(ids).repeat(BS, 1))

ro = HybridEngineRollout(Shim(m), tok, HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))
ro.generate(req, SamplingConfig(max_new_tokens=32, temperature=0.0))  # warmup+capture
torch.cuda.synchronize()

# wrap per-module norms after injection (module tree is final now)
for name, mod in m.named_modules():
    if name.endswith("linear_attn.norm") or name.endswith(".linear_attn") and False:
        pass
for name, mod in m.named_modules():
    base = name.split(".")[-1]
    if base == "norm" and "linear_attn" in name:
        mod.forward = scoped(mod.forward, "ZZ_gdn_norm")
    if base in ("q_norm", "k_norm"):
        mod.forward = scoped(mod.forward, "ZZ_qk_norm")

# NOTE: the graph was captured during warmup (before these last wrappers), so
# profile a FRESH generate: capture re-runs every generate call.
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    ro.generate(req, SamplingConfig(max_new_tokens=N, temperature=0.0))
    torch.cuda.synchronize()

rows = []
for e in prof.key_averages():
    if e.key.startswith("ZZ_") and e.device_time_total > 0:
        rows.append((e.key, e.device_time_total / 1000 / (N - 1), e.count / max(1, N - 1)))
rows.sort(key=lambda r: -r[1])
print("ATTR ms/step  calls/step  scope")
for k, v, c in rows:
    print(f"ATTR {v:7.3f} {c:9.1f}  {k}")
print("ATTR-DONE")
