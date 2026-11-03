# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""ZeRO-3 trainer -> TP=2 segKI rollout end-to-end (arena-sync contract).

Two real AdamW steps on a ZeRO-3 engine, weights gathered and written IN
PLACE into the rollout's sharded storage (arena-sync semantics), segki_
refresh, generate — must be BITWISE equal to a fresh injection over the
same weights. Run: DS_RUN_ZERO3_E2E=1 torchrun --nproc_per_node=2 \
    tests/integration/zero3_arena_e2e.py
Opt-in (needs 2 GPUs + the Qwen3.5 test model cached).
"""
import os
import sys
if __name__ == "__main__" and os.environ.get("DS_RUN_ZERO3_E2E") != "1":
    print("set DS_RUN_ZERO3_E2E=1 to run this integration test", file=sys.stderr)
    sys.exit(0)

import sys, os, torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29710")
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/pr8602-wt")
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
import deepspeed
from deepspeed.module_inject.segment_ki import segki_refresh

MODEL = "Qwen/Qwen3.5-0.8B-Base"
tok = AutoTokenizer.from_pretrained(MODEL)
object.__setattr__(tok, "eos_token_id", -1)
rank = int(os.environ.get("RANK", "0"))
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))  #ignore-cuda
DEV = f"cuda:{torch.cuda.current_device()}"  #ignore-cuda

Z3 = {
    "train_batch_size": 2,
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5
        }
    },
}
TPSPECS = [
    {
        "patterns": [r".*\.linear_attn\.in_proj_z"],
        "partition_type": "column"
    },
    {
        "patterns": [r".*\.linear_attn\.in_proj_b"],
        "partition_type": "column"
    },
    {
        "patterns": [r".*\.linear_attn\.in_proj_a"],
        "partition_type": "column"
    },
    {
        "patterns": [r".*\.linear_attn\.out_proj"],
        "partition_type": "row"
    },
    {
        "patterns": [r".*\.self_attn\.q_proj"],
        "partition_type": "column"
    },
    {
        "patterns": [r".*\.self_attn\.k_proj"],
        "partition_type": "column"
    },
    {
        "patterns": [r".*\.self_attn\.v_proj"],
        "partition_type": "column"
    },
    {
        "patterns": [r".*\.self_attn\.o_proj"],
        "partition_type": "row"
    },
    {
        "patterns": [r".*\.mlp\.gate_proj"],
        "partition_type": "column"
    },
    {
        "patterns": [r".*\.mlp\.up_proj"],
        "partition_type": "column"
    },
    {
        "patterns": [r".*\.mlp\.down_proj"],
        "partition_type": "row"
    },
]
TP = {
    "train_batch_size": 2,
    "tensor_parallel": {
        "autotp_size": 2,
        "partition_config": {
            "use_default_specs": False,
            "strict_mode": False,
            "layer_specs": TPSPECS
        }
    },
    "bf16": {
        "enabled": True
    }
}


class Shim:

    def __init__(self, m):
        self.module = m
        self._cfg = m.config


def build_tp_rollout():
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    eng, _, _, _ = deepspeed.initialize(model=m, config=TP)
    shim = Shim(eng)
    if isinstance(getattr(shim.module, "config", None), dict):
        shim.module.config = m.config
    from deepspeed.comm import custom_ar
    custom_ar.initialize()
    ro = HybridEngineRollout(shim, tok, HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))
    return ro, eng


def gather_z3_weights(engine):
    """Pull full bf16 weights out of the ZeRO-3 engine (both ranks agree)."""
    from deepspeed.runtime.zero import GatheredParameters
    params = list(engine.module.parameters())
    with GatheredParameters(params, modifier_rank=0):
        out = {n: p.detach().to(torch.bfloat16).clone() for n, p in engine.module.named_parameters()}
    return out


def train_a_step(engine, seed):
    torch.manual_seed(seed)
    ids = torch.randint(1000, 50000, (2, 128), device=DEV)
    loss = engine(ids, labels=ids).loss
    engine.backward(loss)
    engine.step()
    engine.zero_grad()
    return loss.item()


torch.manual_seed(11)
prompt = torch.randint(1000, 50000, (4, 128), device=DEV)


def gen(ro):
    out = ro.generate(RolloutRequest(prompt_ids=prompt, prompt_attention_mask=torch.ones_like(prompt)),
                      SamplingConfig(max_new_tokens=12, temperature=0.0))
    return out.input_ids[:, prompt.shape[1]:].clone()


# --- phase 1: ZeRO-3 trainer, one REAL step on partitioned weights
mt = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
trainer, _, _, _ = deepspeed.initialize(model=mt, config=Z3, model_parameters=list(mt.parameters()))
l0 = train_a_step(trainer, 101)
print(f"[r{rank}] zero3 step1 loss={l0:.4f}", flush=True)
w1 = gather_z3_weights(trainer)  # weights AFTER step 1


def write_full_into_sharded(eng, full_sd, rank, world):
    """Arena-sync semantics: write FULL weights into the TP-sharded
    rollout storage in place (split dim inferred from shape mismatch —
    the same contract their storage-map reshard implements)."""
    written = 0
    with torch.no_grad():
        for n, p in eng.module.named_parameters():
            if n not in full_sd:
                continue
            full = full_sd[n].to(p.device)
            if full.shape == p.shape:
                p.data.copy_(full)
            else:
                d = next(i for i, (a, b) in enumerate(zip(full.shape, p.shape)) if a != b)
                p.data.copy_(full.chunk(world, dim=d)[rank])
            written += 1
    return written


# --- phase 2: rollout engine over post-step-1 weights; caches build
ro_b, eng_b = build_tp_rollout()
n_w = write_full_into_sharded(eng_b, w1, rank, 2)
print(f"[r{rank}] phase2 loaded {n_w} sharded params from w1", flush=True)
gen(ro_b)  # build derived caches on w1
print(f"[r{rank}] rollout caches built", flush=True)

# --- phase 3: trainer takes ANOTHER real step; sync = in-place write
l1 = train_a_step(trainer, 202)
w2 = gather_z3_weights(trainer)
print(f"[r{rank}] zero3 step2 loss={l1:.4f}", flush=True)
n_w2 = write_full_into_sharded(eng_b, w2, rank, 2)  # in-place, same storage
assert n_w2 == n_w, "param sets differ between syncs"
segki_refresh(eng_b.module)
out_refreshed = gen(ro_b)
print(f"[r{rank}] refreshed generate done", flush=True)

# --- phase 4: fresh injection over the SAME post-step-2 weights
del trainer, mt
torch.cuda.empty_cache()  #ignore-cuda
ro_c, eng_c = build_tp_rollout()
write_full_into_sharded(eng_c, w2, rank, 2)
gen(ro_c)  # fresh caches on w2
out_fresh = gen(ro_c)

same = torch.equal(out_refreshed, out_fresh)
diffs = (out_refreshed != out_fresh).sum().item() if not same else 0
print(f"[r{rank}] ZERO3-E2E {'PASS' if same else 'FAIL'} bitwise={same} diffs={diffs}", flush=True)
print(f"[r{rank}] sample refreshed[0,:6]: {out_refreshed[0,:6].tolist()}", flush=True)
print(f"[r{rank}] sample fresh[0,:6]:     {out_fresh[0,:6].tolist()}", flush=True)
assert same, f"refresh diverged from fresh: {diffs} diffs"
print(f"[r{rank}] ZERO3-E2E-DONE", flush=True)
