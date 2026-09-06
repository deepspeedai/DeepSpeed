# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Functional check for the segment-style KI prototype (fused_glu).

Modes (all greedy, golden must match bit-exactly):
  baseline   - single process, plain HF generate (golden reference)
  autotp     - torchrun x2: AutoTP only (known-good control path)
  autotp_ki  - torchrun x2: AutoTP + apply_segment_ki (fused_glu segments)

The acceptance claim for the segment KI: autotp_ki produces the same greedy
output as autotp and baseline, while replacing gate/up GEMM pairs with one
fused GEMM per shard and delegating the down_proj collective untouched.
"""

import argparse
import os
import sys

_ds_src = os.environ.get("DS_SRC")
if _ds_src:
    sys.path.insert(0, _ds_src)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")

import torch  # noqa: E402
from deepspeed.accelerator import get_accelerator  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPTS = ["The capital of France is", "The largest ocean on Earth is"]


def has_accelerator():  # noqa: E731
    # CPU is a first-class accelerator in DeepSpeed (its is_available() is
    # always True by design), so the usable question here is backend identity:
    # which accelerator won decides the comm backend (gloo vs nccl) and the
    # default dtype convention, not whether an accelerator exists.
    return get_accelerator().device_name() != 'cpu'


def load(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    return tok, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "autotp", "autotp_ki"], required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--ref", default=None)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--autotp", type=int, default=2)
    p.add_argument("--model", default=MODEL_NAME)
    args = p.parse_args()

    torch.set_num_threads(8)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tok, model = load(args.model)

    # AutoTP is TP semantics: every rank must run the same prompt.
    enc = tok(PROMPTS[0], return_tensors="pt")
    prompt_ids, attn = enc.input_ids, enc.attention_mask
    prompt_len = prompt_ids.shape[1]
    tag = f"[{args.mode} r{rank}]"

    if args.mode == "baseline":
        model.eval()
        golden = []
        with torch.no_grad():
            for text in PROMPTS:
                e = tok(text, return_tensors="pt")
                out = model.generate(e.input_ids,
                                     attention_mask=e.attention_mask,
                                     max_new_tokens=args.max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
                golden.append(out[0, e.input_ids.shape[1]:].cpu())
                print(f"{tag} GOLDEN[{text!r}] text: {tok.decode(golden[-1])!r}", flush=True)
        if args.out:
            torch.save({"per_prompt": golden}, args.out)
        return 0

    import deepspeed
    from deepspeed.module_inject.segment_ki import apply_segment_ki

    deepspeed.init_distributed(dist_backend="gloo" if not has_accelerator() else "nccl")

    cfg = {
        "train_micro_batch_size_per_gpu": 1,
        "hybrid_engine": {
            "enabled": True,
            "inference_tp_size": 1,
            "max_out_tokens": 64,
        },
        "tensor_parallel": {
            "autotp_size": args.autotp
        },
    }
    if has_accelerator():
        cfg["bf16"] = {"enabled": True}

    engine, _, _, _ = deepspeed.initialize(model=model, config=cfg)
    engine.eval()

    if args.mode == "autotp_ki":
        # Segment KI runs strictly AFTER AutoTP: it consumes the sharded tree
        # as-is and never touches the collective-bearing boundary modules.
        report = apply_segment_ki(engine.module)
        print(f"{tag} KI_REPORT={report}", flush=True)

    out = engine.module.generate(prompt_ids,
                                 attention_mask=attn,
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
    new_ids = out[0, prompt_len:]
    print(f"{tag} text: {tok.decode(new_ids)!r}", flush=True)

    import deepspeed.comm as dist
    gathered = [torch.zeros_like(out) for _ in range(world)]
    dist.all_gather(gathered, out.contiguous())
    ranks_consistent = all(torch.equal(g, out) for g in gathered)

    verdict = None
    if args.ref:
        ref = torch.load(args.ref, weights_only=False)
        same = torch.equal(ref["per_prompt"][0].cpu(), new_ids.cpu())
        print(f"{tag} MATCH_GOLDEN={same} RANKS_CONSISTENT={ranks_consistent}")
        verdict = "PASS" if (same and ranks_consistent) else "FAIL"
        print(f"{tag} VERDICT={verdict}", flush=True)

    dist.barrier()
    if rank == 0:
        print(f"[{args.mode}] DONE verdict={verdict}", flush=True)
    return 0 if (verdict in (None, "PASS")) else 1


if __name__ == "__main__":
    sys.exit(main())
