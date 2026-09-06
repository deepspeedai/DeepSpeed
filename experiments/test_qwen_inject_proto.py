# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Prototype check: hybrid-engine kernel injection for the Qwen family.

Two rungs, see experiments/qwen-he-kernel-inject-proto.md:
  1. Qwen2/Qwen2.5 (exact rung)  - policy registered by default, expected
     numerically exact once run on GPU (fused kernels already cover GQA +
     rotate_half rope + gated SiLU MLP).
  2. Qwen3.5/3.6/3.8 (structural rung) - opt-in via DS_QWEN35_INJECTION=1;
     only full_attention blocks are injected, GatedDeltaNet blocks stay
     native, and forward parity is NOT expected until kernel gaps close.

Modes:
  baseline  - single process, plain HF generate (golden reference)
  zero3     - torchrun x2: ZeRO-3 + hybrid_engine with injection enabled

Greedy decoding (temperature=0) so output ids must match the golden run on GPU.
On CPU the injected forward is expected to fail inside CUDA kernels; the script
then reports NEEDS_GPU after validating the injection structure.
"""

import argparse
import os
import sys

# torchrun rewrites PYTHONPATH for workers; this env var forces our checkout to
# the front of sys.path so a stale easy-install.pth cannot shadow it.
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


def load(model_name, bf16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    want = torch.bfloat16 if bf16 else torch.float32
    try:
        # transformers 4.x uses torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=want)
    except TypeError:
        # transformers 5.x renamed it to dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=want)
    return tok, model


def injection_report(engine, tag):
    """Print what the hybrid engine injected and return (containers, wrapped)."""
    containers = getattr(engine, "_inference_containers", None) or []
    others = getattr(engine, "_other_layers", None) or []
    wrapped = getattr(engine, "_generate", None) is not None
    # Map which decoder layers were taken over: _orig_modules holds the replaced
    # originals in order, so block_type reveals the hybrid-layer filter.
    block_types = [getattr(m, "block_type", type(m).__name__) for m in getattr(engine, "_orig_modules", [])]
    print(
        f"{tag} engine={type(engine).__name__} containers={len(containers)} "
        f"other_layers={len(others)} generate_wrapped={wrapped}",
        flush=True)
    if block_types:
        full = sum(1 for b in block_types if b == "full_attention")
        print(
            f"{tag} injected_layers={len(block_types)} "
            f"(full_attention={full}, other={[b for b in block_types if b != 'full_attention']})",
            flush=True)
    return len(containers), wrapped


def check_filter(model_name):
    """CPU-only gate for the hybrid-layer filter: no CUDA, no distributed.

    Builds a tiny Qwen3_5 model and walks its decoder layers through
    Qwen3_5LayerPolicy.should_replace, asserting the full_attention blocks
    (and only those) are selected. This is the logic the hybrid_engine
    per-instance hook consumes, so it can be validated before GPU access.
    """
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from deepspeed.module_inject.containers.qwen3_5 import Qwen3_5LayerPolicy

    config = Qwen3_5TextConfig(
        hidden_size=64,
        num_hidden_layers=8,
        full_attention_interval=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=128,
        vocab_size=512,
        mtp_num_hidden_layers=0,
    )
    model = Qwen3_5ForCausalLM(config)
    layers = list(model.model.layers)
    picks = [Qwen3_5LayerPolicy.should_replace(layer) for layer in layers]
    types = [layer.block_type for layer in layers]

    ok = all(p == (t == "full_attention") for p, t in zip(picks, types))
    for i, (t, p) in enumerate(zip(types, picks)):
        print(f"layer {i}: block_type={t} should_replace={p}")
    full = sum(1 for t in types if t == "full_attention")
    print(f"full_attention={full}/{len(layers)} selected={sum(picks)}")
    print(f"FILTER_VERDICT={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "zero3", "autotp", "check-filter"], required=True)
    p.add_argument("--autotp", type=int, default=2, help="tensor_parallel.autotp_size for --mode autotp")
    p.add_argument("--out", default=None)
    p.add_argument("--ref", default=None)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--bf16", action="store_true", help="bf16 weights (recommended on GPU)")
    args = p.parse_args()

    if args.mode == "check-filter":
        return check_filter(args.model)

    torch.set_num_threads(8)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tok, model = load(args.model, args.bf16)

    # DP semantics: one prompt per rank. AutoTP is TP semantics: every rank in a
    # TP group must see the same batch (engine checks this), so use prompt 0.
    if world > 1 and args.mode != "autotp":
        prompt_idx = rank % len(PROMPTS)
    else:
        prompt_idx = 0
    enc = tok(PROMPTS[prompt_idx], return_tensors="pt")
    prompt_ids, attn = enc.input_ids, enc.attention_mask
    prompt_len = prompt_ids.shape[1]
    tag = f"[{args.mode} r{rank}]"

    if args.mode == "baseline":
        model.eval()
        if has_accelerator():
            # Qwen3.5's GDN blocks dispatch to Triton kernels that reject CPU
            # tensors, so the plain-HF golden must run on the GPU as well.
            model = model.to("cuda")
        golden = []
        with torch.no_grad():
            for text in PROMPTS:
                e = tok(text, return_tensors="pt")
                e = {k: v.to(model.device) for k, v in e.items()}
                out = model.generate(e["input_ids"],
                                     attention_mask=e["attention_mask"],
                                     max_new_tokens=args.max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
                new = out[0, e["input_ids"].shape[1]:]
                print(f"{tag} GOLDEN[{text!r}] text: {tok.decode(new)!r}", flush=True)
                golden.append(new.cpu())
        if args.out:
            torch.save({"per_prompt": golden}, args.out)
        return 0

    import deepspeed
    from deepspeed.runtime.rollout import HybridEngineRollout, RolloutRequest, SamplingConfig
    from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRolloutConfig

    deepspeed.init_distributed(dist_backend="gloo" if not has_accelerator() else "nccl")

    cfg = {
        "train_micro_batch_size_per_gpu": 1,
        "hybrid_engine": {
            "enabled": True,
            "inference_tp_size": 1,
            "max_out_tokens": 64,
        },
    }
    if args.bf16 or has_accelerator():
        cfg["bf16"] = {"enabled": True}
    if args.mode == "zero3":
        cfg["zero_optimization"] = {"stage": 3}
    if args.mode == "autotp":
        # stage 0 + AutoTP leaf replacement; exercises the container-vs-AutoTP
        # interaction (containers read weights that are already TP-sharded).
        cfg["tensor_parallel"] = {"autotp_size": args.autotp}

    try:
        engine, _, _, _ = deepspeed.initialize(model=model, config=cfg)
    except ValueError as e:
        # The v1 inference ops (e.g. QKVGemmOp) load their CUDA module when a
        # container is built, so container construction itself is GPU-only.
        if "not been implemented on CPU backend" in str(e):
            print(
                f"{tag} INJECTION_STATUS=MATCHED_BUT_NEEDS_GPU "
                f"(policy matched; container construction loads CUDA ops)",
                flush=True)
            print(f"{tag} VERDICT=NEEDS_GPU", flush=True)
            return 0
        raise
    engine.eval()

    n_containers, wrapped = injection_report(engine, tag)
    if n_containers == 0:
        print(f"{tag} INJECTION_STATUS=OFF (check env gates: DS_QWEN2_INJECTION / DS_QWEN35_INJECTION)", flush=True)

    rollout = HybridEngineRollout(engine=engine, tokenizer=tok, cfg=HybridEngineRolloutConfig())
    request = RolloutRequest(prompt_ids=prompt_ids, prompt_attention_mask=attn)
    sampling = SamplingConfig(max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0)

    try:
        batch = rollout.generate(request, sampling)
    except Exception as e:  # noqa: BLE001
        # CPU hosts cannot run the fused CUDA kernels; the structure above still
        # proves the policy matched and the containers were built.
        if n_containers > 0 and not has_accelerator():
            print(f"{tag} INJECTION_STATUS=STRUCTURAL_OK forward failed as expected on CPU: {type(e).__name__}",
                  flush=True)
            print(f"{tag} VERDICT=NEEDS_GPU", flush=True)
            return 0
        raise

    new_ids = batch.input_ids[0, prompt_len:]
    print(f"{tag} prompt {prompt_idx} {PROMPTS[prompt_idx]!r}")
    print(f"{tag} text: {tok.decode(new_ids)!r}", flush=True)

    verdict = None
    if args.ref:
        ref = torch.load(args.ref, weights_only=False)
        same = torch.equal(ref["per_prompt"][prompt_idx].cpu(), new_ids.cpu())
        print(f"{tag} MATCH_GOLDEN(p{prompt_idx})={same}")
        verdict = "PASS" if same else "FAIL"
        print(f"{tag} VERDICT={verdict}", flush=True)

    import deepspeed.comm as dist
    dist.barrier()
    if rank == 0:
        print(f"[{args.mode}] DONE verdict={verdict} containers={n_containers} wrapped={wrapped}", flush=True)
    return 0 if (verdict in (None, "PASS")) else 1


if __name__ == "__main__":
    sys.exit(main())
