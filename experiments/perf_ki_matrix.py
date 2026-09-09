# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Performance matrix for the Qwen KI prototypes vs hf.generate.

Modes:
  hf           - plain HF generate (single GPU baseline)
  hf_graph     - HF generate with graph capture (torch.compile reduce-overhead)
  zero3_ki     - old KI: ZeRO-3 DP=2 + hybrid-engine container injection
                 (op_builder csrc kernels; numerics known-broken on tf>=5, perf only)
  autotp_segki - segment KI: AutoTP TP=2 + native CUDA fused_glu
                 (requires DS_QWEN2_INJECTION=0: the two KI lines are mutually
                 exclusive at runtime)

Timing: batch 1, greedy; prefill measured with max_new_tokens=1, decode
estimated as (total_128 - prefill). tokens/s = 127 / decode.
"""

import argparse
import os
import sys

_ds_src = os.environ.get("DS_SRC")
if _ds_src:
    sys.path.insert(0, _ds_src)

import time  # noqa: E402

import torch  # noqa: E402
from deepspeed.accelerator import get_accelerator  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of France is"
NEW_TOKENS = 128


def bench(generate_fn, tok, warmup=2, batch=1, out_path=None, ref_path=None, tag=""):
    device = get_accelerator().current_device_name()
    enc = {k: v.to(device) for k, v in tok([PROMPT] * batch, return_tensors="pt").items()}

    for _ in range(warmup):
        generate_fn(enc["input_ids"], enc["attention_mask"], 4)
    get_accelerator().synchronize()

    t0 = time.perf_counter()
    generate_fn(enc["input_ids"], enc["attention_mask"], 1)
    get_accelerator().synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    out = generate_fn(enc["input_ids"], enc["attention_mask"], NEW_TOKENS)
    get_accelerator().synchronize()
    total_ms = (time.perf_counter() - t0) * 1e3

    decode_ms = max(total_ms - prefill_ms, 1e-6)
    tps = (NEW_TOKENS - 1) * batch / (decode_ms / 1e3)
    if out_path is not None:
        torch.save({"ids": out.cpu()}, out_path)
    if ref_path is not None:
        ref = torch.load(ref_path, weights_only=False)["ids"]
        match = torch.equal(ref.cpu(), out.cpu())
        print(f"{tag} MATCH_REF={match}", flush=True)
    return prefill_ms, total_ms, decode_ms, tps, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",
                   choices=["hf", "hf_graph", "hf_segki", "zero0_ki", "zero3_ki", "autotp", "autotp_segki"],
                   required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--out", default=None, help="save generated ids for correctness comparison")
    p.add_argument("--ref", default=None, help="compare greedy ids against a saved reference")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    want = torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=want)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=want)
    model = model.to(get_accelerator().current_device_name()).eval()

    def hf_generate(input_ids, attention_mask, n):
        return model.generate(input_ids,
                              attention_mask=attention_mask,
                              max_new_tokens=n,
                              do_sample=False,
                              pad_token_id=tok.pad_token_id)

    tag = f"[{args.mode} r{os.environ.get('RANK', '0')}]"

    if args.mode == "hf":
        prefill, total, decode, tps, out = bench(hf_generate,
                                                 tok,
                                                 batch=args.batch,
                                                 out_path=args.out,
                                                 ref_path=args.ref,
                                                 tag=tag)
    elif args.mode == "hf_segki":
        # Segment KI on the plain single-GPU HF model (no AutoTP): isolates the
        # kernel contribution from any TP/communication effects.
        from deepspeed.module_inject.segment_ki import apply_segment_ki
        report = apply_segment_ki(model, backend="auto")
        print(f"{tag} KI_REPORT={report}", flush=True)
        prefill, total, decode, tps, out = bench(hf_generate,
                                                 tok,
                                                 batch=args.batch,
                                                 out_path=args.out,
                                                 ref_path=args.ref,
                                                 tag=tag)
    elif args.mode == "hf_graph":
        # Graph capture via cudagraph trees; HF decode steps keep a [1,1]
        # input shape with KV caches so the captured graph can replay.
        compiled = torch.compile(model, mode="reduce-overhead")

        def graph_generate(input_ids, attention_mask, n):
            return compiled.generate(input_ids,
                                     attention_mask=attention_mask,
                                     max_new_tokens=n,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)

        prefill, total, decode, tps, out = bench(graph_generate,
                                                 tok,
                                                 warmup=4,
                                                 batch=args.batch,
                                                 out_path=args.out,
                                                 ref_path=args.ref,
                                                 tag=tag)
    else:
        import deepspeed
        deepspeed.init_distributed(dist_backend="nccl")

        cfg = {
            "train_micro_batch_size_per_gpu": 1,
            "bf16": {
                "enabled": True
            },
            "hybrid_engine": {
                "enabled": True,
                "inference_tp_size": 1,
                "max_out_tokens": 256,
            },
        }
        if args.mode == "zero3_ki":
            cfg["zero_optimization"] = {"stage": 3}
        # zero0_ki: stage 0 so no ZeRO gather overhead confounds the kernels.
        if args.mode in ("autotp", "autotp_segki"):
            cfg["tensor_parallel"] = {"autotp_size": 2}

        engine, _, _, _ = deepspeed.initialize(model=model, config=cfg)
        engine.eval()

        if args.mode == "autotp_segki":
            from deepspeed.module_inject.segment_ki import apply_segment_ki
            report = apply_segment_ki(engine.module, backend="auto")
            print(f"{tag} KI_REPORT={report}", flush=True)

        def engine_generate(input_ids, attention_mask, n):
            return engine.module.generate(input_ids,
                                          attention_mask=attention_mask,
                                          max_new_tokens=n,
                                          do_sample=False,
                                          pad_token_id=tok.pad_token_id)

        prefill, total, decode, tps, out = bench(engine_generate,
                                                 tok,
                                                 batch=args.batch,
                                                 out_path=args.out,
                                                 ref_path=args.ref,
                                                 tag=tag)

    new = out[0, -NEW_TOKENS:]
    print(
        f"{tag} prefill_ms={prefill:.1f} total_ms={total:.1f} decode_ms={decode:.1f} "
        f"decode_tokens_per_sec={tps:.1f}",
        flush=True)
    print(f"{tag} text_tail={tok.decode(new[-24:])!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
