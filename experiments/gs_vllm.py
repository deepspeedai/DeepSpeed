# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""vLLM greedy vs sampling (slope method), short + 2K, b=1 — same session as ours."""

import time

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3.5-4B-Base"

para = ("The Eiffel Tower is located in Paris, France. It was built in 1889 "
        "and stands 330 meters tall. The Louvre museum houses the Mona Lisa. ")
SHORT_PROMPT = "The capital of France is"
LONG_PROMPT = para * 120


def timed(llm, text, sp):
    llm.generate([text], sp)  # warmup
    t0 = time.perf_counter()
    outs = llm.generate([text], sp)
    dt = time.perf_counter() - t0
    return dt


def main():
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.92, max_model_len=8192)
    cases = [
        ("greedy", dict(temperature=0.0)),
        ("sample t1.0", dict(temperature=1.0, top_p=1.0, seed=0)),
        ("sample t1.0 p0.9", dict(temperature=1.0, top_p=0.9, seed=0)),
    ]
    for label, kw in cases:
        for ctx, text in [("short", SHORT_PROMPT), ("2k", LONG_PROMPT)]:
            t_s = timed(llm, text, SamplingParams(max_tokens=256, ignore_eos=True, **kw))
            t_l = timed(llm, text, SamplingParams(max_tokens=1024, ignore_eos=True, **kw))
            print(f"GS vllm {ctx:5s} {label:18s}: decode-only {768 / (t_l - t_s):7.1f} tok/s"
                  f" (T256 {t_s:.2f}s T1024 {t_l:.2f}s)", flush=True)
    print("GS-VLLM-DONE", flush=True)


if __name__ == "__main__":
    main()
