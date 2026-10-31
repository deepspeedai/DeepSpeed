# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""vLLM bs=8 greedy, slope method, same prompts as ours."""

import time

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3.5-4B-Base"
BS = 8

para = ("The Eiffel Tower is located in Paris, France. It was built in 1889 "
        "and stands 330 meters tall. The Louvre museum houses the Mona Lisa. ")
SHORT_PROMPT = "The capital of France is"
LONG_PROMPT = para * 120


def timed(llm, text, n):
    prompts = [text] * BS
    sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
    llm.generate(prompts, sp)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    got = len(outs[0].outputs[0].token_ids)
    assert got == n, f"early stop {got}/{n}"
    return dt


def main():
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.92, max_model_len=8192)
    for ctx, text in [("short", SHORT_PROMPT), ("2k", LONG_PROMPT)]:
        t_s = timed(llm, text, 256)
        t_l = timed(llm, text, 1024)
        print(f"BS8 vllm {ctx:5s}: decode-only {768 * BS / (t_l - t_s):7.1f} tok/s total"
              f" | {768 / (t_l - t_s):6.2f} ms/step (T256 {t_s:.2f}s T1024 {t_l:.2f}s)", flush=True)
    print("BS8-VLLM-DONE", flush=True)


if __name__ == "__main__":
    main()
