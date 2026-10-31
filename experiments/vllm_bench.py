# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""vLLM bench with the same slope method as the DeepSpeed side.

decode-only = (T1024 - T256) / 768; e2e = 256 / T256. Metrics printed if present.
"""

import time

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3.5-4B-Base"

para = ("The Eiffel Tower is located in Paris, France. It was built in 1889 "
        "and stands 330 meters tall. The Louvre museum houses the Mona Lisa. ")
SHORT_PROMPT = "The capital of France is"
LONG_PROMPT = para * 120


def timed(llm, text, n):
    llm.generate([text], SamplingParams(temperature=0.0, max_tokens=n))  # warmup
    t0 = time.perf_counter()
    outs = llm.generate([text], SamplingParams(temperature=0.0, max_tokens=n))
    dt = time.perf_counter() - t0
    return dt, len(outs[0].outputs[0].token_ids), outs[0].metrics


def main():
    llm = LLM(model=MODEL,
              dtype="bfloat16",
              gpu_memory_utilization=0.92,
              max_model_len=8192)
    for label, text in [("short", SHORT_PROMPT), ("2k", LONG_PROMPT)]:
        t_s, n_s, m = timed(llm, text, 256)
        t_l, n_l, _ = timed(llm, text, 1024)
        decode_tps = (n_l - n_s) / (t_l - t_s)
        e2e_tps = n_s / t_s
        print(f"VLLM {label:5s}: e2e {e2e_tps:7.1f} tok/s | decode-only {decode_tps:7.1f} tok/s"
              f" | (T256 {t_s:.2f}s T1024 {t_l:.2f}s)", flush=True)
    print("VLLM-DONE", flush=True)


if __name__ == "__main__":
    main()
