# PR description — Performance section update

## Performance

Qwen3.5-4B, RTX 4080 SUPER, greedy, bf16. Decode-only tok/s (slope
method: (T₁₀₂₄−T₂₅₆)/768 — per-call graph capture excluded; all engines
measured identically in-session).

### b=1

| | short ctx | 2K ctx | vs vLLM |
|---|---|---|---|
| HF eager | 18.4 | 21.5 | 24% / 29% |
| graph capture only | 63.9 | 59.3 | 84% / 80% |
| **DeepSpeed (this PR)** | **68.2** | **65.2** | **89% / 88%** |
| vLLM 0.31 | 76.4 | 74.3 | — |

### b=8

| | short ctx | 2K ctx | vs vLLM |
|---|---|---|---|
| graph capture only | 360 | 274 | 67% / 54% |
| **DeepSpeed (this PR)** | **445** | **409** | **83% / 81%** |
| vLLM 0.31 | 537 | 508 | — |

Batched decode: the custom decode-attention kernel runs at any batch
size (one block per row·head, per-row positions); at b=8 it lifts
throughput +24% (short) / +50% (2K) over the SDPA fallback — 54% → 81%
of vLLM at 2K context. Batched GEMV kernels were measured slower than
cuBLAS GEMM at b=8 (weight re-streaming dominates) and stay on the
composite path by design. The remaining b>1 gap to vLLM is per-GEMM
efficiency, not stepping overhead (batched step kernel: neutral).
