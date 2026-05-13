# SDMA AllGather (transparent backend in `deepspeed.comm`)

End-to-end example for the SDMA fast-path inside
`TorchBackend.all_gather_into_tensor`.  When the runtime is AMD/ROCm and
the `mori` package is importable, `deepspeed.comm` auto-acquires the SDMA
backend at `init_distributed()` time and transparently routes WORLD-group
`all_gather_into_tensor` calls through `mori_cpp.AllGatherIntoTensor`
(intra-node SDMA copy on MI300).  RCCL/NCCL is used as the fallback on
any condition that makes the SDMA path unsafe (non-WORLD process group,
shard larger than the transit buffer, unsupported dtype, init failure).

This means:

- No `ds_config` knob — works out of the box for ZeRO-3 (sequential and
  coalesced prefetch paths both benefit).
- No source modifications in `partition_parameters.py`: ZeRO-3 just calls
  `dist.allgather_fn`, which lands on the backend's
  `all_gather_into_tensor`.
- Sub-group allgathers (e.g. when ZeRO is initialised with a non-WORLD
  data-parallel group, or with a secondary zero-param group) are routed
  through RCCL/NCCL automatically, since the SDMA backend is bound to
  WORLD.

## Environment variables

| Var | Purpose |
|---|---|
| `MORI_ENABLE_SDMA=1` | **Required for the SDMA path.**  Tells mori to allocate `hipExtMallocWithFlags + hipDeviceMallocUncached` transit buffers; without it the SDMA kernel reads cached memory and faults at NULL. |
| `DS_DISABLE_SDMA_ALLGATHER=1` | Debug / A-B baseline switch.  Force-disables the SDMA fast-path even when mori is available. |
| `DS_SDMA_ALLGATHER_MAX_NUMEL=N` | Transit buffer size in elements (default 64M = 256 MiB per-rank input, ~2 GiB output on 8 ranks).  Calls larger than this fall back to RCCL/NCCL. |

The `run_*_sdma_on.sh` scripts export `MORI_ENABLE_SDMA=1`; the
`run_*_sdma_off.sh` scripts export `DS_DISABLE_SDMA_ALLGATHER=1`.  Both
variants share the same `ds_config_zero3.json` — the SDMA decision is
made entirely by env vars.

## Verified results on 8x MI300X

| | GPT-7B-ish | Qwen3-32B |
|---|---|---|
| trainer | `train_zero3.py` | `train_qwen3_zero3.py` |
| seq / micro batch | 2048 / 1 | 1024 / 1 |
| dataset | wikitext-2-raw-v1 | wikitext-103-raw-v1 (10 %) |
| measured / warmup steps | 100 / 10 | 100 / 10 |
| **SDMA off (RCCL)** | 697.7 ms / step (11.6 samples/s) | 1402.5 ms / step (5841 tok/s) |
| **SDMA on (this PR)** | **622.0 ms / step (13.0 samples/s)** | **1263.2 ms / step (6486 tok/s)** |
| **gain** | **+10.85 %** | **+9.93 %** |
| peak mem (rank 0) | unchanged off ↔ on | 96.45 GB, unchanged off ↔ on |

The Qwen3-32B number is averaged over two fresh rounds; per-round delta
was +10.85 % and +9.92 %, with 0.29 % run-to-run variance on the off
baseline, so the gap is well outside per-step jitter (~1.5–2.7 %).

Speedup is workload-dependent — gains shrink (or invert) when allgather
cannot be overlapped with compute (e.g. very small payloads, or
`overlap_comm=false`).

### Loss curves match across off ↔ on (2000-step runs)

A long-horizon sanity check on each demo confirms the SDMA path
introduces no numerical drift: 2000 training steps on the same wikitext
shuffle, off vs on traces overlap throughout.  Both trainers use the
standard "concat the corpus + slice into fixed `seq_length` chunks"
pattern, so every sample has the same number of real tokens and per-step
loss has no variance from padding fraction.  Bucketed mean |off − on|
over the full 2000 steps is ≤ **0.026** on GPT and ≤ **0.048** on Qwen3,
well below natural per-step jitter.

![GPT-7B-ish — training loss vs step, SDMA off vs on, 2000 steps](images/loss_gpt_2k.png)

![Qwen3-32B — training loss vs step, SDMA off vs on, 2000 steps](images/loss_qwen3_2k.png)

## Reproduction

```bash
cd examples/sdma_allgather

# Demo 1 — GPT-7B-ish, ~minute run, no HF download
bash run_gpt_sdma_off.sh    # DS_DISABLE_SDMA_ALLGATHER=1, RCCL baseline
bash run_gpt_sdma_on.sh     # MORI_ENABLE_SDMA=1, transparent SDMA path -> +10.85 %

# Demo 2 — Qwen3-32B, ~few-minute run, weight-free (random init via from_config)
bash run_qwen3_sdma_off.sh  # ~1402 ms / step
bash run_qwen3_sdma_on.sh   # ~1263 ms / step       -> +9.93 %
```

Override knobs via env vars: `SEQ_LEN`, `BATCH_SIZE`, `NUM_STEPS`,
`WARMUP_STEPS`, `NUM_GPUS`, `MODEL`, `DS_CONFIG`.

## Files

```
ds_config_zero3.json            single shared ZeRO-3 + bf16 + DS-default buckets config
run_gpt_sdma_off.sh             GPT-7B-ish + ZeRO-3, SDMA disabled via env var
run_gpt_sdma_on.sh              GPT-7B-ish + ZeRO-3, SDMA enabled via env var
run_qwen3_sdma_off.sh           Qwen3-32B + ZeRO-3, SDMA disabled via env var
run_qwen3_sdma_on.sh            Qwen3-32B + ZeRO-3, SDMA enabled via env var
test_sdma_allgather_zero3.py    unit test exercising the transparent SDMA path
train_qwen3_zero3.py            Qwen3 trainer (self-contained, wikitext)
train_zero3.py                  GPT trainer
images/loss_gpt_2k.png          GPT loss curve, off vs on, 2000 steps
images/loss_qwen3_2k.png        Qwen3-32B loss curve, off vs on, 2000 steps
```
