# SDMA AllGather for ZeRO-3

End-to-end example of the `sdma_allgather` flag wired into ZeRO-3's parameter
prefetch path.  When enabled, ZeRO-3's `_dist_allgather_fn` routes through
`mori_cpp.AllGatherIntoTensor` (intra-node SDMA copy on AMD MI300), with a
transparent fallback to `dist.allgather_fn` (RCCL/NCCL) on init failure.

## Enabling the SDMA path

ZeRO-3 config knob and one env var:

```jsonc
"zero_optimization": {
    "stage": 3,
    "sdma_allgather": true,
    "sdma_allgather_max_numel": 67108864
}
```

```bash
export MORI_ENABLE_SDMA=1   # uncached transit buffers required by the kernel
```

`MORI_ENABLE_SDMA` is required because the SDMA copy kernel reads transit
memory directly; without it mori's `SymmMemManager` falls back to cached
allocations and the kernel faults at NULL on every rank.  All
`run_*_sdma_on.sh` scripts in this directory export it for you.

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

The Qwen3-32B number is averaged over two fresh rounds; per-round delta was
+10.85 % and +9.92 %, with 0.29 % run-to-run variance on the off baseline, so
the gap is well outside per-step jitter (~1.5–2.7 %).

### Loss curves match across off ↔ on (2000-step runs)

A long-horizon sanity check on each demo confirms the SDMA path introduces
no numerical drift: 2000 training steps on the same wikitext shuffle, off
vs on traces overlap throughout.  Both trainers use the standard "concat
the corpus + slice into fixed `seq_length` chunks" pattern, so every
sample has the same number of real tokens and per-step loss has no
variance from padding fraction.  Bucketed mean |off − on| over the full
2000 steps is ≤ **0.026** on GPT and ≤ **0.048** on Qwen3, well below
natural per-step jitter.

![GPT-7B-ish — training loss vs step, SDMA off vs on, 2000 steps](images/loss_gpt_2k.png)

![Qwen3-32B — training loss vs step, SDMA off vs on, 2000 steps](images/loss_qwen3_2k.png)

The SDMA path is a pure plumbing change with no numerical impact in either
workload.

## Reproduction

```bash
cd examples/sdma_allgather

# Demo 1 — GPT-7B-ish, ~minute run, no HF download
bash run_gpt_sdma_off.sh    # baseline RCCL allgather
bash run_gpt_sdma_on.sh     # mori SDMA allgather   -> +10.85 %

# Demo 2 — Qwen3-32B, ~few-minute run, weight-free (random init via from_config)
bash run_qwen3_sdma_off.sh  # ~1402 ms / step
bash run_qwen3_sdma_on.sh   # ~1263 ms / step       -> +9.93 %
```

The configs already use DeepSpeed's default ZeRO-3 bucket sizes, so the
numbers above are reproducible without any tuning.  Override knobs via env
vars: `SEQ_LEN`, `BATCH_SIZE`, `NUM_STEPS`, `WARMUP_STEPS`, `NUM_GPUS`,
`MODEL`, `DS_CONFIG`.

## Files

```
ds_config_zero3_nosdma.json     ZeRO-3 + bf16 + DS-default buckets, sdma off
ds_config_zero3_sdma.json       same as above + sdma_allgather = true
run_gpt_sdma_off.sh             GPT-7B-ish + ZeRO-3, SDMA off
run_gpt_sdma_on.sh              GPT-7B-ish + ZeRO-3, SDMA on
run_qwen3_sdma_off.sh           Qwen3-32B + ZeRO-3, SDMA off
run_qwen3_sdma_on.sh            Qwen3-32B + ZeRO-3, SDMA on
test_sdma_allgather_zero3.py    unit test exercising the ZeRO-3 SDMA path
train_qwen3_zero3.py            Qwen3 trainer (self-contained, wikitext)
train_zero3.py                  GPT trainer (existing, unchanged)
images/loss_gpt_2k.png          GPT loss curve, off vs on, 2000 steps
images/loss_qwen3_2k.png        Qwen3-32B loss curve, off vs on, 2000 steps
```
