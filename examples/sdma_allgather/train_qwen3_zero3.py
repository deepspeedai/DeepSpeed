# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Qwen3 + DeepSpeed ZeRO-3 benchmark for the SDMA allgather feature.

Loads a Qwen3 model with random initialisation under `deepspeed.zero.Init`
so each rank only allocates its 1/world_size shard, then runs a small number
of training steps on either real wikitext or synthetic random tokens.  Step
time is measured rank-0 side and reported with peak memory and the average
loss.  The same trainer is used for the SDMA-on and SDMA-off comparison runs
in run_qwen3_sdma_{on,off}.sh.

The SDMA fast-path is opt-in via a single env var: ``deepspeed.comm``
brings up the mori SDMA backend at init time when ``DS_SDMA_ALLGATHER=1``
and routes WORLD-group ``all_gather_into_tensor`` through
``mori_cpp.AllGatherIntoTensor`` on AMD MI300.  No ``ds_config`` flag is
required.  Leaving ``DS_SDMA_ALLGATHER`` unset (the default) reproduces
the RCCL/NCCL baseline for A/B comparisons even with mori installed.

Real-data path uses HuggingFace `datasets` to stream wikitext-103 and the
model's own tokenizer to pad/truncate to seq_length.  No external benchmark
repo is required.
"""

import argparse
import os
import time

import deepspeed
import torch
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-32B")
    p.add_argument("--num_layers",
                   type=int,
                   default=0,
                   help="0 = use model default; smaller values for quick smoke runs")
    p.add_argument("--seq_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--ds_config", required=True)
    p.add_argument("--dataset",
                   default="wikitext",
                   choices=["wikitext", "synthetic"],
                   help="Real text (wikitext-103) or pre-generated random ids")
    p.add_argument("--dataset_percentage",
                   type=float,
                   default=10.0,
                   help="Percentage of wikitext train split to load (1.0-100.0)")
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Self-contained data pipeline (no external benchmark repo dependency).
# ---------------------------------------------------------------------------
class _SyntheticDataset(Dataset):
    """Pre-generated random token ids for deterministic timing runs."""

    def __init__(self, vocab_size, seq_length, num_samples=10000, seed=42):
        gen = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(0, vocab_size, (num_samples, seq_length), generator=gen, dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones(self.seq_length, dtype=torch.long),
        }


def _build_wikitext_loader(model_name, seq_length, batch_size, dataset_percentage, rank, world_size, is_main):
    """Stream wikitext-103-raw-v1 as a concatenated token stream sliced into
    fixed `seq_length` chunks.

    This is the standard "group_texts" / GPT-style chunking pattern: every
    sample is exactly seq_length REAL tokens with no padding and no per-row
    boundaries.  Result is uniform-difficulty samples, so the per-step loss
    has no variance from "this row was 90 % padding" effects — which is what
    made the per-row+padding variant of this loader jittery on bs=1 demos.
    """
    from datasets import DownloadConfig, load_dataset
    from datasets.utils.logging import disable_progress_bar
    if not is_main:
        disable_progress_bar()

    fraction = max(1, int(dataset_percentage))
    split = "train" if dataset_percentage >= 100 else f"train[:{fraction}%]"

    if is_main:
        print(f"[trainer] loading wikitext-103-raw-v1 split={split}")
    raw = load_dataset("wikitext",
                       "wikitext-103-raw-v1",
                       split=split,
                       download_config=DownloadConfig(disable_tqdm=True))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.convert_ids_to_tokens(2)

    if is_main:
        print(f"[trainer] encoding {len(raw)} rows as a single stream ...")
    text = "\n\n".join(t for t in raw["text"] if t.strip())
    all_ids = tokenizer.encode(text, add_special_tokens=False)

    # Optional cap on number of chunks (env var) so the per-epoch length can
    # be tuned for short demos.  0 = use all available chunks.
    max_chunks = int(os.environ.get("QWEN3_MAX_CHUNKS", "0"))
    n_full = len(all_ids) // seq_length
    if max_chunks > 0:
        n_full = min(n_full, max_chunks)
    chunks = torch.tensor(all_ids[:n_full * seq_length], dtype=torch.long).view(n_full, seq_length)
    if is_main:
        print(f"[trainer] chunked: {len(all_ids)} tokens -> {n_full} "
              f"sequences of {seq_length} (no padding)",
              flush=True)

    class _ChunkDataset(Dataset):

        def __init__(self, t):
            self.t = t

        def __len__(self):
            return self.t.shape[0]

        def __getitem__(self, idx):
            ids = self.t[idx]
            return {
                "input_ids": ids,
                "labels": ids.clone(),
                "attention_mask": torch.ones(ids.shape[0], dtype=torch.long),
            }

    ds = _ChunkDataset(chunks)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=True, pin_memory=True)


def _build_loader(args, vocab_size, rank, world_size, is_main):
    if args.dataset == "wikitext":
        return _build_wikitext_loader(args.model_name, args.seq_length, args.batch_size, args.dataset_percentage, rank,
                                      world_size, is_main)
    ds = _SyntheticDataset(vocab_size, args.seq_length)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=True)


# ---------------------------------------------------------------------------
# Model construction under deepspeed.zero.Init so each rank only materialises
# its shard.
# ---------------------------------------------------------------------------
def build_model(model_name, num_layers, ds_config_path):
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if num_layers > 0:
        cfg.num_hidden_layers = num_layers
    cfg.torch_dtype = torch.bfloat16
    cfg.use_cache = False
    cfg.attn_implementation = "eager"  # FA2 not always available on AMD; eager is safe.
    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"[trainer] {model_name}: layers={cfg.num_hidden_layers} "
              f"hidden={cfg.hidden_size} heads={cfg.num_attention_heads} "
              f"kv_heads={cfg.num_key_value_heads} vocab={cfg.vocab_size}")
    with deepspeed.zero.Init(config_dict_or_path=ds_config_path):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    return model, cfg


def main():
    args = parse_args()
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world = dist.get_world_size()
    accel = get_accelerator()
    device_idx = args.local_rank if args.local_rank >= 0 else rank % accel.device_count()
    device = torch.device(accel.device_name(device_idx))
    accel.set_device(device_idx)

    if rank == 0:
        print(f"[trainer] world={world}  device={device}  ds_config={args.ds_config}")

    model, cfg = build_model(args.model_name, args.num_layers, args.ds_config)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=args.ds_config,
    )

    if rank == 0:
        from deepspeed.comm import mori as _mori
        print(f"[trainer] SDMA handle is_enabled={_mori.is_enabled()}", flush=True)

    loader = _build_loader(args, cfg.vocab_size, rank, world, rank == 0)
    if rank == 0:
        print(f"[trainer] dataloader: {len(loader)} batches/epoch, "
              f"running {args.num_steps} steps", flush=True)

    step_times, losses = [], []
    get_accelerator().reset_peak_memory_stats()
    t_train_start = time.perf_counter()
    step, epoch = 0, 0
    data_iter = iter(loader)
    skipped_empty = 0
    while step < args.num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch = next(data_iter)
        ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        # Defensive: on the chunked wikitext loader every chunk is full of
        # real tokens so these guards are no-ops, but they keep the trainer
        # safe for the synthetic mode and any future padded variants.
        if int(attn.sum().item()) == 0:
            skipped_empty += 1
            continue
        labels = labels.masked_fill(attn == 0, -100)
        get_accelerator().synchronize()
        t0 = time.perf_counter()
        out = engine(input_ids=ids, labels=labels, attention_mask=attn)
        engine.backward(out.loss)
        engine.step()
        get_accelerator().synchronize()
        dt = time.perf_counter() - t0

        if step >= args.warmup_steps:
            step_times.append(dt)
            losses.append(out.loss.detach().item())

        if rank == 0 and step % args.log_interval == 0:
            tag = "warmup" if step < args.warmup_steps else "measured"
            tps = args.batch_size * args.seq_length * world / dt
            print(
                f"[trainer] step {step:4d} ({tag:7s}) | loss {out.loss.item():8.4f} | "
                f"step {dt*1000:7.1f} ms | {tps:8.0f} tok/s",
                flush=True)
        step += 1

    t_train_end = time.perf_counter()

    if rank == 0:
        n = len(step_times)
        avg_dt = sum(step_times) / n
        tokens_per_step = args.batch_size * args.seq_length * world
        tps = tokens_per_step / avg_dt
        peak_gb = get_accelerator().max_memory_allocated() / 1e9
        avg_loss = sum(losses) / n
        print()
        print("=" * 70)
        print("Qwen3 ZeRO-3 benchmark complete")
        print(f"  measured steps   : {n} (warmup={args.warmup_steps} skipped)")
        print(f"  total wall (s)   : {t_train_end - t_train_start:.1f}")
        print(f"  avg step (ms)    : {avg_dt * 1000:.1f}")
        print(f"  tokens/sec (ws)  : {tps:.1f}")
        print(f"  peak mem (GB,r0) : {peak_gb:.2f}")
        print(f"  avg loss         : {avg_loss:.4f}")
        print(f"  final loss       : {losses[-1]:.4f}")
        print(f"  empty batches    : {skipped_empty}")
        print("=" * 70)


if __name__ == "__main__":
    main()
