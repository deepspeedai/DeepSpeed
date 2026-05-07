"""Qwen3 + DeepSpeed ZeRO-3 benchmark for the SDMA allgather feature.

Loads a Qwen3 model with random initialisation under `deepspeed.zero.Init`
so each rank only allocates its 1/world_size shard, then runs a small number
of training steps on either real wikitext or synthetic random tokens.  Step
time is measured rank-0 side and reported with peak memory and the average
loss.  The same trainer is used for the SDMA-on and SDMA-off comparison runs
in run_qwen3_sdma_{on,off}.sh.

The ZeRO-3 config (passed via --ds_config) controls whether the SDMA path is
taken: setting `sdma_allgather: true` makes _dist_allgather_fn route through
mori_cpp.AllGatherIntoTensor (this PR), `false` falls back to the upstream
RCCL/NCCL allgather.

Real-data path uses HuggingFace `datasets` to stream wikitext-103 and the
model's own tokenizer to pad/truncate to seq_length.  No external benchmark
repo is required.
"""

import argparse
import os
import time

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-32B")
    p.add_argument("--num_layers", type=int, default=0,
                   help="0 = use model default; smaller values for quick smoke runs")
    p.add_argument("--seq_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--ds_config", required=True)
    p.add_argument("--dataset", default="wikitext",
                   choices=["wikitext", "synthetic"],
                   help="Real text (wikitext-103) or pre-generated random ids")
    p.add_argument("--dataset_percentage", type=float, default=10.0,
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
        self.input_ids = torch.randint(0, vocab_size, (num_samples, seq_length),
                                       generator=gen, dtype=torch.long)
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


def _build_wikitext_loader(model_name, seq_length, batch_size, dataset_percentage,
                           rank, world_size, is_main):
    """Stream wikitext-103-raw-v1, tokenise with the model's tokenizer."""
    from datasets import DownloadConfig, load_dataset
    from datasets.utils.logging import disable_progress_bar
    if not is_main:
        disable_progress_bar()

    fraction = max(1, int(dataset_percentage))
    split = "train" if dataset_percentage >= 100 else f"train[:{fraction}%]"

    if is_main:
        print(f"[trainer] loading wikitext-103-raw-v1 split={split}")
    raw = load_dataset("wikitext", "wikitext-103-raw-v1", split=split,
                       download_config=DownloadConfig(disable_tqdm=True))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.convert_ids_to_tokens(2)

    def tok_fn(batch):
        return tokenizer(batch["text"], padding="max_length",
                         max_length=seq_length, truncation=True)

    if is_main:
        print(f"[trainer] tokenising {len(raw)} rows ...")
    tokenised = raw.map(tok_fn, batched=True, num_proc=1, keep_in_memory=True)
    tokenised.set_format(type="torch", columns=["input_ids", "attention_mask"])

    class _Labelled(Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            it = self.base[idx]
            return {
                "input_ids": it["input_ids"],
                "labels": it["input_ids"].clone(),
                "attention_mask": it["attention_mask"],
            }

    sampler = DistributedSampler(tokenised, num_replicas=world_size, rank=rank)
    return DataLoader(_Labelled(tokenised), batch_size=batch_size, sampler=sampler,
                      num_workers=2, drop_last=True, pin_memory=True)


def _build_loader(args, vocab_size, rank, world_size, is_main):
    if args.dataset == "wikitext":
        return _build_wikitext_loader(args.model_name, args.seq_length, args.batch_size,
                                      args.dataset_percentage, rank, world_size, is_main)
    ds = _SyntheticDataset(vocab_size, args.seq_length)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=True,
                      num_workers=0, pin_memory=True)


# ---------------------------------------------------------------------------
# Model construction under deepspeed.zero.Init so each rank only materialises
# its shard.  Passing the config_path here is required: Init reads
# zero_config.sdma_allgather and constructs the mori SDMA handle when true.
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
    device = torch.device(f"cuda:{args.local_rank if args.local_rank >= 0 else rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

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
        from deepspeed.runtime.comm import mori as _mori
        print(f"[trainer] SDMA handle is_enabled={_mori.is_enabled()}", flush=True)

    loader = _build_loader(args, cfg.vocab_size, rank, world, rank == 0)
    if rank == 0:
        print(f"[trainer] dataloader: {len(loader)} batches/epoch, "
              f"running {args.num_steps} steps", flush=True)

    step_times, losses = [], []
    torch.cuda.reset_peak_memory_stats()
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
        # Wikitext rows are highly variable; many are nearly empty (section
        # headers etc.) and become an all-pad batch after padding.  Such
        # batches contribute nothing to LM training (loss would be NaN under
        # the -100 mask below) and are skipped without consuming a step.
        if int(attn.sum().item()) == 0:
            skipped_empty += 1
            continue
        # Standard HF causal-LM training: padded positions must NOT contribute
        # to the loss.  Without this masking the model trivially predicts
        # pad_token on mostly-empty rows and reported loss collapses to ~0.
        labels = labels.masked_fill(attn == 0, -100)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = engine(input_ids=ids, labels=labels, attention_mask=attn)
        engine.backward(out.loss)
        engine.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if step >= args.warmup_steps:
            step_times.append(dt)
            losses.append(out.loss.detach().item())

        if rank == 0 and step % args.log_interval == 0:
            tag = "warmup" if step < args.warmup_steps else "measured"
            tps = args.batch_size * args.seq_length * world / dt
            print(f"[trainer] step {step:4d} ({tag:7s}) | loss {out.loss.item():8.4f} | "
                  f"step {dt*1000:7.1f} ms | {tps:8.0f} tok/s", flush=True)
        step += 1

    t_train_end = time.perf_counter()

    if rank == 0:
        n = len(step_times)
        avg_dt = sum(step_times) / n
        tokens_per_step = args.batch_size * args.seq_length * world
        tps = tokens_per_step / avg_dt
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
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
