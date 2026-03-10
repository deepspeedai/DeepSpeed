"""
DeepSpeed ZeRO-3 training example with allgather overlap.
Trains a GPT-2-style transformer on synthetic data for demonstration.
Designed for single-node 8x AMD GPU setup.
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Model: minimal GPT-2-style transformer
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len),
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(hidden_size, num_heads, max_seq_len, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.size()
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        return loss, logits


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
class SyntheticTextDataset(Dataset):
    """Generates random token sequences, deterministic across runs."""

    def __init__(self, vocab_size, seq_len, num_samples, seed=42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        g = torch.Generator()
        g.manual_seed(self.seed + idx)
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=g)
        return tokens[:-1], tokens[1:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed ZeRO-3 training with allgather overlap")
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    ds_config_path = args.deepspeed_config
    if ds_config_path and not os.path.isfile(ds_config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ds_config_path = os.path.join(script_dir, ds_config_path)
    args.deepspeed_config = ds_config_path

    deepspeed.init_distributed(dist_backend="cpu:gloo,cuda:nccl")
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    with deepspeed.zero.Init(config_dict_or_path=ds_config_path):
        model = GPT2Model(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
        )

    total_params = sum(p.numel() for p in model.parameters())
    num_gpus = torch.distributed.get_world_size()
    if local_rank == 0:
        print(f"Model parameters: {total_params / 1e6:.1f}M")
        print(f"GPUs: {num_gpus}")

    # FLOPs per token (forward + backward): 6*params + 12*L*H*S
    # Reference: "Efficient Large-Scale Language Model Training on GPU Clusters
    #             Using Megatron-LM" (Narayanan et al., 2021)
    flops_per_token = 6 * total_params + 12 * args.num_layers * args.hidden_size * args.max_seq_len

    dataset = SyntheticTextDataset(args.vocab_size, args.max_seq_len, args.num_samples)

    model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        training_data=dataset,
    )

    device = model_engine.device

    step = 0
    t_start = time.time()
    while step < args.train_steps:
        for batch in train_loader:
            if step >= args.train_steps:
                break

            input_ids = batch[0].to(device)
            labels = batch[1].to(device)

            loss, _ = model_engine(input_ids, labels=labels)
            model_engine.backward(loss)
            model_engine.step()

            if step % 10 == 0 and local_rank == 0:
                elapsed = time.time() - t_start
                global_batch = model_engine.train_batch_size()
                samples_per_sec = (step + 1) * global_batch / elapsed
                tokens_per_sec = samples_per_sec * args.max_seq_len
                tflops = tokens_per_sec * flops_per_token / 1e12
                tflops_per_gpu = tflops / num_gpus
                print(
                    f"step {step:5d} | loss {loss.item():.4f} | "
                    f"lr {lr_scheduler.get_last_lr()[0]:.6f} | "
                    f"{samples_per_sec:.1f} samples/s | "
                    f"{tflops:.1f} TFLOPS | "
                    f"{tflops_per_gpu:.1f} TFLOPS/GPU"
                )
            step += 1

    if local_rank == 0:
        total_time = time.time() - t_start
        total_samples = args.train_steps * model_engine.train_batch_size()
        avg_samples_per_sec = total_samples / total_time
        avg_tokens_per_sec = avg_samples_per_sec * args.max_seq_len
        avg_tflops = avg_tokens_per_sec * flops_per_token / 1e12
        avg_tflops_per_gpu = avg_tflops / num_gpus
        print(f"\nTraining complete: {args.train_steps} steps in {total_time:.1f}s")
        print(f"  Avg throughput : {avg_samples_per_sec:.1f} samples/s")
        print(f"  Avg TFLOPS     : {avg_tflops:.1f} (total) | {avg_tflops_per_gpu:.1f} (per GPU)")


if __name__ == "__main__":
    main()
