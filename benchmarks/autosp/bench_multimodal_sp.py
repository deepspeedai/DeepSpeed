# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Benchmark: AutoSP multimodal sequence parallelism (ViT SP + fusion adapter).

Measures per-iteration latency, throughput, and peak GPU memory for the
ViT-SP + fusion-adapter pipeline at a given SP degree.

Launch (from repo root):

    # SP degree 2 — two GPUs:
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 \\
        benchmarks/autosp/bench_multimodal_sp.py [args]

    # Baseline — single GPU (all-gather/scatter are no-ops):
    torchrun --nproc_per_node=1 \\
        benchmarks/autosp/bench_multimodal_sp.py [args]

Compare the two output tables to quantify memory savings and throughput scaling.

Arguments:
    --arch          {internvl, qwen2vl}   architecture to simulate (default: internvl)
    --batch-size    N                     samples per batch (default: 2)
    --seq-len       N                     text sequence length (default: 512)
    --visual-tokens N                     total visual tokens per sample (default: 256)
    --hidden        N                     hidden dimension (default: 1024)
    --num-layers    N                     ViT and LLM layers each (default: 2)
    --iters         N                     measured iterations (default: 50)
    --warmup        N                     warmup iterations (default: 10)
"""

import argparse
import logging
import statistics

import torch
import torch.nn as nn

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.sequence.auto_sp import auto_wrap_model_for_sp
from deepspeed.sequence.autosp_fusion import InternVLFusionAdapter, Qwen2VLFusionAdapter

# ---------------------------------------------------------------------------
# Token IDs
# ---------------------------------------------------------------------------

_INTERNVL_CONTEXT_ID = 92546
_QWEN2VL_START_ID = 151652
_QWEN2VL_END_ID = 151653

# ---------------------------------------------------------------------------
# Mock attention classes — names match autosp_detector registries exactly
# ---------------------------------------------------------------------------


class InternVisionAttention(nn.Module):

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class InternLM2Attention(nn.Module):

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class Qwen2VLVisionAttention(nn.Module):

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class Qwen2Attention(nn.Module):

    def forward(self, hidden_states, **kwargs):
        return hidden_states


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------


class _ViTBlock(nn.Module):
    """One ViT transformer block: attention (to be SP-wrapped) + linear FFN."""

    def __init__(self, attn_cls, hidden: int) -> None:
        super().__init__()
        self.attn = attn_cls()
        self.ffn = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, **kwargs):
        out = self.attn(x, **kwargs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return self.ffn(out)


class _MinimalInternVLModel(nn.Module):
    """InternVL-like benchmark model.

    Module paths detected by autosp_detector:
    - ``vision_encoder.*.attn``  -> InternVisionAttention  (_VIT_ATTN_CLASSNAMES)
    - ``mm_projector``           -> keyword in _VISION_PROJ_KEYWORDS

    ``language_model`` uses plain nn.Linear layers so it is NOT wrapped by
    DistributedAttention (avoids the Q/K/V interface requirement) yet still
    contributes realistic compute on the scattered fused sequence.
    """

    def __init__(self, hidden: int, num_layers: int) -> None:
        super().__init__()
        self.vision_encoder = nn.Sequential(*[_ViTBlock(InternVisionAttention, hidden) for _ in range(num_layers)])
        self.mm_projector = nn.Identity()
        self.language_model = nn.Sequential(*[nn.Linear(hidden, hidden, bias=False) for _ in range(num_layers)])
        self.fusion = None

    def forward(self, local_patches: torch.Tensor, text_embeds: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        local_visual = self.vision_encoder(local_patches)
        local_fused = self.fusion(local_visual, text_embeds, input_ids)
        return self.language_model(local_fused)


class _MinimalQwen2VLModel(nn.Module):
    """Qwen2VL-like benchmark model."""

    def __init__(self, hidden: int, num_layers: int) -> None:
        super().__init__()
        self.visual = nn.Sequential(*[_ViTBlock(Qwen2VLVisionAttention, hidden) for _ in range(num_layers)])
        self.multi_modal_projector = nn.Identity()
        self.model = nn.Sequential(*[nn.Linear(hidden, hidden, bias=False) for _ in range(num_layers)])
        self.fusion = None

    def forward(self, local_patches: torch.Tensor, text_embeds: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        local_visual = self.visual(local_patches)
        local_fused = self.fusion(local_visual, text_embeds, input_ids)
        return self.model(local_fused)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _build_model_and_inputs(arch: str, args, sp_group, device):
    rank = dist.get_rank(sp_group)
    world_size = dist.get_world_size(sp_group)

    local_v = args.visual_tokens // world_size
    bs, text_len, hidden = args.batch_size, args.seq_len, args.hidden

    torch.manual_seed(0)
    local_patches = torch.randn(bs, local_v, hidden, device=device)
    text_embeds = torch.randn(bs, text_len, hidden, device=device)
    input_ids = torch.zeros(bs, text_len, dtype=torch.long, device=device)

    if arch == "internvl":
        num_ctx = min(local_v * world_size, text_len - 2)
        input_ids[:, 2:2 + num_ctx] = _INTERNVL_CONTEXT_ID

        model = _MinimalInternVLModel(hidden, args.num_layers).to(device)
        # Suppress the Phase 2 projection-layer warning: we wrap manually below.
        _auto_sp_logger = logging.getLogger("deepspeed.sequence.auto_sp")
        _prev_level = _auto_sp_logger.level
        _auto_sp_logger.setLevel(logging.ERROR)
        auto_wrap_model_for_sp(model, sp_group)
        _auto_sp_logger.setLevel(_prev_level)
        model.fusion = InternVLFusionAdapter(model.mm_projector, sp_group,
                                             image_token_id=_INTERNVL_CONTEXT_ID).to(device)
    else:  # qwen2vl
        num_inner = min(local_v * world_size, text_len - 3)
        input_ids[:, 1] = _QWEN2VL_START_ID
        input_ids[:, 2 + num_inner] = _QWEN2VL_END_ID

        model = _MinimalQwen2VLModel(hidden, args.num_layers).to(device)
        _auto_sp_logger = logging.getLogger("deepspeed.sequence.auto_sp")
        _prev_level = _auto_sp_logger.level
        _auto_sp_logger.setLevel(logging.ERROR)
        auto_wrap_model_for_sp(model, sp_group)
        _auto_sp_logger.setLevel(_prev_level)
        model.fusion = Qwen2VLFusionAdapter(model.multi_modal_projector,
                                            sp_group,
                                            vision_start_token_id=_QWEN2VL_START_ID,
                                            vision_end_token_id=_QWEN2VL_END_ID).to(device)

    return model, local_patches, text_embeds, input_ids


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run(arch: str, args) -> None:
    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(get_accelerator().device_name(), rank % get_accelerator().device_count())
    get_accelerator().set_device(rank % get_accelerator().device_count())

    sp_group = dist.new_group(ranks=list(range(world_size)))
    model, local_patches, text_embeds, input_ids = _build_model_and_inputs(arch, args, sp_group, device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            model(local_patches, text_embeds, input_ids)
    get_accelerator().synchronize()
    get_accelerator().reset_peak_memory_stats()

    # Timed iterations using CUDA events for accurate GPU-side measurement.
    latencies_ms = []
    with torch.no_grad():
        for _ in range(args.iters):
            t_start = get_accelerator().Event(enable_timing=True)
            t_end = get_accelerator().Event(enable_timing=True)
            t_start.record()
            model(local_patches, text_embeds, input_ids)
            t_end.record()
            get_accelerator().synchronize()
            latencies_ms.append(t_start.elapsed_time(t_end))

    peak_mem_mb = get_accelerator().max_memory_allocated() / 1024**2
    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    # tokens/s: fused sequence length approximated by seq_len (length-preserving adapters).
    throughput = (args.batch_size * args.seq_len) / (mean_ms / 1000.0)

    if rank == 0:
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  AutoSP Benchmark  arch={arch}  sp_degree={world_size}")
        print(sep)
        print(f"  batch_size      : {args.batch_size}")
        print(f"  seq_len         : {args.seq_len}")
        print(f"  visual_tokens   : {args.visual_tokens}  (local={args.visual_tokens // world_size}/rank)")
        print(f"  hidden          : {args.hidden}")
        print(f"  num_layers      : {args.num_layers}")
        print(f"  warmup / iters  : {args.warmup} / {args.iters}")
        print(f"  {'─' * 58}")
        print(f"  Latency         : {mean_ms:.2f} ± {std_ms:.2f} ms/iter")
        print(f"  Throughput      : {throughput:,.0f} tokens/s")
        print(f"  Peak GPU memory : {peak_mem_mb:.1f} MB")
        print(f"{sep}\n")

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoSP multimodal SP benchmark")
    parser.add_argument("--arch",
                        choices=["internvl", "qwen2vl"],
                        default="internvl",
                        help="Model architecture to simulate")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--visual-tokens",
                        type=int,
                        default=256,
                        help="Total visual tokens (must be divisible by --nproc_per_node)")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=2, help="Number of ViT blocks and LLM linear layers each")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    _run(args.arch, args)


if __name__ == "__main__":
    main()
