# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Frozen teacher: two-phase forward with CPU-cached logits.

The trainer runs each step in two phases:

  1. **Teacher phase.** Forward over the prompt+response. The full ``[B, T, V]``
     logit tensor is moved off the GPU into a :class:`TeacherLogitCache` so that
     teacher weight buffers can be released before the student backward pass.
  2. **Student phase.** Forward + backward on the student. The distillation
     loss pulls teacher logits back to GPU **one sequence chunk at a time** via
     :meth:`TeacherLogitCache.chunk_to_device`, so peak GPU memory for teacher
     data is only ``[B, chunk, V]``.

This module deliberately lazy-imports ``deepspeed`` and ``transformers`` so
that the pure data-handling pieces (``TeacherLogitCache`` and the streamed
loss in :mod:`opsd.losses`) remain importable in CPU-only unit tests that do
not have a working DeepSpeed launcher.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# ``opsd.config`` is pure-Python (no distributed imports), so we can import it
# at module load time without pulling in DeepSpeed.
from opsd.config import TeacherConfig


@dataclass
class TeacherLogitCache:
    """CPU-resident teacher logits with on-demand chunk fetch.

    Stored in low precision (default ``bfloat16``) to halve host memory; the
    consumer in :mod:`opsd.losses` promotes back to fp32 inside the divergence
    so the KD math stays well-conditioned.
    """

    cpu_logits: torch.Tensor  # [B, T, V]

    def __post_init__(self) -> None:
        if self.cpu_logits.dim() != 3:
            raise ValueError(f"cpu_logits must be 3-D [B, T, V]; got shape "
                             f"{tuple(self.cpu_logits.shape)}")
        if self.cpu_logits.device.type != "cpu":
            raise ValueError(f"cpu_logits must live on CPU; got device "
                             f"{self.cpu_logits.device}")

    @classmethod
    def from_gpu_logits(cls, logits: torch.Tensor, store_dtype: torch.dtype = torch.bfloat16) -> "TeacherLogitCache":
        """Detach + downcast + move to (pinned) host memory.

        ``non_blocking=True`` lets the copy overlap with the next CUDA op when
        the destination is pinned; we try to pin and fall back silently if the
        host doesn't support it (e.g. CPU-only test environments).
        """
        downcast = logits.detach().to(dtype=store_dtype)
        try:
            host = torch.empty(downcast.shape, dtype=store_dtype, pin_memory=True)
            host.copy_(downcast, non_blocking=True)
        except RuntimeError:
            host = downcast.cpu()
        return cls(cpu_logits=host)

    @property
    def shape(self) -> Tuple[int, int, int]:
        s = self.cpu_logits.shape
        return (int(s[0]), int(s[1]), int(s[2]))

    @property
    def dtype(self) -> torch.dtype:
        return self.cpu_logits.dtype

    def chunk_to_device(self,
                        start: int,
                        end: int,
                        device: torch.device,
                        dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Slice ``[:, start:end, :]`` and stage it on ``device``.

        ``dtype`` is the dtype on the destination; if ``None``, the stored
        dtype is preserved.
        """
        _, T, _ = self.shape
        if not (0 <= start < end <= T):
            raise ValueError(f"chunk bounds [{start}, {end}) invalid for T={T}")
        chunk = self.cpu_logits[:, start:end]
        out = chunk.to(device=device, dtype=dtype if dtype is not None else chunk.dtype, non_blocking=True)
        return out

    def free(self) -> None:
        """Drop the underlying buffer so a step's teacher cache can be GC'd
        before the next teacher forward."""
        self.cpu_logits = torch.empty(0)


_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype {name!r}; choose from {sorted(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


class TeacherWrapper:
    """Frozen teacher.

    Two modes depending on ``cfg.offload_to_cpu``:

      * ``offload_to_cpu=False`` — load the teacher with HF's standard
        ``from_pretrained`` and pin it on the local accelerator device. The
        whole teacher lives in GPU memory; simplest path and what to use when
        the teacher fits.

      * ``offload_to_cpu=True`` — wrap the loaded model with
        ``deepspeed.initialize`` using a ZeRO-3 config with
        ``offload_param.device='cpu'``. The optimizer slot is unused (no
        trainable params) but ZeRO-3 gives us per-forward parameter gather
        / release and keeps weights on the host between forwards. This is the
        path to use when the teacher would otherwise not fit alongside the
        student.

    Both paths load the full checkpoint on each rank before DeepSpeed (if
    used) partitions; we intentionally do **not** wrap ``from_pretrained``
    in ``deepspeed.zero.Init()`` because HF's loader partitions
    ``low_cpu_mem_usage`` params to zero-width shards before the checkpoint
    can fill them, which surfaces as a "size mismatch" load error.
    """

    def __init__(self, cfg: TeacherConfig, world_size: int):
        from deepspeed.accelerator import get_accelerator
        from transformers import AutoModelForCausalLM

        self.cfg = cfg
        dtype = _resolve_dtype(cfg.dtype)
        device = get_accelerator().current_device_name()

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        if cfg.offload_to_cpu:
            import deepspeed

            ds_config = {
                "train_micro_batch_size_per_gpu": 1,
                "bf16": {
                    "enabled": dtype is torch.bfloat16
                },
                "fp16": {
                    "enabled": dtype is torch.float16
                },
                "zero_optimization": {
                    "stage": 3,
                    "offload_param": {
                        "device": "cpu"
                    },
                },
            }
            engine, *_ = deepspeed.initialize(model=model, config=ds_config)
            self._callable = engine
            self._uses_ds = True
        else:
            model.to(device)
            self._callable = model
            self._uses_ds = False

    @torch.no_grad()
    def forward_to_cache(self,
                         input_ids: torch.Tensor,
                         attention_mask: torch.Tensor,
                         store_dtype: torch.dtype = torch.bfloat16) -> TeacherLogitCache:
        """Run teacher forward and stage logits onto the host."""
        outputs = self._callable(input_ids=input_ids, attention_mask=attention_mask)
        return TeacherLogitCache.from_gpu_logits(outputs.logits, store_dtype=store_dtype)
