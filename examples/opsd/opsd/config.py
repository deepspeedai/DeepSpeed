# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Configuration dataclasses for OPSD training.

A single :class:`OPSDConfig` is loaded from a JSON file (see ``configs/`` for
examples) and threaded through the rest of the pipeline. We use plain
dataclasses instead of Hydra/pydantic to match the rest of the DeepSpeed
example apps and to keep the dependency surface minimal.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class StudentConfig:
    model_name_or_path: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    # Architecture key used to look up the weight bridge for vLLM rollout. If
    # unset, the trainer will infer it from the HF config's ``model_type``.
    arch: Optional[str] = None


@dataclass
class TeacherConfig:
    model_name_or_path: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    # Keep teacher params on CPU and gather per-forward via ZeRO-3. Saves GPU
    # memory at the cost of host<->device transfer each step.
    offload_to_cpu: bool = True


@dataclass
class RolloutConfig:
    # "hybrid_engine" | "vllm"
    engine: str = "hybrid_engine"

    # Generation knobs (apply to either engine)
    max_prompt_length: int = 1024
    max_response_length: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    n_samples_per_prompt: int = 1

    # vLLM-specific. ``gpus`` is the disjoint set of CUDA device indices vLLM
    # may use; the training ranks must not overlap with these. If None, the
    # trainer will refuse to start in vllm mode.
    gpus: Optional[List[int]] = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    vllm_dtype: str = "bfloat16"
    # Push student weights into vLLM every N optimizer steps. Larger values
    # trade staleness for throughput.
    weight_sync_interval: int = 1
    # Pinned vLLM version known to expose the worker APIs we rely on.
    vllm_min_version: str = "0.6.4"
    # Skip CUDA-graph capture at vLLM startup. Saves several minutes of
    # one-time compilation (worth it for smoke tests / short-lived runs);
    # leave False for steady-state throughput.
    vllm_enforce_eager: bool = False


@dataclass
class DistillationConfig:
    # "forward_kl" | "reverse_kl" | "jsd"
    loss_type: str = "reverse_kl"
    temperature: float = 1.0
    # Chunk size along the sequence dimension for the per-token divergence.
    # Bounds peak memory: full [B, T, V] is never materialized at once when
    # T > chunk_size.
    chunk_size: int = 512


@dataclass
class TrainingConfig:
    train_batch_size: int = 8
    micro_batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    num_train_epochs: int = 1
    max_steps: int = -1
    warmup_steps: int = 0
    save_steps: int = 500
    logging_steps: int = 10
    save_dir: str = "./opsd_ckpt"
    seed: int = 42


@dataclass
class DataConfig:
    path: str = ""
    prompt_field: str = "prompt"
    # Optional HF chat template override; if None we use the student tokenizer's
    # default.
    chat_template: Optional[str] = None
    shuffle: bool = True


@dataclass
class OPSDConfig:
    student: StudentConfig
    teacher: TeacherConfig
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    # Path to the DeepSpeed JSON config used for ``deepspeed.initialize`` on the
    # student. Kept as a separate file because it has its own schema owned by
    # DeepSpeed.
    deepspeed_config: str = ""

    @classmethod
    def from_json(cls, path: str) -> "OPSDConfig":
        with open(path, "r") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "OPSDConfig":
        return cls(
            student=StudentConfig(**raw["student"]),
            teacher=TeacherConfig(**raw["teacher"]),
            rollout=RolloutConfig(**raw.get("rollout", {})),
            distillation=DistillationConfig(**raw.get("distillation", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
            deepspeed_config=raw.get("deepspeed_config", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.distillation.loss_type not in ("forward_kl", "reverse_kl", "jsd"):
            raise ValueError(f"Unknown loss_type {self.distillation.loss_type!r}")
        if self.rollout.engine not in ("hybrid_engine", "vllm"):
            raise ValueError(f"Unknown rollout engine {self.rollout.engine!r}")
        # rollout.gpus may be left empty for the "shared" topology where vLLM
        # runs in-process on the same GPU as training rank 0; populated for
        # the "disjoint" topology where it runs on a separate set of devices.
        if self.distillation.chunk_size <= 0:
            raise ValueError("distillation.chunk_size must be positive")
