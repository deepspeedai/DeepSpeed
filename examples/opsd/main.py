# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""OPSD training entry point.

Launch with the DeepSpeed launcher::

    deepspeed --num_gpus 8 main.py --config configs/opsd_hybrid_engine.json

The DeepSpeed launcher sets ``LOCAL_RANK``, ``RANK``, and ``WORLD_SIZE`` in
the environment; we call :func:`deepspeed.init_distributed` to take that over.
"""

import argparse
import json
import os
import random

import deepspeed
import numpy as np
import torch
from deepspeed.accelerator import get_accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from opsd.config import OPSDConfig
from opsd.data import LeftPaddedPromptCollator, PromptDataset
from opsd.rollout import build_rollout
from opsd.teacher import TeacherWrapper
from opsd.trainer import OPSDTrainer


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if get_accelerator().is_available():
        get_accelerator().manual_seed_all(seed)


def _resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _load_ds_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to OPSDConfig JSON")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = parser.parse_args()

    cfg = OPSDConfig.from_json(args.config)
    cfg.validate()
    _seed_everything(cfg.training.seed)

    deepspeed.init_distributed()

    # --- tokenizer (shared between data + rollout) -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.student.model_name_or_path,
        trust_remote_code=cfg.student.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- student model + DeepSpeed engine ----------------------------------
    student_dtype = _resolve_dtype(cfg.student.dtype)
    student_model = AutoModelForCausalLM.from_pretrained(
        cfg.student.model_name_or_path,
        torch_dtype=student_dtype,
        trust_remote_code=cfg.student.trust_remote_code,
    )

    ds_config = _load_ds_config(cfg.deepspeed_config)
    ds_config["train_micro_batch_size_per_gpu"] = cfg.training.micro_batch_size_per_gpu
    ds_config["train_batch_size"] = cfg.training.train_batch_size
    ds_config["gradient_accumulation_steps"] = cfg.training.gradient_accumulation_steps

    student_engine, *_ = deepspeed.initialize(
        model=student_model,
        model_parameters=student_model.parameters(),
        config=ds_config,
    )

    # --- frozen teacher ----------------------------------------------------
    teacher = TeacherWrapper(cfg.teacher, world_size=dist_world_size())

    # --- rollout engine ----------------------------------------------------
    rollout = build_rollout(
        cfg.rollout,
        student_engine=student_engine,
        tokenizer=tokenizer,
        student_model_path=cfg.student.model_name_or_path,
        arch=cfg.student.arch,
    )

    # --- dataloader --------------------------------------------------------
    dataset = PromptDataset(
        path=cfg.data.path,
        tokenizer=tokenizer,
        max_prompt_length=cfg.rollout.max_prompt_length,
        prompt_field=cfg.data.prompt_field,
        chat_template=cfg.data.chat_template,
    )
    collator = LeftPaddedPromptCollator(tokenizer=tokenizer, max_prompt_length=cfg.rollout.max_prompt_length)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.micro_batch_size_per_gpu,
        shuffle=cfg.data.shuffle,
        collate_fn=collator,
        drop_last=True,
    )

    OPSDTrainer(
        cfg=cfg,
        student_engine=student_engine,
        teacher=teacher,
        tokenizer=tokenizer,
        rollout=rollout,
        dataloader=loader,
    ).train()


def dist_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


if __name__ == "__main__":
    main()
