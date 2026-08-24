#!/usr/bin/env python
# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared BIRD (txt2sql) data + reward for the TRL async-GRPO runs.

Adapts the verl txt2sql recipe (``Arctic-Platform/recipes/rl/verl/txt2sql``) to TRL's async-GRPO loop:

* ``load_bird_dataset`` reads the recipe's verl parquet (produced by ``preprocess_bird.py``) and flattens the
  nested ``reward_model``/``extra_info`` structs into the top-level columns the rollout workers broadcast to the
  reward function (``prompt``, ``ground_truth``, ``db_path``, ``data_source``, ``db_id``).
* ``sql_reward`` is the TRL-batched reward: it wraps the recipe's per-sample ``bird_reward.compute_score``
  (SQLite exec-match) and returns one float per completion.

Both the Arctic rollout worker (``ArcticRolloutWorker._score``, in-thread) and the native
``AsyncRolloutWorker._score_group`` (spawned child, pickles ``reward_funcs``) call the reward as
``fn(prompts=..., completions=..., completion_ids=..., **broadcast_columns)`` -- so ``sql_reward`` must be this
module-level function (never a lambda/closure) and this module must be importable in the spawned child (the
baseline launcher puts its dir on ``PYTHONPATH``). ``compute_score`` is loaded from the recipe file by absolute
path at import time, so the child re-loads it the same way.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

# Absolute path to the recipe's reward module (not an installed package), loaded by file path.
_BIRD_REWARD_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "Arctic-Platform",
        "recipes",
        "rl",
        "verl",
        "txt2sql",
        "bird_reward.py",
    )
)


def _load_compute_score():
    if not os.path.exists(_BIRD_REWARD_PATH):
        raise FileNotFoundError(
            f"BIRD reward module not found at {_BIRD_REWARD_PATH}; expected the txt2sql recipe under Arctic-Platform."
        )
    spec = importlib.util.spec_from_file_location("bird_reward", _BIRD_REWARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.compute_score


# Loaded once per process (re-loaded in the spawned child when it re-imports this module).
compute_score = _load_compute_score()


def _completion_text(completion: Any) -> str:
    """Extract the assistant text from a TRL completion.

    Handles both conversational completions (``[{"role": "assistant", "content": "..."}]`` -- what both rollout
    workers emit) and plain string completions.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", "") or "")
        return str(last)
    if isinstance(completion, dict):
        return str(completion.get("content", "") or "")
    return str(completion or "")


def sql_reward(
    prompts: list[Any] | None = None,
    completions: list[Any] | None = None,
    completion_ids: list[Any] | None = None,
    ground_truth: list[Any] | None = None,
    db_path: list[Any] | None = None,
    data_source: list[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """BIRD SQLite exec-match reward, one score per completion.

    ``ground_truth`` (gold SQL) and ``db_path`` (SQLite file) are broadcast per-sample columns from the dataset
    row; each completion's SQL is extracted + executed and compared to the gold result set by ``compute_score``.
    """
    if completions is None:
        return []
    n = len(completions)
    gts = ground_truth if ground_truth is not None else [None] * n
    dbs = db_path if db_path is not None else [None] * n
    srcs = data_source if data_source is not None else ["bird"] * n

    scores: list[float] = []
    for i in range(n):
        text = _completion_text(completions[i])
        extra = {"db_path": dbs[i] if i < len(dbs) else None}
        out = compute_score(
            srcs[i] if i < len(srcs) else "bird",
            text,
            gts[i] if i < len(gts) else None,
            extra_info=extra,
        )
        score = out.get("score", 0.0) if isinstance(out, dict) else float(out)
        scores.append(float(score))
    return scores


def load_bird_dataset(parquet_path: str, num_prompts: int = -1):
    """Load the recipe's verl parquet and flatten it to TRL columns.

    Keeps only what the reward + rollout need: ``prompt`` (chat messages), ``ground_truth``, ``db_path``,
    ``data_source``, ``db_id``. Extra columns would be harmlessly broadcast to the reward, but dropping them
    keeps the rollout payloads small.
    """
    from datasets import load_dataset

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"BIRD parquet not found at {parquet_path}; run preprocess_bird.py (README step 4) first."
        )
    ds = load_dataset("parquet", data_files=parquet_path, split="train")

    def _flatten(row: dict) -> dict:
        reward_model = row.get("reward_model") or {}
        extra_info = row.get("extra_info") or {}
        return {
            "prompt": row["prompt"],
            "ground_truth": reward_model.get("ground_truth", ""),
            "db_path": extra_info.get("db_path", ""),
            "data_source": row.get("data_source", "bird"),
            "db_id": extra_info.get("db_id", ""),
        }

    ds = ds.map(_flatten, remove_columns=ds.column_names)
    if num_prompts and num_prompts > 0:
        ds = ds.select(range(min(num_prompts, len(ds))))
    return ds
