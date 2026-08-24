#!/usr/bin/env python
# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the BIRD task adapter (bird_task.sql_reward): SQLite exec-match wiring + TRL batching.

Builds a tiny SQLite DB so the reward path (extract SQL -> execute -> compare to gold result set) is exercised
end-to-end without any model or GPU.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile

import bird_task


def _make_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO t (id, name) VALUES (?, ?)", [(1, "a"), (2, "b"), (3, "c")])
    conn.commit()
    conn.close()


def _assistant(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


def test_exec_match_wrong_and_missing():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "toy.sqlite")
        _make_db(db)
        gold = "SELECT id FROM t ORDER BY id"

        completions = [
            _assistant("<think>reason</think>\n```sql\nSELECT id FROM t ORDER BY id\n```"),  # exec-match -> 1.0
            _assistant("<think>reason</think>\n```sql\nSELECT name FROM t ORDER BY id\n```"),  # executes, wrong -> 0.1
            _assistant("no sql here at all"),  # nothing extracted -> 0.0
        ]
        n = len(completions)
        scores = bird_task.sql_reward(
            prompts=[[{"role": "user", "content": "q"}]] * n,
            completions=completions,
            completion_ids=[[] for _ in range(n)],
            ground_truth=[gold] * n,
            db_path=[db] * n,
            data_source=["bird"] * n,
        )
        assert scores[0] == 1.0, scores
        assert scores[1] == 0.1, scores
        assert scores[2] == 0.0, scores


def test_string_completion_and_defaults():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "toy.sqlite")
        _make_db(db)
        gold = "SELECT id FROM t ORDER BY id"
        # Plain-string completion (non-conversational) must also work.
        scores = bird_task.sql_reward(
            completions=["```sql\nSELECT id FROM t ORDER BY id\n```"],
            ground_truth=[gold],
            db_path=[db],
        )
        assert scores == [1.0], scores


def test_empty_completions():
    assert bird_task.sql_reward(completions=[]) == []
    assert bird_task.sql_reward() == []
