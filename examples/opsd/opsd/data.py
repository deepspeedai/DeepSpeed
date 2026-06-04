# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Prompt dataset and left-padding collator for OPSD rollouts.

The dataset reads a JSONL file with one record per line; each record must
contain a string under :attr:`DataConfig.prompt_field` (default ``"prompt"``).
If the tokenizer exposes ``apply_chat_template``, single-turn prompts are
wrapped in a user-role message with ``add_generation_prompt=True`` so the
student generates the assistant turn.

Batches are **left-padded** because causal generation requires real tokens at
    the right edge — :class:`deepspeed.runtime.rollout.RolloutRequest` and the hybrid-engine
backend both assume this layout.
"""

import json
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """Reads ``{prompt_field: str}`` records from a JSONL file."""

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        max_prompt_length: int,
        prompt_field: str = "prompt",
        chat_template: Optional[str] = None,
    ):
        self.records = self._load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompt_field = prompt_field
        self.chat_template = chat_template

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> str:
        rec = self.records[idx]
        if self.prompt_field not in rec:
            raise KeyError(f"record {idx} missing field {self.prompt_field!r}")
        text = rec[self.prompt_field]

        # If the tokenizer knows a chat template, render the prompt as a single
        # user-role turn and request the generation prompt. This matches how
        # instruction-tuned student/teacher checkpoints expect inputs.
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": text}] if isinstance(text, str) else text
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=self.chat_template,
            )
        return text


class LeftPaddedPromptCollator:
    """Tokenizes a batch of prompt strings into a left-padded tensor batch."""

    def __init__(self, tokenizer: Any, max_prompt_length: int):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if self.pad_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id; "
                             "cannot construct a padding collator")

    def __call__(self, batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        per_sample = [
            self.tokenizer(
                t,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_prompt_length,
                return_tensors="pt",
            )["input_ids"].squeeze(0) for t in batch_texts
        ]
        max_len = max(int(x.shape[0]) for x in per_sample)
        B = len(per_sample)

        prompt_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, ids in enumerate(per_sample):
            n = int(ids.shape[0])
            # left-pad: real tokens at the right edge
            prompt_ids[i, max_len - n:] = ids
            attention_mask[i, max_len - n:] = 1

        return {"prompt_ids": prompt_ids, "prompt_attention_mask": attention_mask}
