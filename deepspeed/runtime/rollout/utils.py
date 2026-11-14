# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from contextlib import contextmanager
from dataclasses import replace

import torch


def get_repetition_penalty(sampling, module):
    if sampling.repetition_penalty is not None:
        return sampling.repetition_penalty
    penalty = getattr(getattr(module, "generation_config", None), "repetition_penalty", None)
    return penalty if isinstance(penalty, (int, float)) else 1.0


def get_eos_token_ids(tokenizer, module):
    generation_config = getattr(module, "generation_config", None)
    eos_ids = []
    for eos in (tokenizer.eos_token_id, getattr(generation_config, "eos_token_id", None)):
        for token in eos if isinstance(eos, (list, tuple)) else [eos]:
            if isinstance(token, int) and token not in eos_ids:
                eos_ids.append(token)
    return eos_ids


def get_pad_token_id(tokenizer):
    """The tokenizer's pad id, falling back to its EOS id."""
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("The tokenizer must define pad_token_id or eos_token_id")
    return pad_token_id


def left_pad_samples(samples, pad_token_id: int):
    """Stack samples' tokens and attention masks into left-padded ``[len(samples), max_len]`` tensors."""
    max_len = max(len(sample.tokens) for sample in samples)
    input_ids = torch.full((len(samples), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(samples), max_len), dtype=torch.long)
    for row, sample in enumerate(samples):
        start = max_len - len(sample.tokens)
        input_ids[row, start:] = torch.tensor(sample.tokens, dtype=torch.long)
        attention_mask[row, start:] = torch.tensor(sample.attention_mask, dtype=torch.long)
    return input_ids, attention_mask


def render_prompts(samples, tokenizer, tool_specs, chat_template_kwargs=None):
    rendered = []
    for sample in samples:
        tokens, attention_mask = sample.tokens, sample.attention_mask
        if not tokens:
            text = sample.prompt
            # A string prompt is taken as already-rendered text
            if not isinstance(text, str):
                text = tokenizer.apply_chat_template(text,
                                                     tools=tool_specs or None,
                                                     tokenize=False,
                                                     add_generation_prompt=True,
                                                     **(chat_template_kwargs or {}))
            tokens = tokenizer.encode(text, add_special_tokens=False)
            attention_mask = [1] * len(tokens)
        rendered.append(replace(sample, tokens=tokens, attention_mask=attention_mask))
    return rendered


def expand_samples(samples, n_samples_per_prompt: int):
    expanded = []
    for group, sample in enumerate(samples):
        group_index = sample.group_index if sample.group_index is not None else group
        for _ in range(n_samples_per_prompt):
            expanded.append(replace(sample, group_index=group_index))
    return expanded


def masked_lists(values, mask):
    """Each row's entries where ``mask`` is 1, as Python lists."""
    return [values[row][mask.bool()[row]].tolist() for row in range(values.shape[0])]


def tokenize_tool_responses(tool_messages, tokenizer, eos_token_ids, chat_template_kwargs=None):
    """Token ids that append ``tool_messages`` after an assistant turn ending in one of ``eos_token_ids``, without
    re-tokenizing it.

    """
    tool_calls = [{
        "type": "function",
        "function": {
            "name": message["name"],
            "arguments": {}
        }
    } for message in tool_messages]
    # Non-empty reasoning makes Qwen3-style templates render this message the
    # same way whether or not it is the last one.
    dummy_assistant = {"role": "assistant", "content": "", "reasoning_content": " ", "tool_calls": tool_calls}
    base = [{"role": "user", "content": "dummy"}, dummy_assistant]
    kwargs = chat_template_kwargs or {}
    without = tokenizer.apply_chat_template(base, tokenize=False, **kwargs)
    with_tools = tokenizer.apply_chat_template(base + list(tool_messages),
                                               tokenize=False,
                                               add_generation_prompt=True,
                                               **kwargs)
    if not with_tools.startswith(without):
        raise ValueError("the chat template renders earlier messages differently once tool messages follow")
    eos_tokens = tokenizer.convert_ids_to_tokens(eos_token_ids)
    end, end_of_turn = max(((without.rfind(token), token) for token in eos_tokens), default=(-1, ""))
    if end < 0:
        raise ValueError("the chat template ends the assistant turn without an EOS token")
    join_suffix = without[end + len(end_of_turn):]
    return tokenizer.encode(join_suffix + with_tools[len(without):], add_special_tokens=False)


def update_sample_from_response(sample, response, response_log_probs, tool_response, tokenizer) -> None:
    """Append one turn, its generated tokens and then its tool output, to ``sample`` in place.

    Tool output gets loss mask 0, since the model should not be trained to produce it.
    """
    turn = response + tool_response
    # The fields are reassigned, not extended, because copies made by expand_samples share their lists.
    sample.tokens = sample.tokens + turn
    sample.attention_mask = sample.attention_mask + [1] * len(turn)
    sample.loss_mask = (sample.loss_mask or []) + [1] * len(response) + [0] * len(tool_response)
    if response_log_probs is not None:
        sample.rollout_log_probs = (sample.rollout_log_probs or []) + response_log_probs + [0.0] * len(tool_response)
    sample.response = sample.response + tokenizer.decode(turn, skip_special_tokens=True)
    sample.response_length = sample.response_length + len(turn)


@contextmanager
def full_vocabulary_logits(module):
    """Have AutoTP's vocabulary-parallel output heads gather their logits while the rollout decodes.

    With a tied output head, AutoTP shards the head by vocabulary and keeps each rank's logits local for the
    vocabulary-parallel training loss. Token selection needs every rank to see the whole vocabulary, so the heads
    gather for the duration of the rollout and go back to rank-local logits for training.
    """
    heads = [submodule for submodule in module.modules() if getattr(submodule, "is_vocab_parallel_lm_head", False)]
    previous = [head.gather_output for head in heads]
    for head in heads:
        head.gather_output = True
    try:
        yield
    finally:
        for head, gather_output in zip(heads, previous):
            head.gather_output = gather_output
