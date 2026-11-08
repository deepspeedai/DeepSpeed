# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from dataclasses import replace

import torch


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


def render_prompts(samples, tokenizer, tool_specs, pad_token_id: int, chat_template_kwargs=None):
    """Copies of ``samples`` with their prompts rendered to tokens and left-padded to one width.

    A string prompt is taken as already-rendered text, and a sample that already has tokens keeps them.
    """
    rendered = []
    for sample in samples:
        tokens, attention_mask = sample.tokens, sample.attention_mask
        if not tokens:
            text = sample.prompt
            if not isinstance(text, str):
                text = tokenizer.apply_chat_template(text,
                                                     tools=tool_specs or None,
                                                     tokenize=False,
                                                     add_generation_prompt=True,
                                                     **(chat_template_kwargs or {}))
            tokens = tokenizer.encode(text, add_special_tokens=False)
            attention_mask = [1] * len(tokens)
        rendered.append(replace(sample, tokens=tokens, attention_mask=attention_mask))

    width = max(len(sample.tokens) for sample in rendered)
    for sample in rendered:
        padding = width - len(sample.tokens)
        sample.tokens = [pad_token_id] * padding + sample.tokens
        sample.attention_mask = [0] * padding + sample.attention_mask
    return rendered


def expand_samples(samples, n_samples_per_prompt: int):
    """``n_samples_per_prompt`` copies of each sample, sharing its ``group_index``."""
    expanded = []
    for group, sample in enumerate(samples):
        group_index = sample.group_index if sample.group_index is not None else group
        for _ in range(n_samples_per_prompt):
            expanded.append(replace(sample, group_index=group_index, index=len(expanded)))
    return expanded


def masked_lists(values, mask):
    """Each row's entries where ``mask`` is 1, as Python lists."""
    return [values[row][mask.bool()[row]].tolist() for row in range(values.shape[0])]


def tokenize_tool_responses(tool_messages, tokenizer, chat_template_kwargs=None):
    """Token ids that append ``tool_messages`` after an assistant turn ending in EOS, without re-tokenizing it.

    ``chat_template_kwargs`` must match the prompt's, since they can change the generation prompt that ends the
    tool output.
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
    # Generation stops at the end-of-turn token, so the template's text after it (a newline for Qwen) comes first.
    end_of_turn = tokenizer.eos_token
    join_suffix = without[without.rfind(end_of_turn) + len(end_of_turn):]
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
