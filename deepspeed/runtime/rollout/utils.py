# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from dataclasses import replace

import torch


def left_pad_samples(samples, pad_token_id: int):
    """Stack samples' ``tokens`` and ``attention_mask`` into left-padded tensors.

    Every row is padded on the left to the longest sample, so generation
    appends right after each sample's real tokens.

    Returns:
        ``(input_ids, attention_mask)``, both ``[len(samples), max_len]`` long tensors.
    """
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

    A message-list prompt is rendered with the chat template, the tool specs,
    ``chat_template_kwargs`` and the generation prompt; a string prompt is
    taken as already-rendered text, as miles does. A sample that already has tokens (e.g. continuing a
    rollout) keeps them. Padding gets attention mask 0.
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
    """``n_samples_per_prompt`` copies of each sample, one per response to generate.

    Copies of one sample share its ``group_index`` (its position in ``samples``
    when unset); ``index`` numbers the copies in order.
    """
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
    """Token ids that append ``tool_messages`` after an assistant turn ending in EOS.

    The chat template renders a fixed dummy conversation with and without the
    tool messages and the difference is kept, so real assistant turns are
    never re-tokenized. The template's text after the end-of-turn token (a
    newline for Qwen) is prepended because generation stops at that token.
    ``chat_template_kwargs`` must match the prompt's, since they can change
    the generation prompt that ends the tool output (Qwen3's empty think block
    when thinking is off).
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
    end_of_turn = tokenizer.eos_token
    join_suffix = without[without.rfind(end_of_turn) + len(end_of_turn):]
    return tokenizer.encode(join_suffix + with_tools[len(without):], add_special_tokens=False)


def update_sample_from_response(sample, response, response_log_probs, tool_response, tokenizer) -> None:
    """Append one turn to ``sample`` in place: generated tokens, then tool output.

    Generated tokens get loss mask 1 and their rollout log-probs; tool output
    gets loss mask 0 and log-prob 0.0, since it is environment text the model
    should not be trained to produce. ``response_log_probs=None`` leaves
    ``rollout_log_probs`` unchanged. Fields are reassigned rather than
    extended because copies made by ``expand_samples`` share list objects
    with their source sample.
    """
    turn = response + tool_response
    sample.tokens = sample.tokens + turn
    sample.attention_mask = sample.attention_mask + [1] * len(turn)
    sample.loss_mask = (sample.loss_mask or []) + [1] * len(response) + [0] * len(tool_response)
    if response_log_probs is not None:
        sample.rollout_log_probs = (sample.rollout_log_probs or []) + response_log_probs + [0.0] * len(tool_response)
    sample.response = sample.response + tokenizer.decode(turn, skip_special_tokens=True)
    sample.response_length = sample.response_length + len(turn)
