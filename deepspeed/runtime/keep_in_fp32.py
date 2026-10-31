# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Keep the tensors a model names for fp32 in fp32 while the rest of the model trains in bf16 or fp16.

Hugging Face transformers models name these tensors in two class attributes:
``_keep_in_fp32_modules_strict`` (kept in fp32 under bf16 and fp16) and ``_keep_in_fp32_modules``
(kept in fp32 under fp16 only). transformers honors both when it loads a model by itself. Under ZeRO-3,
``zero.Init`` turns every floating tensor created in ``__init__`` into the training dtype, and
transformers' ZeRO-3 loader then copies the checkpoint into those tensors, so the lists were lost.

The main case is the MoE routing bias ``e_score_correction_bias`` (DeepSeek-V3, GLM-4.5, GLM-5 and others),
a buffer. In GLM-5.2 its values lie near 5 to 20 and differ from each other by less than 0.01, while bf16
values are 0.03 to 0.125 apart there, so bf16 leaves 3 to 16 distinct values for 256 experts and changes
which experts tokens are sent to.

``data_types.keep_in_fp32_modules`` selects the names:

* ``"auto"`` (default): the model's own transformers lists, if it has them.
* a list of patterns, matched the way transformers matches them: ``*`` stands for any characters,
  and a pattern may match anywhere in the full parameter or buffer name.
* ``[]``: keep nothing in fp32 (the behavior before this option existed).

Buffers are kept in fp32 in every configuration. Parameters are kept in fp32 only under ZeRO-3 without
parameter offload, the one optimizer that holds partitions of more than one dtype; DeepSeek-V4's
hyper-connection weights, attention sinks and norms, and gpt-oss's attention sinks, are examples.
"""

import re

import torch

KEEP_IN_FP32_AUTO = "auto"


def keep_in_fp32_pattern(module, setting, dtype):
    """Compiled pattern for the tensor names of ``module`` to keep in fp32, or None when there are none.

    ``dtype`` is the training dtype. Under fp32 training nothing needs keeping.
    """
    if dtype not in (torch.float16, torch.bfloat16):
        return None
    if setting is None or setting == KEEP_IN_FP32_AUTO:
        patterns = set()
        for module_name, child in module.named_modules():
            child_names = set(getattr(child, "_keep_in_fp32_modules_strict", None) or [])
            if dtype == torch.float16:
                child_names |= set(getattr(child, "_keep_in_fp32_modules", None) or [])
            for name in child_names:
                pattern = name.replace("*", ".*")
                if module_name:
                    pattern = rf"^{re.escape(module_name)}\..*{pattern}"
                patterns.add(pattern)
    else:
        patterns = {name.replace("*", ".*") for name in setting}
    if not patterns:
        return None
    # The same rule as transformers' core_model_loading.build_glob_alternation followed by re.search.
    return re.compile("|".join(sorted(patterns)))


def tensors_to_keep_in_fp32(module, pattern):
    """(full name, tensor, is_buffer) for every floating parameter and buffer whose full name matches."""
    matches = []
    for module_name, owner in module.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for name, param in owner.named_parameters(recurse=False):
            if param.is_floating_point() and pattern.search(prefix + name):
                matches.append((prefix + name, param, False))
        for name, buf in owner.named_buffers(recurse=False):
            if buf is not None and buf.is_floating_point() and pattern.search(prefix + name):
                matches.append((prefix + name, buf, True))
    return matches


def _stored_dtype(tensor):
    # A ZeRO-3 parameter keeps its values in its partition; tensor.data is a placeholder.
    partition = getattr(tensor, "ds_tensor", None)
    return partition.dtype if partition is not None else tensor.dtype


def keep_tensors_in_fp32(module, pattern, include_params=True):
    """Convert the matching parameters and buffers of ``module`` to fp32 in place.

    A ZeRO-3 parameter is converted in its partition, so a later gather returns fp32 and a later
    checkpoint load writes exact fp32 values into it. With ``include_params=False`` only buffers are
    converted. Returns (parameters converted, buffers converted).
    """
    num_params, num_buffers = 0, 0
    for _, tensor, is_buffer in tensors_to_keep_in_fp32(module, pattern):
        if not is_buffer and not include_params:
            continue
        if _stored_dtype(tensor) == torch.float32:
            continue
        partition = getattr(tensor, "ds_tensor", None)
        if partition is not None:
            partition.data = partition.data.float()
        # For a ZeRO-3 parameter this converts the empty placeholder, or the full tensor when the
        # parameter is currently gathered, so the next gather and the next free keep fp32.
        tensor.data = tensor.data.float()
        if is_buffer:
            num_buffers += 1
        else:
            num_params += 1
    return num_params, num_buffers
