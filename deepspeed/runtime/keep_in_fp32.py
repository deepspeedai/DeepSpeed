# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Keep the buffers a model names for fp32 in fp32 while the rest of the model trains in bf16 or fp16.

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
  and a pattern may match anywhere in the full buffer name.
* ``[]``: keep nothing in fp32 (the behavior before this option existed).

Only buffers are kept in fp32 here; parameters named by the lists still follow the training dtype.
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


def buffers_to_keep_in_fp32(module, pattern):
    """The floating buffers of ``module`` whose full names match ``pattern``."""
    matches = []
    for module_name, owner in module.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for name, buf in owner.named_buffers(recurse=False):
            if buf is not None and buf.is_floating_point() and pattern.search(prefix + name):
                matches.append(buf)
    return matches


def keep_buffers_in_fp32(module, pattern):
    """Convert the matching buffers of ``module`` to fp32 in place. Returns how many were converted."""
    converted = 0
    for buf in buffers_to_keep_in_fp32(module, pattern):
        if buf.dtype != torch.float32:
            buf.data = buf.data.float()
            converted += 1
    return converted
