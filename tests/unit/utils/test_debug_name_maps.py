# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""The debug name maps must not keep the model they snapshot alive.

``debug_extract_module_and_param_names`` runs once during ``engine.__init__`` and the maps
are only reset in ``destroy()``. Anything that swaps a submodule out afterwards -- the
expert-parallel replacement in ``_configure_expert_parallel``, kernel injection -- leaves
the replaced weights reachable from those maps for the rest of the process.
"""

import gc
import weakref

import torch.nn as nn

from deepspeed.utils.debug import (
    debug_clear_module_and_param_names,
    debug_extract_module_and_param_names,
    debug_module2name,
    debug_param2name,
    module_names,
    param_names,
)


class _Block(nn.Module):

    def __init__(self, dim=8):
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)


class _Model(nn.Module):

    def __init__(self, num_blocks=3, dim=8):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(num_blocks)])
        self.head = nn.Linear(dim, dim, bias=False)


def test_names_resolve_and_fall_back():
    model = _Model()
    debug_extract_module_and_param_names(model)

    assert debug_param2name(model.blocks[0].lin.weight) == "blocks.0.lin.weight"
    assert debug_module2name(model.blocks[0].lin) == "blocks.0.lin"
    assert debug_param2name(nn.Linear(2, 2, bias=False).weight) == "unknown"
    assert debug_module2name(nn.Identity()) == "unknown"


def test_replaced_submodule_is_released():
    model = _Model()
    debug_extract_module_and_param_names(model)

    replaced = model.blocks[0]
    alive = [weakref.ref(replaced)] + [weakref.ref(p) for p in replaced.parameters()]
    entries_before = len(param_names)

    model.blocks[0] = nn.Identity()
    del replaced
    gc.collect()

    assert all(ref() is None for ref in alive)
    assert len(param_names) < entries_before


def test_clear_and_re_extract():
    debug_extract_module_and_param_names(_Model())
    debug_clear_module_and_param_names()

    assert len(module_names) == 0
    assert len(param_names) == 0

    other = _Model(num_blocks=1)
    debug_extract_module_and_param_names(other)

    assert debug_param2name(other.head.weight) == "head.weight"


def test_a_recycled_id_is_not_a_stale_hit():
    debug_clear_module_and_param_names()
    doomed = nn.Linear(4, 4, bias=False)
    param_names[doomed.weight] = "ghost"

    del doomed
    gc.collect()

    assert debug_param2name(nn.Linear(4, 4, bias=False).weight) == "unknown"
