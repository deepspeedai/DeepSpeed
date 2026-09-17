# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import inspect
import textwrap
import torch._functorch.partitioners as _partitioners

# The custom should_ban_recomputation to splice into solve_min_cut.
# All names it references (aten, operator, config, op_types, min_cut_options,
# is_materialized_backwards, get_aten_target, _size_of, fx, torch,
# CheckpointPolicy) are either module-level in torch._functorch.partitioners
# or local variables already in scope when this function executes inside
# solve_min_cut.
_CUSTOM_SHOULD_BAN = """\
def should_ban_recomputation(node):
    \"\"\"Sequence-aware recomputation banning logic\"\"\"
    if node.op != "call_function":
        return False
    if node.target == operator.getitem:
        return False
    if node.meta.get("recompute", None) == CheckpointPolicy.MUST_SAVE:
        return True
    if config.recompute_views and op_types.is_view(node):
        return False
    if node.target in [aten.lift_fresh_copy.default, aten.lift_fresh.default]:
        return False

    must_save_set = [
        aten.convolution,
        aten.convolution_backward,
        aten._scaled_dot_product_flash_attention,
        aten._scaled_dot_product_efficient_attention,
        aten._flash_attention_forward,
        aten._efficient_attention_forward,
        aten.upsample_bilinear2d,
        aten.native_dropout,
        aten.rand_like,
        aten.randn_like,
    ]

    if get_aten_target(node) in must_save_set:
        return True

    def heuristic(node):
        if "val" in node.meta:
            if isinstance(node.meta["val"], torch.Tensor) and node.meta["val"].dim() >= 2:
                return node.meta["val"].shape[1] >= 4096
        return False

    if min_cut_options.ban_if_not_in_allowlist:
        if not op_types.is_recomputable(node):
            return False

    if min_cut_options.ban_if_materialized_backward and is_materialized_backwards(node):
        if heuristic(node):
            return False
        return True

    if node.dist_from_bw < 1000 and node.dist_from_bw > config.max_dist_from_bw:
        return False

    if min_cut_options.ban_if_reduction:
        input_tensors_size = sum(
            _size_of(i) for i in node.args if isinstance(i, fx.Node)
        )
        output_size = _size_of(node)
        return output_size * 4 < input_tensors_size
    return False
"""


def register_long_context_checkpointing():
    """Splice the custom should_ban_recomputation into solve_min_cut.

    Uses inspect.getsource to extract solve_min_cut's source, replaces the
    original should_ban_recomputation with _CUSTOM_SHOULD_BAN, then execs the
    result directly in _partitioners.__dict__.

    The exec'd function's __globals__ is the real partitioners module dict, so
    every other nested function (is_fusible, is_materialized_backwards,
    can_fuse_into_*, etc.) and every local/closure variable (op_types,
    min_cut_options, node_info, config, …) is exactly as in the original —
    nothing else changes.

    Backward compatible: if solve_min_cut gains new heuristics in a future
    PyTorch version the exec automatically picks them up; only
    _CUSTOM_SHOULD_BAN needs to stay in sync with any changes to the
    original should_ban_recomputation signature/contract.
    """
    src = inspect.getsource(_partitioners.solve_min_cut)
    lines = src.split('\n')

    # Locate the original should_ban_recomputation and the function after it.
    start = next(i for i, l in enumerate(lines) if l.startswith('    def should_ban_recomputation('))
    end = next(i for i, l in enumerate(lines) if i > start and l.startswith('    def '))

    # Indent the replacement to the nesting level inside solve_min_cut (4 spaces).
    replacement = textwrap.indent(_CUSTOM_SHOULD_BAN, '    ')

    new_src = '\n'.join(lines[:start]) + '\n' + replacement + '\n'.join(lines[end:])
    exec(new_src, _partitioners.__dict__)  # redefines _partitioners.solve_min_cut
