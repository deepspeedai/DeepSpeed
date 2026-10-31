# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""TEMPORARY measurement-only patch: DS_SEGKI_SKIP leave-one-out gates.

Applied on the GPU box only; revert with `git checkout -- .`. Not for the PR.
"""

import os

SKIP = "__import__('os').environ.get('DS_SEGKI_SKIP','').split(',')"

P = 'deepspeed/module_inject/segment_ki.py'
s = open(P).read()

old = '''    segments = find_glu_segments(model)
    replaced = 0
    for seg in segments:'''
new = '''    _skip = __import__('os').environ.get('DS_SEGKI_SKIP', '').split(',')
    segments = [] if 'glu' in _skip else find_glu_segments(model)
    replaced = 0
    for seg in segments:'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''    gdn_segments = find_gdn_segments(model)
    gdn_replaced = sum(1 for seg in gdn_segments if _install_gdn_segment(seg))'''
new = '''    gdn_segments = [] if 'gdn' in _skip else find_gdn_segments(model)
    gdn_replaced = sum(1 for seg in gdn_segments if _install_gdn_segment(seg))'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''def install_fused_norm(model: torch.nn.Module, kernel_op) -> int:'''
new = '''def install_fused_norm(model: torch.nn.Module, kernel_op) -> int:
    if 'norm' in __import__('os').environ.get('DS_SEGKI_SKIP', '').split(','):
        return 0'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''def install_decode_attention(model: torch.nn.Module, write_pos: torch.Tensor, kernel_op) -> int:'''
new = '''def install_decode_attention(model: torch.nn.Module, write_pos: torch.Tensor, kernel_op) -> int:
    if 'attn' in __import__('os').environ.get('DS_SEGKI_SKIP', '').split(','):
        return 0'''
assert s.count(old) == 1
s = s.replace(old, new)
open(P, 'w').write(s)

P2 = 'deepspeed/runtime/rollout/hybrid_engine_rollout.py'
t = open(P2).read()
old = '''                if (graph_op is not None and batch_size == 1 and static_logits.dtype is torch.bfloat16
                        and get_accelerator().on_accelerator(static_logits)):'''
new = '''                if (graph_op is not None and batch_size == 1 and static_logits.dtype is torch.bfloat16
                        and get_accelerator().on_accelerator(static_logits)
                        and 'step' not in __import__('os').environ.get('DS_SEGKI_SKIP', '').split(',')):'''
assert t.count(old) == 1
t = t.replace(old, new)
open(P2, 'w').write(t)
print("ablation gates applied")
