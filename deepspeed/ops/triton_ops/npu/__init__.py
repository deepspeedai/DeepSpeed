# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Ascend (NPU) variants of the triton_ops kernels.

The NPU accelerator's ``triton_ops_dir`` points here. A portable public
function carrying ``_triton.dispatch`` runs the same-named function from this
subpackage when the platform defines one; a function the subpackage does not
define (or only re-exports) keeps the portable implementation. Features
without a module here fall back wholesale to the portable module. Every
variant must stay importable without a device so the resolution never depends
on hardware being present.

The variants follow flash-linear-attention's ``triton_ascend`` backends:
tiles are budgeted from the Ascend unified buffer (``_ub``) instead of fixed
constants, and the kernel arithmetic keeps the portable modules' eager-mirroring
roundings unchanged, so outputs stay elementwise equal to eager. Features whose
kernels flash-linear-attention does not implement keep the portable Triton
kernels.
"""
