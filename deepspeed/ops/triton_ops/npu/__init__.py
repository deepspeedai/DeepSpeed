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
"""
