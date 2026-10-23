# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Ascend (NPU) variants of the triton_ops kernels.

The NPU accelerator's ``triton_ops_dir`` points here. Each module in this
subpackage overrides one portable triton_ops feature for Ascend; features
without a module here fall back to the portable implementation. Every variant
must stay importable without a device so the resolution never depends on
hardware being present.
"""
