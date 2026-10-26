# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from . import adam
from . import adagrad
from . import lamb
from . import lion
from . import transformer
from . import fp_quantizer
from .transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig


# compatible_ops list is recreated for each launch
def _probe_compatible_ops():
    from .op_builder.all_ops import ALL_OPS
    from .op_builder.builder import probe_is_compatible

    compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
    for op_name, builder in ALL_OPS.items():
        op_compatible = probe_is_compatible(builder)
        compatible_ops[op_name] = op_compatible
        compatible_ops["deepspeed_not_implemented"] = False
    return compatible_ops


def __getattr__(name):
    # Probing runs compiler subprocesses and queries GPUs, and import deepspeed does not need the result, so it runs
    # on first access. Caching it in the module keeps one dict, so edits to it persist.
    if name == "__compatible_ops__":
        globals()["__compatible_ops__"] = _probe_compatible_ops()
        return globals()["__compatible_ops__"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
