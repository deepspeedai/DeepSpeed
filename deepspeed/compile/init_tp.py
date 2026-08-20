# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from packaging.version import Version
from torch.fx import GraphModule

from deepspeed.utils.torch import required_torch_version

from .passes.tp_compile import apply_autotp, defer_collectives_to_compiler

AUTOTP_MIN_TORCH_VERSION = 2.6
BROKEN_TRANSFORMERS_MOE_VERSIONS = ("5.8.0", "5.10.1")


def _check_autotp_compatibility(model):
    if not required_torch_version(min_version=AUTOTP_MIN_TORCH_VERSION):
        raise RuntimeError(f"The AutoTP compile pass requires PyTorch >= {AUTOTP_MIN_TORCH_VERSION}, found "
                           f"{torch.__version__}.")
    _check_broken_transformers_moe(model)


def _check_broken_transformers_moe(model):
    """Reject models whose experts forward the installed transformers cannot capture in a graph."""

    try:
        import transformers
    except ImportError:
        return
    first_broken, first_fixed = BROKEN_TRANSFORMERS_MOE_VERSIONS
    if not (Version(first_broken) <= Version(transformers.__version__) < Version(first_fixed)):
        return
    experts_modules = [
        name for name, module in model.named_modules()
        if hasattr(module, "_apply_gate") and hasattr(module, "is_concatenated")
        and getattr(getattr(module, "config", None), "_experts_implementation", None) == "batched_mm"
    ]
    if experts_modules:
        raise RuntimeError(
            f"transformers {transformers.__version__} mutates the MoE routing tensor in place inside "
            "batched_mm_experts_forward (huggingface/transformers#45621, fixed by #45634), and this model "
            f"routes through it (e.g. '{experts_modules[0]}'), so the AutoTP compile pass cannot capture a "
            f"full graph. Upgrade to transformers >= {first_fixed}.")


def init_autotp(model):
    """Hand the tensor-parallel collectives of an AutoTP-partitioned model over to the compiler.

    The model is expected to have been partitioned already by the regular AutoTP path, so this only
    suppresses the module-level collectives and returns a backend that emits them as graph nodes.
    """
    _check_autotp_compatibility(model)
    defer_collectives_to_compiler(model)

    def backend_fn(gm: GraphModule, real_inputs):
        apply_autotp(gm, real_inputs)
        return torch._inductor.compile(gm, real_inputs)

    return backend_fn
