# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

try:
    from op_builder.pin_memory import PinMemoryBuilder
except ImportError:
    from deepspeed.ops.op_builder.pin_memory import PinMemoryBuilder

__all__ = ["PinMemoryBuilder"]
