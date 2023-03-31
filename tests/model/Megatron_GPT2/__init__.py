# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Note: please copy webtext data to "Megatron-LM" folder, before running this script.
"""

from .run_func_test import GPT2FuncTestCase
from .run_checkpoint_test import GPT2CheckpointTestCase, checkpoint_suite
from .run_func_test import suite
