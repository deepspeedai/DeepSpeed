# Copyright (c) 2019, The Microsoft DeepSpeed Team. All rights reserved.
#
# Note: please copy webtext data to "Megatron-LM" folder, before running this script.

from .run_func_test import GPT2FuncTestCase
from .run_checkpoint_test import GPT2CheckpointTestCase, checkpoint_suite
from .run_func_test import suite
