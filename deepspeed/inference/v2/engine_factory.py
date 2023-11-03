# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import logging
from typing import Any

from .engine_v2 import InferenceEngineV2
from .config_v2 import RaggedInferenceEngineConfig
from .checkpoint import HuggingFaceCheckpointEngine
from .logging import inference_logger


def build_hf_engine(path: str,
                    engine_config: RaggedInferenceEngineConfig,
                    debug_level: int = logging.INFO,
                    random_weights_config: Any = None,
                    fill_random: bool = False) -> InferenceEngineV2:
    """
    Build an InferenceV2 engine for HuggingFace models.
    """
    # Set up logging
    inference_logger(level=debug_level)

    # get HF checkpoint engine
    checkpoint_engine = HuggingFaceCheckpointEngine(path)

    # get model config from HF AutoConfig
    model_config = checkpoint_engine.model_config

    # get the policy
    # TODO: generalize this to other models
    if model_config.model_type == "opt":
        from .model_implementations.opt.policy import OPTPolicy
        policy = OPTPolicy(checkpoint_engine, model_config)
    elif model_config.model_type == "llama":
        from .model_implementations.llama_v2.llama_v2_policy import Llama2Policy
        policy = Llama2Policy(checkpoint_engine, model_config)
    elif model_config.model_type == "mistral":
        from .model_implementations.mistral.policy import MistralPolicy
        policy = MistralPolicy(checkpoint_engine, model_config)
    else:
        raise ValueError(f"Unsupported model type {model_config.model_type}")

    return InferenceEngineV2(policy, engine_config)
