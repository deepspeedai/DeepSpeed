# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
import logging
import os
import pickle
from packaging import version

from .engine_v2 import InferenceEngineV2
from .config_v2 import RaggedInferenceEngineConfig
from .checkpoint import HuggingFaceCheckpointEngine
from .logging import inference_logger
from .model_implementations import (
    OPTPolicy,
    Llama2Policy,
    MistralPolicy,
    MixtralPolicy,
    FalconPolicy,
    PhiPolicy,
    QwenPolicy,
    Qwen2Policy,
)
from .model_implementations.inference_policy_base import POLICIES, InferenceV2Policy
from .model_implementations.flat_model_helpers import make_metadata_filename, ModelMetadata


def build_engine_from_ds_checkpoint(path: str,
                                    engine_config: RaggedInferenceEngineConfig,
                                    debug_level: int = logging.INFO) -> InferenceEngineV2:
    """
    Creates an engine from a checkpoint saved by ``InferenceEngineV2``.

    Arguments:
        path: Path to the checkpoint. This does not need to point to any files in particular,
            just the directory containing the checkpoint.
        engine_config: Engine configuration. See ``RaggedInferenceEngineConfig`` for details.
        debug_level: Logging level to use. Unless you are actively seeing issues, the recommended
            value is ``logging.INFO``.

    Returns:
        Fully initialized inference engine ready to serve queries.
    """

    inference_logger(level=debug_level)
    # Load metadata, for grabbing the policy name we'll have all ranks just check for
    # rank 0.
    metadata_filename = make_metadata_filename(path, 0, engine_config.tensor_parallel.tp_size)
    metadata = json.load(open(metadata_filename, "r"))
    metadata = ModelMetadata.parse_raw(metadata)

    # Get the policy
    try:
        policy_cls: InferenceV2Policy = POLICIES[metadata.policy]
    except KeyError:
        raise ValueError(f"Unknown policy {metadata.policy} for model {path}")

    # Load the model config
    model_config = pickle.load(open(os.path.join(path, "ds_model_config.pkl"), "rb"))
    policy = policy_cls(model_config, inf_checkpoint_path=path)

    return InferenceEngineV2(policy, engine_config)


def build_hf_engine(path: str,
                    engine_config: RaggedInferenceEngineConfig,
                    debug_level: int = logging.INFO) -> InferenceEngineV2:
    """
    Build an InferenceV2 engine for HuggingFace models. This can accept both a HuggingFace
    model name or a path to an Inference-V2 checkpoint.

    Arguments:
        path: Path to the checkpoint. This does not need to point to any files in particular,
            just the directory containing the checkpoint.
        engine_config: Engine configuration. See ``RaggedInferenceEngineConfig`` for details.
        debug_level: Logging level to use. Unless you are actively seeing issues, the recommended
            value is ``logging.INFO``.

    Returns:
        Fully initialized inference engine ready to serve queries.
    """

    if os.path.exists(os.path.join(path, "ds_model_config.pkl")):
        return build_engine_from_ds_checkpoint(path, engine_config, debug_level=debug_level)
    else:
        # Set up logging
        inference_logger(level=debug_level)
        # get HF checkpoint engine
        checkpoint_engine = HuggingFaceCheckpointEngine(path)

        # get model config from HF AutoConfig
        model_config = checkpoint_engine.model_config

        # get the policy
        # TODO: generalize this to other models
        if model_config.model_type == "opt":
            if not model_config.do_layer_norm_before:
                raise ValueError(
                    "Detected OPT-350m model. This model is not currently supported. If this is not the 350m model, please open an issue: https://github.com/microsoft/DeepSpeed-MII/issues"
                )
            policy = OPTPolicy(model_config, checkpoint_engine=checkpoint_engine)
        elif model_config.model_type == "llama":
            policy = Llama2Policy(model_config, checkpoint_engine=checkpoint_engine)
        elif model_config.model_type == "mistral":
            # Ensure we're using the correct version of transformers for mistral
            import transformers
            assert version.parse(transformers.__version__) >= version.parse("4.34.0"), \
                f"Mistral requires transformers >= 4.34.0, you have version {transformers.__version__}"
            policy = MistralPolicy(model_config, checkpoint_engine=checkpoint_engine)
        elif model_config.model_type == "mixtral":
            # Ensure we're using the correct version of transformers for mistral
            import transformers
            assert version.parse(transformers.__version__) >= version.parse("4.36.1"), \
                f"Mistral requires transformers >= 4.36.1, you have version {transformers.__version__}"
            policy = MixtralPolicy(model_config, checkpoint_engine=checkpoint_engine)
        elif model_config.model_type == "falcon":
            policy = FalconPolicy(model_config, checkpoint_engine=checkpoint_engine)
        elif model_config.model_type == "phi":
            policy = PhiPolicy(model_config, checkpoint_engine=checkpoint_engine)
        elif model_config.model_type == "qwen":
            policy = QwenPolicy(model_config, checkpoint_engine=checkpoint_engine)
        elif model_config.model_type == "qwen2":
            policy = Qwen2Policy(model_config, checkpoint_engine=checkpoint_engine)
        else:
            raise ValueError(f"Unsupported model type {model_config.model_type}")

        return InferenceEngineV2(policy, engine_config)
