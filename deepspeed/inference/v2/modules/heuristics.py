# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from ..config_v2 import RaggedInferenceEngineConfig
from ..inference_utils import NormTypeEnum

from .module_registry import ConfigBundle
from ..modules.configs import (
    DSEmbeddingsConfig,
    DSLinearConfig,
    DSMoEConfig,
    DSNormConfig,
    DSSelfAttentionConfig,
    DSUnembedConfig,
)
from ..modules.interfaces import (
    DSEmbeddingBase,
    DSEmbeddingRegistry,
    DSLinearBase,
    DSLinearRegistry,
    DSMoEBase,
    DSMoERegistry,
    DSPostNormBase,
    DSPostNormRegistry,
    DSPreNormBase,
    DSPreNormRegistry,
    DSSelfAttentionBase,
    DSSelfAttentionRegistry,
    DSUnembedBase,
    DSUnembedRegistry,
)


def instantiate_attention(attention_config: DSSelfAttentionConfig,
                          engine_config: RaggedInferenceEngineConfig) -> DSSelfAttentionBase:
    """
    Choose an appropriate attention implementation based on the given configurations. This
    method is currently a stub, but as more implementations may be developed  we can centralize
    the logic for choosing between them here.

    Arguments:
        attention_config (DSSelfAttentionConfig): Configuration for the attention module.
        engine_config (RaggedInferenceEngineConfig): Configuration for the inference engine.

    Returns:
        An attention module implementing the given configuration.
    """

    # Currently, we only have one implementation, so we just return it.
    config = ConfigBundle(name="dense_blocked_attention", config=attention_config)
    return DSSelfAttentionRegistry.instantiate_config(config)


def instantiate_embed(embed_config: DSEmbeddingsConfig, engine_config: RaggedInferenceEngineConfig) -> DSEmbeddingBase:
    """
    Choose an appropriate embedding implementation based on the given configurations. This
    method is currently a stub, but as more implementations may be developed  we can centralize
    the logic for choosing between them here.

    Arguments:
        embed_config (DSEmbeddingsConfig): Configuration for the embedding module.
        engine_config (RaggedInferenceEngineConfig): Configuration for the inference engine.

    Returns:
        An embedding module implementing the given configuration.
    """

    # Currently, we only have one implementation, so we just return it.
    config = ConfigBundle(name="ragged_embedding", config=embed_config)
    return DSEmbeddingRegistry.instantiate_config(config)


def instantiate_linear(linear_config: DSLinearConfig, engine_config: RaggedInferenceEngineConfig) -> DSLinearBase:
    """
    Choose an appropriate linear implementation based on the given configurations. This
    method is currently a stub, but as more implementations may be developed  we can centralize
    the logic for choosing between them here.

    Arguments:
        linear_config (DSLinearConfig): Configuration for the linear module.
        engine_config (RaggedInferenceEngineConfig): Configuration for the inference engine.

    Returns:
        A linear module implementing the given configuration.
    """

    quantization_mode = engine_config.quantization.quantization_mode
    if quantization_mode is None:
        config = ConfigBundle(name="blas_fp_linear", config=linear_config)
    else:
        # Currently, we only support ``quantized_wf6af16_linear``.
        if quantization_mode == "wf6af16":
            config = ConfigBundle(name="quantized_wf6af16_linear", config=linear_config)
        else:
            raise ValueError(f"Unsupported quantization mode: {quantization_mode}")
    return DSLinearRegistry.instantiate_config(config)


def instantiate_moe(moe_config: DSMoEConfig, engine_config: RaggedInferenceEngineConfig) -> DSMoEBase:
    """
    Choose an appropriate MoE implementation based on the given configurations. This
    method is currently a stub, but as more implementations may be developed  we can centralize
    the logic for choosing between them here.

    Arguments:
        moe_config (DSMoEConfig): Configuration for the MoE module.
        engine_config (RaggedInferenceEngineConfig): Configuration for the inference engine.

    Returns:
        A MoE module implementing the given configuration.
    """

    moe_type = "cutlass_multi_gemm_moe"

    if moe_type == "cutlass_multi_gemm_moe":
        # TODO: Get this off an engine config
        implementation_config = {
            "weight_dtype": moe_config.input_dtype,
        }

    # Currently, we only have one implementation, so we just return it.
    config = ConfigBundle(name="cutlass_multi_gemm_moe",
                          config=moe_config,
                          implementation_config=implementation_config)
    return DSMoERegistry.instantiate_config(config)


def instantiate_post_norm(norm_config: DSNormConfig, engine_config: RaggedInferenceEngineConfig) -> DSPostNormBase:
    """
    Choose an appropriate post-norm implementation based on the given configurations. This
    method is currently a stub, but as more implementations may be developed  we can centralize
    the logic for choosing between them here.

    Arguments:
        norm_config (DSNormConfig): Configuration for the post-norm module.
        engine_config (RaggedInferenceEngineConfig): Configuration for the inference engine.

    Returns:
        A post-norm module implementing the given configuration.
    """

    # Currently, we only have one implementation, so we just return it.
    config = ConfigBundle(name="cuda_post_ln", config=norm_config)
    return DSPostNormRegistry.instantiate_config(config)


def instantiate_pre_norm(norm_config: DSNormConfig, engine_config: RaggedInferenceEngineConfig) -> DSPreNormBase:
    """
    Choose an appropriate pre-norm implementation based on the given configurations. Currently,
    this will select between two CUDA implementations, one for LayerNorm and one for RMSNorm.

    Arguments:
        norm_config (DSNormConfig): Configuration for the pre-norm module.
        engine_config (RaggedInferenceEngineConfig): Configuration for the inference engine.

    Returns:
        A pre-norm module implementing the given configuration.
    """
    if NormTypeEnum(norm_config.type) == NormTypeEnum.LayerNorm:
        module_name = "cuda_pre_ln"
    elif NormTypeEnum(norm_config.type) == NormTypeEnum.RMSNorm:
        module_name = "cuda_pre_rms"

    config = ConfigBundle(name=module_name, config=norm_config)
    return DSPreNormRegistry.instantiate_config(config)


def instantiate_unembed(unembed_config: DSUnembedConfig, engine_config: RaggedInferenceEngineConfig) -> DSUnembedBase:
    """
    Choose an appropriate unembedding implementation based on the given configurations. This
    method is currently a stub, but as more implementations may be developed  we can centralize
    the logic for choosing between them here.

    Arguments:
        unembed_config (DSUnembedConfig): Configuration for the unembed module.
        engine_config (RaggedInferenceEngineConfig): Configuration for the inference engine.

    Returns:
        An unembed module implementing the given configuration.
    """

    # Currently, we only have one implementation, so we just return it.
    config = ConfigBundle(name="ragged_unembed", config=unembed_config)
    return DSUnembedRegistry.instantiate_config(config)
