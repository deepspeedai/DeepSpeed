# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Dict, Iterable, Tuple, Optional
from os import path

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import RaggedUtilsBuilder
from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from .layer_container_base import LayerContainer
from ..inference_parameter import InferenceParameter, STR_TO_DTYPE
from ..inference_utils import elem_size


def pad_to_aligned_offset(offset: int, alignment: int = 256) -> int:
    """
    Pad the provided offset to a well-aligned value.
    """
    return ((offset + alignment - 1) // alignment) * alignment


class TensorMetadata(DeepSpeedConfigModel):
    """
    A class to represent a tensor specification.
    """
    dtype: Optional[str]
    shape: Optional[Tuple[int, ...]]
    strides: Optional[Tuple[int, ...]]
    offset: int


class ParameterMetadata(DeepSpeedConfigModel):
    """
    A class to represent a parameter specification.
    """
    core_param: TensorMetadata = None
    aux_params: Dict[str, TensorMetadata] = {}


class LayerMetadata(DeepSpeedConfigModel):
    """
    A class to represent a layer specification.
    """
    params: Dict[str, ParameterMetadata] = {}


class ModelMetadata(DeepSpeedConfigModel):
    """
    A class to represent a model specification.
    """
    policy: str = ""
    layers: Dict[str, LayerMetadata] = {}


def make_param_filename(base: str, rank: int, n_ranks: int) -> str:
    """
    Make a filename for a parameter file.

    Arguments:
        rank: Rank of the file.
        n_ranks: Total number of ranks.

    Returns:
        str: Filename.
    """
    return path.join(base, f"params_rank_{rank}_of_{n_ranks}.pt")


def make_metadata_filename(base: str, rank: int, n_ranks: int) -> str:
    """
    Make a filename for a metadata file.

    Arguments:
        rank: Rank of the file.
        n_ranks: Total number of ranks.

    Returns:
        str: Filename.
    """
    return path.join(base, f"metadata_rank_{rank}_of_{n_ranks}.json")


def make_model_config_filename(base: str) -> str:
    """
    Make a filename for a model config file.

    Arguments:
        base: Base directory.

    Returns:
        str: Filename.
    """
    return path.join(base, "ds_model_config.json")


def flatten_inference_model(
    transformer_containers: Iterable[LayerContainer],
    non_transformer_container: LayerContainer,
    policy_name: str,
) -> Tuple[torch.Tensor, ModelMetadata]:
    """
    Flatten the underlying parameters into

    Arguments:
        transformer_containers: Iterable of layer containers corresponding to the transformer
            parameters.
        non_transformer_container: Layer container corresponding to the non-transformer parameters.
        policy_name: The name of the policy class (typically accessed with `type(policy).__name__`).

    Returns:
        Iterable[Any]: Flattened list of parameters.
    """
    alloc_fn = RaggedUtilsBuilder().load().allocate_view_on

    total_size = 0
    metadata = ModelMetadata(policy=policy_name)

    def process_layer(layer_container: LayerContainer, l_name: str, cur_offset: int) -> int:
        """
        Iterate over the parameters of a single container and collect metadata for the final
        flattened buffer.

        Arguments:
            layer_container: The layer container to process.
            l_name: The name of the layer container to key the metadata.
            cur_offset: The current offset into the flattened buffer.

        Captured Variables:
            metadata: The metadata object to populate.

        Returns:
            int: The updated offset into the flattened buffer.
        """
        try:
            _ = layer_container.is_populated
        except ValueError as e:
            raise ValueError(f"Layer container {l_name} is not populated.") from e

        layer_metadata = LayerMetadata()

        for p_name in layer_container.annotation_attrs:
            param = getattr(layer_container, p_name)
            param_metadata = ParameterMetadata()

            if param is None:
                param_metadata.core_param = TensorMetadata(offset=-1)
                layer_metadata.params[p_name] = param_metadata
                continue

            param_metadata.core_param = TensorMetadata(dtype=str(param.dtype),
                                                       shape=param.shape,
                                                       strides=param.stride(),
                                                       offset=cur_offset)

            cur_offset += pad_to_aligned_offset(elem_size(param.dtype) * param.numel())

            for t_name, tensor in param.aux_attrs.items():
                param_metadata.aux_params[t_name] = TensorMetadata(dtype=str(tensor.dtype),
                                                                   shape=tensor.shape,
                                                                   strides=tensor.stride(),
                                                                   offset=cur_offset)

                cur_offset += pad_to_aligned_offset(elem_size(tensor.dtype) * tensor.numel())

            layer_metadata.params[p_name] = param_metadata

        metadata.layers[l_name] = layer_metadata
        return cur_offset

    for i, layer in enumerate(transformer_containers):
        l_name = f"transformer_layer_{i}"
        total_size = process_layer(layer, l_name, total_size)

    l_name = "non_transformer"
    total_size = process_layer(non_transformer_container, l_name, total_size)

    buffer = torch.empty(total_size, dtype=torch.uint8, device=get_accelerator().current_device())

    def copy_layer(layer_container: LayerContainer, l_name: str) -> None:
        """
        Local method for copying from the layer container to the flattened buffer.

        Arguments:
            layer_container: The layer container to copy from.
            l_name: The name of the layer container to key the metadata.

        Captured Variables:
            buffer: The flattened buffer to copy into.
            metadata: The metadata object to populate.
        """
        l_metadata = metadata.layers[l_name]
        for p_name in layer_container.annotation_attrs:
            p_metadata = l_metadata.params[p_name]
            param = getattr(layer_container, p_name)

            if param is None:
                continue

            core_param = alloc_fn(param, buffer, p_metadata.core_param.offset)
            core_param.copy_(param)

            aux_params = {}

            for t_name, tensor in param.aux_attrs.items():
                t_view = alloc_fn(tensor, buffer, p_metadata.aux_params[t_name].offset)
                aux_params[t_name] = t_view
                t_view.copy_(tensor)

            setattr(layer_container, p_name, InferenceParameter.initialize(core_param, **aux_params))

    for i, layer in enumerate(transformer_containers):
        l_name = f"transformer_layer_{i}"
        copy_layer(layer, l_name)

    l_name = "non_transformer"
    copy_layer(non_transformer_container, l_name)

    return buffer, metadata


def restore_inference_model(buffer: torch.Tensor, metadata: ModelMetadata,
                            transformer_containers: Iterable[LayerContainer],
                            non_transformer_container: LayerContainer) -> None:
    """
    Restore the model from the buffer and metadata.

    Arguments:
        buffer: Buffer containing the model parameters.
        metadata: Metadata for the model.
        transformer_containers: Iterable of transformer layer containers.
        non_transformer_container: Non-transformer layer container.
    """
    alloc_fn = RaggedUtilsBuilder().load().allocate_view_like

    def restore_layer(layer_container: LayerContainer, l_name: str) -> None:
        """
        Local method for restoring a layer container from a flattened buffer. This
        only constructs views for the parameters onto the buffer. No data movement
        is performed.

        Arguments:
            layer_container: The layer container to restore.
            l_name: The name of the layer container to key the metadata.

        Captured Variables:
            buffer: The flattened buffer to reconstruct views on top of.
            metadata: The metadata object describing the each parameter in the model.
        """
        l_metadata = metadata.layers[l_name]

        for p_name in layer_container.annotation_attrs:
            p_metadata = l_metadata.params[p_name]

            if p_metadata.core_param.offset == -1:
                layer_container.direct_injection(p_name, None)
                continue

            dummy_tensor = torch.empty([], dtype=STR_TO_DTYPE[p_metadata.core_param.dtype])
            core_param = alloc_fn(p_metadata.core_param.shape, p_metadata.core_param.strides, dummy_tensor, buffer,
                                  p_metadata.core_param.offset)

            aux_params = {}

            for t_name, t_metadata in p_metadata.aux_params.items():
                dummy_tensor = torch.empty([], dtype=STR_TO_DTYPE[t_metadata.dtype])
                t_view = alloc_fn(t_metadata.shape, t_metadata.strides, dummy_tensor, buffer, t_metadata.offset)

                aux_params[t_name] = t_view

            restored_param = InferenceParameter.initialize(core_param, **aux_params)
            layer_container.direct_injection(p_name, restored_param)

    for i, layer in enumerate(transformer_containers):
        l_name = f"transformer_layer_{i}"
        restore_layer(layer, l_name)

    l_name = "non_transformer"
    restore_layer(non_transformer_container, l_name)
