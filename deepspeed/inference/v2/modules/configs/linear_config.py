# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from ...inference_utils import ActivationType, DtypeEnum
from ...modules.ds_module import DSModuleConfig


class DSLinearConfig(DSModuleConfig):
    """
    Config class for DSLinearBase.
    """

    in_channels: int
    """
    Number of input channels
    """

    out_channels: int
    """
    Number of output channels. NOTE: If this linear layer is using a gated activation function,
    the value for ``out_channels`` passed here should refer to the number of channels after
    gating (i.e., the expected weight shape before transformations will be ``[out_channels * 2, in_channels]``).
    """

    activation: ActivationType = ActivationType.IDENTITY
    """
    The activation function for this layer. See :class:`deepspeed.inference.inference_utils.ActivationType` for
    supported activation functions.
    """

    input_dtype: DtypeEnum = DtypeEnum.fp16
    """
    The data type of the input tensor. See :class:`deepspeed.inference.inference_utils.DtypeEnum` for supported
    data types.
    """

    output_dtype: DtypeEnum = DtypeEnum.fp16
    """
    The data type of the output tensor. See :class:`deepspeed.inference.inference_utils.DtypeEnum` for supported
    data types.
    """
