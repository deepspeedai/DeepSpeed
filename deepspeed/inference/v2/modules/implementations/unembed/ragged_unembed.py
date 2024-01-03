# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Dict, Optional

import torch

from deepspeed.accelerator import get_accelerator
from ....allocator import empty_from
from ....inference_utils import DtypeEnum, ActivationType
from ....kernels.core_ops import CUDAFPLN, BlasLibLinear, CUDARMSNorm, CUDABiasActivation
from ....kernels.ragged_ops import RaggedLogitsGather
from ....ragged import RaggedBatchWrapper
from ...interfaces import DSUnembedBase, DSUnembedRegistry
from ...configs import DSUnembedConfig


@DSUnembedRegistry.register_module
class DSRaggedUnembed(DSUnembedBase):
    """
    Ragged unembedding implementation. This implementation will gather only the last token
    of each sequence in the ragged inflight batch and calculate the logits only for those rows.
    """

    @staticmethod
    def name():
        return 'ragged_unembed'

    @staticmethod
    def supports_config(config: DSUnembedConfig):

        if DtypeEnum(config.dtype) not in [DtypeEnum.fp16, DtypeEnum.bf16, DtypeEnum.fp32]:
            return False

        try:
            _ = RaggedLogitsGather(config.model_dim, config.dtype)
        except ValueError:
            return False

        if config.norm_type == 'rms_norm':
            try:
                _ = CUDARMSNorm(config.model_dim, config.dtype)
            except ValueError:
                return False
        elif config.norm_type == 'layer_norm':
            try:
                _ = CUDAFPLN(config.model_dim, config.dtype)
            except ValueError:
                return False

        return True

    def __init__(self, config: DSUnembedConfig, implementation_config: Dict[str, Any]) -> None:
        super().__init__(config, implementation_config)

        self._logits_gather = RaggedLogitsGather(config.model_dim, self._config.dtype)

        if self._config.norm_type == 'layer_norm':
            self._norm = CUDAFPLN(self._config.model_dim, self._config.dtype)
        elif self._config.norm_type == 'rms_norm':
            self._norm = CUDARMSNorm(self._config.model_dim, self._config.dtype)
        else:
            self._norm = None

        self._linear = BlasLibLinear(self._config.dtype)
        # Here the activation kernel is being used to apply bias, hence the identity activation type!
        self._act_fn = CUDABiasActivation(self._config.vocab_size, self._config.dtype, ActivationType.IDENTITY)

        self._intermediate = torch.empty((self._config.max_sequences, self._config.model_dim),
                                         dtype=self._config.dtype,
                                         device=get_accelerator().current_device())

        self._output = torch.empty((self._config.max_sequences, self._config.vocab_size),
                                   dtype=self._config.dtype,
                                   device=get_accelerator().current_device())

    @property
    def output(self) -> torch.Tensor:
        return self._output

    def forward(self,
                hidden_states: torch.Tensor,
                vocab_embedding: torch.Tensor,
                ragged_metadata: RaggedBatchWrapper,
                bias: Optional[torch.Tensor] = None,
                gamma: Optional[torch.Tensor] = None,
                beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return final model logits.

        Args:
            hidden_states (torch.Tensor): The hidden states from the model. This is the output of the
                final layer of the model.
            vocab_embedding (torch.Tensor): The vocab embedding table.
            raged_metadata (RaggedBatchWrapper): The ragged batch metadata.
            gamma (Optional[torch.Tensor]): The gamma tensor for normalization.
            beta (Optional[torch.Tensor]): The beta tensor for normalization.
        """

        cut_down_hidden_states = empty_from(self._intermediate,
                                            (ragged_metadata.current_sequences, self._config.model_dim))
        self._logits_gather(cut_down_hidden_states, hidden_states, ragged_metadata)

        if self._config.norm_type == 'rms_norm':
            if gamma is None:
                raise ValueError('RMS Normalization enabled but gamma not provided.')
            self._norm(cut_down_hidden_states, cut_down_hidden_states, gamma)
        elif self._config.norm_type == 'layer_norm':
            if gamma is None or beta is None:
                raise ValueError('Normalization enabled but gamma and/or beta not provided.')
            self._norm(cut_down_hidden_states, cut_down_hidden_states, gamma, beta)

        output = empty_from(self._output, (ragged_metadata.current_sequences, self._config.vocab_size))
        self._linear(output, cut_down_hidden_states, vocab_embedding)
        if bias is not None:
            self._act_fn(output, bias)

        return output
