# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from .config import TPTrainingConfig, TPConfig
from deepspeed.utils import groups
import deepspeed.comm as dist


class TpTrainingManager():

    def __init__(self, model, tp_size, dtype):
        self.module = model
        self.config = self._initialize_config(dtype)

        from deepspeed.module_inject.auto_tp import AutoTP
        from deepspeed import get_accelerator

        # Parse model configuration
        parser_dict = AutoTP.tp_parser(model)
        print("AutoTP: ", parser_dict)

        # Initialize TP configuration and model
        self._initialize_tp_config(tp_size)
        self._get_model_config_generate()

        # Synchronize random number generator state across devices
        _rng_state = get_accelerator().get_rng_state().to(get_accelerator().current_device_name())
        dist.broadcast(_rng_state, groups.get_tensor_model_parallel_src_rank(), self.tp_config.tp_group)
        get_accelerator().set_rng_state(_rng_state.cpu())

        # Apply injection policies
        self._apply_policies(parser_dict)

    def _initialize_config(self, dtype):
        """Initialize and return the DeepSpeed TP training configuration."""
        config = TPTrainingConfig()
        config.dtype = dtype
        return config

    def _apply_policies(self, parser_dict):
        """Apply injection policies to the parsed modules."""
        for client_module, injection_policy in parser_dict:
            self.config.injection_policy_tuple = injection_policy
            self._apply_injection_policy(self.config, client_module)

    def _apply_injection_policy(self, config, client_module=None):
        from deepspeed.module_inject import replace_transformer_layer
        """Apply the given injection policy to a client module."""
        if isinstance(self.module, torch.nn.Module):
            replace_transformer_layer(client_module, self.module, None, self.config, self.model_config)

    def _initialize_tp_config(self, tp_size):
        """Perform TP configuration initialization."""
        self.tp_config = TPConfig()
        self.tp_config.tp_size = tp_size

        groups._init_tp_mesh_device(tp_size)
        self.tp_config.tp_group = groups.get_tensor_model_parallel_group()
        self.config.tensor_parallel = self.tp_config

    def _get_model_config_generate(self):
        """Generate and apply HF model  configuration."""
        self.model_config = getattr(self.module, 'config', None)
