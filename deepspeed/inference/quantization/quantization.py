# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch import nn
from typing import Dict
import gc
from deepspeed.inference.quantization import layers
from .layers import QUANTIZATION_LAYER_MAPPINGS
from .utils import get_AsyncPartitionedParameterSwapper, recursive_setattr
from deepspeed.utils.logging import logger
from collections import deque
from transformers.utils.generic import ContextManagers
from .quantization_context import QuantizationContext
import contextlib


def _init_group_wise_weight_quantization(model: nn.Module, ds_config: Dict) -> nn.Module:
    """[Experimental] Apply group-wise weight quantization to model. Replace layers module according to config_list

    Args:
        model (nn.Module): A nn.Module
        ds_config (Dict, optional): The ds_config dictionary. use None for non-deepspeed managed model.

    Returns:
        nn.Module: Quantized nn.Module
    """

    # global quantized_weight_registry

    matched_module_list_by_key = {}
    matched_module_count = 0

    assert 'weight_quantization' in ds_config, 'Please provide quantization config in ds_config'
    quantization_config = ds_config['weight_quantization']['post_init_quant']

    # Return nvme swapper if exists, else return None.
    # For nvme offloading we must use the same swapper here as model initialized.
    nvme_swapper = get_AsyncPartitionedParameterSwapper(model)
    is_zero3_enabled = 'zero_optimization' in ds_config and \
            'stage' in ds_config['zero_optimization'] and \
            ds_config['zero_optimization']['stage'] == 3
    is_offloading_enabled = 'zero_optimization' in ds_config and \
                            'offload_param' in ds_config['zero_optimization']

    layers.is_zero3_enabled = is_zero3_enabled

    context_mgr = ContextManagers([QuantizationContext(config_dict_or_path=ds_config, param_swapper=nvme_swapper)]) \
                    if is_zero3_enabled else contextlib.suppress()
    with context_mgr:
        module_list = list(
            filter(lambda named_module: type(named_module[1]) in QUANTIZATION_LAYER_MAPPINGS, model.named_modules()))

        # Quantize small weight first then large.
        if not is_offloading_enabled:
            module_list.sort(key=lambda named_module: named_module[1].weight.ds_tensor.numel()
                             if is_zero3_enabled else named_module[1].weight.numel())
        module_list = deque(module_list)

        while len(module_list) > 0:
            # Use popleft to timely release module's memory of replaced module after each loop iteration
            module_name, module = module_list.popleft()

            matched_key = None
            matched_quantization_config = None

            for key, config in quantization_config.items():
                if key in module_name:
                    assert matched_key is None, f'{module_name} matched multiple quantization key word {matched_key} and {key}'
                    matched_key = key
                    matched_quantization_config = config

            if matched_key is None:
                continue

            if is_zero3_enabled:
                module.weight.all_gather()

            assert module.weight.dtype == torch.float16, 'Model weight is expected in half.'

            new_module = QUANTIZATION_LAYER_MAPPINGS[type(module)](matched_quantization_config, module)

            if is_zero3_enabled:
                module.weight.partition()

            recursive_setattr(model, module_name, new_module)

            if matched_key not in matched_module_list_by_key:
                matched_module_list_by_key[matched_key] = []
            matched_module_list_by_key[matched_key].append(module_name)
            matched_module_count += 1

            # Timely recycle memory to prevent OOM on large models
            gc.collect()

    # Clear registry after model construction.
    layers.quantized_weight_registry.clear()

    logger.info(
        f'Group-wise weight quantization summary: convert {matched_module_count} node(s) to quantized implementation')
    summary_str = '\n'

    for key, module_list in matched_module_list_by_key.items():
        summary_str += f'Key: {key}, matched modules:\n'
        for module_name in module_list:
            summary_str += f'\t{module_name}\n'
    logger.info(summary_str)

    return model
