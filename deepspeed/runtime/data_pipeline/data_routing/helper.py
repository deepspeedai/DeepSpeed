# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .basic_layer import RandomLayerTokenDrop
from collections import OrderedDict
from deepspeed.compression.helper import recursive_getattr, recursive_setattr


def convert_to_random_ltd(model, convert_type):
    if hasattr(model, 'module'):
        c_model = model.module
    else:
        c_model = model

    for name, module in c_model.named_modules():

        if isinstance(module, convert_type):
            old_module = recursive_getattr(c_model, name)
            new_module = RandomLayerTokenDrop(old_module)
            recursive_setattr(c_model, name, new_module)

    model.random_ltd_initialize()
    return model


def save_without_random_ltd(model):
    if hasattr(model, 'module'):
        c_model = model.module
    else:
        c_model = model

    model_dic = c_model.state_dict()
    return remove_random_ltd_state_dict(model_dic)


def remove_random_ltd_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if '.random_ltd_layer' in key:
            new_key = ''.join(key.split('.random_ltd_layer'))
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
