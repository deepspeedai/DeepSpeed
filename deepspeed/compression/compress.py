# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import re
from .helper import compression_preparation, fix_compression, recursive_getattr, is_module_compressible
from .config import get_compression_config
from ..runtime.config_utils import dict_raise_error_on_duplicate_keys
from .constants import *
import os
import json

try:
    import neural_compressor as nc
except ImportError as e:
    nc = None


def check_deepspeed_config(config):
    if isinstance(config, dict):
        return config
    elif os.path.exists(config):
        return json.load(open(config, "r"), object_pairs_hook=dict_raise_error_on_duplicate_keys)
    else:
        raise ValueError(
            f"Expected a string path to an existing deepspeed config, or a dictionary. Received: {config}")


def get_module_name(group_name, model, key_word, exist_module_name, mpu=None, verbose=True):
    '''
    get the associated module name from the model based on the key_word provided by users
    '''
    return_module_name = []
    for name, module in model.named_modules():

        module_check = is_module_compressible(module, mpu)

        if re.search(key_word, name) is not None and module_check:
            if name in exist_module_name and verbose:
                # logger.warning
                raise ValueError(
                    f"{name} is already added to compression, please check your config file for {group_name}.")
            if name not in exist_module_name:
                exist_module_name.add(name)
                return_module_name.append(name)
    return return_module_name, exist_module_name


def get_compress_methods(model, compress_methods, mpu=None):
    # extract the compression module for each method in compress_methods
    layer_added_compress_methods = []
    for method, method_content in compress_methods.items():
        if LAYER_REDUCTION in method:
            continue
        # for loop different methods, i.e., weight quantization, activation quantization etc
        exist_module_name = set()
        shared_parameters = method_content[SHARED_PARAMETERS]  # get all the shared parameters
        for group_name, method_parameters in method_content[DIFFERENT_GROUPS].items():
            # for loop different groups, i.e., weight quantization group 1, weight quantization group 2 etc
            module_name_list = []
            related_module_name_list = []
            if method_parameters[DIFFERENT_GROUPS_RELATED_MODULE_SCOPE]:
                # this is used for head/row/channel pruning, if users provide the related module scope, we can shrink the layer dim for them
                # otherwise we just mask those as zeros
                for key_word, related_key_words in zip(method_parameters[DIFFERENT_GROUPS_MODULE_SCOPE],
                                                       method_parameters[DIFFERENT_GROUPS_RELATED_MODULE_SCOPE]):
                    module_name, exist_module_name = get_module_name(group_name,
                                                                     model,
                                                                     key_word,
                                                                     exist_module_name,
                                                                     mpu=mpu)
                    module_name_list.append(module_name)
                    tmp_related_module_name_list = []
                    for rkw in related_key_words:
                        # related key word can be a list, for instance the QKV for O matrix in Attention
                        module_name, _ = get_module_name(group_name, model, rkw, set(), mpu=mpu)
                        tmp_related_module_name_list.append(module_name)
                    related_module_name_list.append(tmp_related_module_name_list)
            else:
                for key_word in method_parameters[DIFFERENT_GROUPS_MODULE_SCOPE]:
                    module_name, exist_module_name = get_module_name(group_name,
                                                                     model,
                                                                     key_word,
                                                                     exist_module_name,
                                                                     mpu=mpu)
                    module_name_list.append(module_name)

            if module_name_list:
                # combine shared parameters with each group
                combined_method_parameters = {
                    **(method_parameters.copy().pop(DIFFERENT_GROUPS_PARAMETERS)),
                    **shared_parameters
                }
                compression_item = [module_name_list, related_module_name_list, {method: combined_method_parameters}]
                layer_added_compress_methods.append(compression_item)
    return layer_added_compress_methods


def init_compression(model, deepspeed_config, teacher_model=None, mpu=None):
    """
    Compress a model: replace linear/conv2d layer with deepspeed compression-aware modules
    Args:
        model (`torch.nn.Module`)
            The model to compress.
        deepspeed_config (`DeepSpeedConfig`)
            The path of ds_config
        mpu
            The mpu module for Row/Column parallelism
    """
    compress_methods = get_compression_config(check_deepspeed_config(deepspeed_config))
    if hasattr(model, 'module'):
        c_model = model.module
    else:
        c_model = model

    # For layer reduction
    if compress_methods[LAYER_REDUCTION][LAYER_REDUCTION_ENABLED]:
        assert teacher_model is not None, "Teacher model is required for layer reduction"
        student_initialization(c_model, teacher_model, deepspeed_config)

    layer_added_compress_methods = get_compress_methods(c_model, compress_methods, mpu=mpu)
    compression_preparation(c_model, layer_added_compress_methods, mpu)

    # For sparse pruning snip_momentum method
    shared_parameters = compress_methods[SPARSE_PRUNING][SHARED_PARAMETERS]
    if shared_parameters[SPARSE_PRUNING_ENABLED] and \
        shared_parameters[SPARSE_PRUNING_METHOD] == SPARSE_PRUNING_METHOD_SNIP_MOMENTUM:

        assert nc is not None, "please ensure the neural_compressor python package is installed by pip or conda if user wants to use snip_momentum sparse pruning"

        from .helper import generate_pruners, register_on_step_begin
        from nc import WeightPruningConfig

        config = WeightPruningConfig(target_sparsity=1 - shared_parameters[SPARSE_PRUNING_DENSE_RATIO],
                                     pattern=shared_parameters[SPARSE_PRUNING_BLOCK_PATTERN],
                                     pruning_frequency=shared_parameters[SPARSE_PRUNING_SCHEDULE_OFFSET_STRIDE],
                                     start_step=shared_parameters[SPARSE_PRUNING_SCHEDULE_OFFSET],
                                     end_step=shared_parameters[SPARSE_PRUNING_SCHEDULE_OFFSET_END],
                                     excluded_op_names=shared_parameters[SPARSE_PRUNING_EXCLUDED_MODULES])
        pruners = generate_pruners(config, c_model)
        c_model.pruners = pruners
        register_on_step_begin(c_model)

    return model


def redundancy_clean(model, deepspeed_config, mpu=None):
    """
    Remove the redundancy of a model
    Args:
        model (`torch.nn.Module`)
            The model to compress.
        deepspeed_config (`DeepSpeedConfig`)
            The path of ds_config
        mpu
            The mpu module for Row/Column parallelism
    """
    compress_methods = get_compression_config(check_deepspeed_config(deepspeed_config))
    if hasattr(model, 'module'):
        c_model = model.module
    else:
        c_model = model

    layer_added_compress_methods_tmp = get_compress_methods(c_model, compress_methods, mpu=mpu)
    # sort methods
    order_list = [
        WEIGHT_QUANTIZATION, SPARSE_PRUNING, ROW_PRUNING, HEAD_PRUNING, CHANNEL_PRUNING, ACTIVATION_QUANTIZATION
    ]
    layer_added_compress_methods = sorted(layer_added_compress_methods_tmp,
                                          key=lambda x: order_list.index(list(x[2].keys())[0]))

    for module_name_lists, related_module_name_lists, compression_technique in layer_added_compress_methods:
        stored_mask = []
        need_mask = True if related_module_name_lists else False
        for i, mnl in enumerate(module_name_lists):
            for module_name in mnl:
                mask = fix_compression(c_model, module_name, compression_technique, dim_reduction=need_mask)
                if need_mask:
                    stored_mask.append(mask)
            if need_mask:
                for rmnl in related_module_name_lists[i]:
                    for j, module_name in enumerate(rmnl):
                        mask = fix_compression(c_model,
                                               module_name,
                                               compression_technique,
                                               mask=stored_mask[j],
                                               dim_reduction=True)
    return model


def student_initialization(student_model, teacher_model, deepspeed_config):
    '''
    Given a student model and a teacher model, select the
    Args:
        student_model (`torch.nn.Module`)
            The model we will update weight
        teacher_model (`torch.nn.Module`)
            The model guide the student to learn
        deepspeed_config (`DeepSpeedConfig`)
            The path of ds_config
    '''
    config = get_compression_config(check_deepspeed_config(deepspeed_config))
    compress_methods = config[LAYER_REDUCTION]

    module_name_prefix = compress_methods[MODULE_NAME_PREFIX]
    teacher_layer = compress_methods[TEACHER_LAYER]
    student_layer = [i for i in range(len(teacher_layer))]
    other_module_name = compress_methods[OTHER_MODULE_NAME]
    '''
        name_prefix (`str`)
            The prefix name before the layer #.
            Example 1: bert.encoder.layer, for BERT_base model's prefix name
            Example 2: transformer.h, for GPT-2 hugging face prefix name
        teacher_layer (`list of integers`)
            The layer of teacher will be used for student's reinitialization
            Example 1: [1,3,5,7,9], means we want to matches the 2nd/4th/6th/8th/10th layer of teacher to the first 5 layers of student
        student_layer (`list` or None)
            The layer of student need to be re-initialized
            Example 1: None, means we want to reinitialize all the layers
            Example 1: [0,1,2,3,4], means  we want to reinitialize the first 5 layers
        other_module_name (`list of string`)
            The modules will be used for student's reinitialization
            Example 1: ['bert.pooler', 'bert.embeddings', 'classifier'], means we want to apply the weight in teacher's embedding/pooler/classier module to the student
            Example 2: ['transformer.w', 'transformer.ln_f', 'lm_head'], means we want to apply the weight in teacher's embedding layers module to the student
    Note that teacher_layer should matches student layer
    '''
    assert len(student_layer) == len(teacher_layer)
    for s_name, t_name in zip(student_layer, teacher_layer):
        s_module = recursive_getattr(student_model, module_name_prefix + '.' + str(s_name))
        t_module = recursive_getattr(teacher_model, module_name_prefix + '.' + str(t_name))
        for s_param, t_param in zip(s_module.parameters(), t_module.parameters()):
            s_param.data.copy_(t_param.data)
    for name in other_module_name:
        s_module = recursive_getattr(student_model, name)
        t_module = recursive_getattr(teacher_model, name)
        print(name)
        for s_param, t_param in zip(s_module.parameters(), t_module.parameters()):
            s_param.data.copy_(t_param.data)
