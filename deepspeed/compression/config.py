# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .constants import *
import copy
from ..runtime.config_utils import get_scalar_param, get_list_param


def get_compression_config(param_dict):
    #
    output = {}

    if COMPRESSION_TRAINING not in param_dict.keys():
        param_dict[COMPRESSION_TRAINING] = {}
    sub_param_dict = param_dict[COMPRESSION_TRAINING]
    output[WEIGHT_QUANTIZATION] = get_weight_quantization(sub_param_dict)
    output[ACTIVATION_QUANTIZATION] = get_activation_quantization(sub_param_dict)
    output[SPARSE_PRUNING] = get_sparse_pruning(sub_param_dict)
    output[ROW_PRUNING] = get_row_pruning(sub_param_dict)
    output[HEAD_PRUNING] = get_head_pruning(sub_param_dict)
    output[CHANNEL_PRUNING] = get_channel_pruning(sub_param_dict)

    output[LAYER_REDUCTION] = get_layer_reduction(sub_param_dict)

    return output


def get_layer_reduction(param_dict):
    output = {}
    output[LAYER_REDUCTION_ENABLED] = LAYER_REDUCTION_ENABLED_DEFAULT
    if get_layer_reduction_enabled(param_dict):
        output[LAYER_REDUCTION_ENABLED] = get_layer_reduction_enabled(param_dict)
        for key, val in get_layer_reduction_params(param_dict).items():
            output[key] = val
    return output


def get_layer_reduction_enabled(param_dict):
    if LAYER_REDUCTION in param_dict.keys():
        return get_scalar_param(param_dict[LAYER_REDUCTION], LAYER_REDUCTION_ENABLED, LAYER_REDUCTION_ENABLED_DEFAULT)
    else:
        return False


def get_layer_reduction_params(param_dict):
    if LAYER_REDUCTION in param_dict.keys():
        layer_reduction_params = copy.copy(param_dict[LAYER_REDUCTION])
        layer_reduction_params.pop(LAYER_REDUCTION_ENABLED)
        return layer_reduction_params
    else:
        return False


def get_quantize_enabled(param_dict):
    if COMPRESSION_TRAINING not in param_dict.keys():
        return False

    sub_param_dict = param_dict[COMPRESSION_TRAINING]
    output = get_weight_quantization_shared_parameters(sub_param_dict)
    return output[WEIGHT_QUANTIZE_ENABLED]


def get_weight_quantization(param_dict):
    output = {}
    if WEIGHT_QUANTIZATION not in param_dict.keys():
        param_dict[WEIGHT_QUANTIZATION] = {SHARED_PARAMETERS: {}, DIFFERENT_GROUPS: {}}
    sub_param_dict = param_dict[WEIGHT_QUANTIZATION]
    # shared parameters
    output[SHARED_PARAMETERS] = get_weight_quantization_shared_parameters(sub_param_dict)
    # each sub-groups
    if output[SHARED_PARAMETERS][WEIGHT_QUANTIZE_ENABLED]:
        assert DIFFERENT_GROUPS in sub_param_dict.keys(
        ), f"Weigh Quantization is enabled, {DIFFERENT_GROUPS} must be specified"
    output[DIFFERENT_GROUPS] = get_weight_quantization_different_groups(sub_param_dict)
    return output


def get_weight_quantization_shared_parameters(param_dict):
    output = {}
    if SHARED_PARAMETERS in param_dict.keys():
        sub_param_dict = param_dict[SHARED_PARAMETERS]
        output[WEIGHT_QUANTIZE_ENABLED] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_ENABLED,
                                                           WEIGHT_QUANTIZE_ENABLED_DEFAULT)
        output[WEIGHT_QUANTIZE_KERNEL] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_KERNEL,
                                                          WEIGHT_QUANTIZE_KERNEL_DEFAULT)
        output[WEIGHT_QUANTIZE_SCHEDULE_OFFSET] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_SCHEDULE_OFFSET,
                                                                   WEIGHT_QUANTIZE_SCHEDULE_OFFSET_DEFAULT)
        output[WEIGHT_QUANTIZE_GROUPS] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_GROUPS,
                                                          WEIGHT_QUANTIZE_GROUPS_DEFAULT)
        output[WEIGHT_QUANTIZE_VERBOSE] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_VERBOSE,
                                                           WEIGHT_QUANTIZE_VERBOSE_DEFAULT)
        output[WEIGHT_QUANTIZE_TYPE] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_TYPE,
                                                        WEIGHT_QUANTIZE_TYPE_DEFAULT)
        output[WEIGHT_QUANTIZE_IN_FORWARD_ENABLED] = get_scalar_param(sub_param_dict,
                                                                      WEIGHT_QUANTIZE_IN_FORWARD_ENABLED,
                                                                      WEIGHT_QUANTIZE_IN_FORWARD_ENABLED_DEFAULT)
        assert output[WEIGHT_QUANTIZE_TYPE] in [
            WEIGHT_QUANTIZE_SYMMETRIC, WEIGHT_QUANTIZE_ASYMMETRIC
        ], f"Invalid weight quantize type. Supported types: [{WEIGHT_QUANTIZE_SYMMETRIC}, {WEIGHT_QUANTIZE_ASYMMETRIC}]"
        output[WEIGHT_QUANTIZE_ROUNDING] = get_scalar_param(sub_param_dict, WEIGHT_QUANTIZE_ROUNDING,
                                                            WEIGHT_QUANTIZE_ROUNDING_DEFAULT)
        assert output[WEIGHT_QUANTIZE_ROUNDING] in [
            WEIGHT_QUANTIZE_NEAREST_ROUNDING, WEIGHT_QUANTIZE_STOCHASTIC_ROUNDING
        ], f"Invalid weight quantize rounding. Supported types: [{WEIGHT_QUANTIZE_NEAREST_ROUNDING}, {WEIGHT_QUANTIZE_STOCHASTIC_ROUNDING}]"
        if WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE in sub_param_dict.keys():
            output[WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE] = get_scalar_param(
                sub_param_dict[WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE], WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE_ENABLED,
                WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE_ENABLED_DEFAULT)
            output[WEIGHT_QUANTIZE_CHANGE_RATIO] = get_scalar_param(
                sub_param_dict[WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE], WEIGHT_QUANTIZE_CHANGE_RATIO,
                WEIGHT_QUANTIZE_CHANGE_RATIO_DEFAULT)
        else:
            output[WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE] = WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE_ENABLED_DEFAULT
            output[WEIGHT_QUANTIZE_CHANGE_RATIO] = WEIGHT_QUANTIZE_CHANGE_RATIO_DEFAULT
    else:
        output[WEIGHT_QUANTIZE_ENABLED] = WEIGHT_QUANTIZE_ENABLED_DEFAULT
        output[WEIGHT_QUANTIZE_KERNEL] = WEIGHT_QUANTIZE_KERNEL_DEFAULT
        output[WEIGHT_QUANTIZE_SCHEDULE_OFFSET] = WEIGHT_QUANTIZE_SCHEDULE_OFFSET_DEFAULT
        output[WEIGHT_QUANTIZE_GROUPS] = WEIGHT_QUANTIZE_GROUPS_DEFAULT
        output[WEIGHT_QUANTIZE_VERBOSE] = WEIGHT_QUANTIZE_VERBOSE_DEFAULT
        output[WEIGHT_QUANTIZE_TYPE] = WEIGHT_QUANTIZE_TYPE_DEFAULT
        output[WEIGHT_QUANTIZE_ROUNDING] = WEIGHT_QUANTIZE_ROUNDING_DEFAULT
        output[WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE] = WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE_ENABLED_DEFAULT
        output[WEIGHT_QUANTIZE_CHANGE_RATIO] = WEIGHT_QUANTIZE_CHANGE_RATIO_DEFAULT
    return output


def get_weight_quantization_different_groups(param_dict):
    output = {}
    sub_param_dict = param_dict[DIFFERENT_GROUPS]

    def get_params(name, group_dict):
        assert WEIGHT_QUANTIZE_START_BITS in group_dict.keys(
        ), f"{WEIGHT_QUANTIZE_START_BITS} must be specified for weight quantization group {name}"
        assert WEIGHT_QUANTIZE_TARGET_BITS in group_dict.keys(
        ), f"{WEIGHT_QUANTIZE_TARGET_BITS} must be specified for weight quantization group {name}"
        group_dict[WEIGHT_QUANTIZATION_PERIOD] = get_scalar_param(group_dict, WEIGHT_QUANTIZATION_PERIOD,
                                                                  WEIGHT_QUANTIZATION_PERIOD_DEFAULT)
        return group_dict

    for k, v in sub_param_dict.items():
        output[k] = {}
        output[k][DIFFERENT_GROUPS_PARAMETERS] = get_params(k, sub_param_dict[k][DIFFERENT_GROUPS_PARAMETERS])
        output[k][DIFFERENT_GROUPS_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_MODULE_SCOPE,
                                                                    DIFFERENT_GROUPS_MODULE_SCOPE_DEFAULT)
        output[k][DIFFERENT_GROUPS_RELATED_MODULE_SCOPE] = get_scalar_param(
            sub_param_dict[k], DIFFERENT_GROUPS_RELATED_MODULE_SCOPE, DIFFERENT_GROUPS_RELATED_MODULE_SCOPE_DEFAULT)

    return output


def get_activation_quantization(param_dict):
    output = {}
    if ACTIVATION_QUANTIZATION not in param_dict.keys():
        param_dict[ACTIVATION_QUANTIZATION] = {SHARED_PARAMETERS: {}, DIFFERENT_GROUPS: {}}
    sub_param_dict = param_dict[ACTIVATION_QUANTIZATION]
    # shared parameters
    output[SHARED_PARAMETERS] = get_activation_quantization_shared_parameters(sub_param_dict)
    # each sub-groups
    if output[SHARED_PARAMETERS][ACTIVATION_QUANTIZATION_ENABLED]:
        assert DIFFERENT_GROUPS in sub_param_dict.keys(
        ), f"Activation Quantization is enabled, {DIFFERENT_GROUPS} must be specified"
    output[DIFFERENT_GROUPS] = get_activation_quantization_different_groups(sub_param_dict)
    return output


def get_activation_quantization_shared_parameters(param_dict):
    output = {}
    if SHARED_PARAMETERS in param_dict.keys():
        sub_param_dict = param_dict[SHARED_PARAMETERS]
        output[ACTIVATION_QUANTIZATION_ENABLED] = get_scalar_param(sub_param_dict, ACTIVATION_QUANTIZATION_ENABLED,
                                                                   ACTIVATION_QUANTIZATION_ENABLED_DEFAULT)
        output[ACTIVATION_QUANTIZE_TYPE] = get_scalar_param(sub_param_dict, ACTIVATION_QUANTIZE_TYPE,
                                                            ACTIVATION_QUANTIZE_TYPE_DEFAULT)
        assert output[ACTIVATION_QUANTIZE_TYPE] in [
            ACTIVATION_QUANTIZE_SYMMETRIC, ACTIVATION_QUANTIZE_ASYMMETRIC
        ], f"Invalid activation quantize type. Supported types: [{ACTIVATION_QUANTIZE_SYMMETRIC}, {ACTIVATION_QUANTIZE_ASYMMETRIC}]"
        output[ACTIVATION_QUANTIZE_RANGE] = get_scalar_param(sub_param_dict, ACTIVATION_QUANTIZE_RANGE,
                                                             ACTIVATION_QUANTIZE_RANGE_DEFAULT)
        assert output[ACTIVATION_QUANTIZE_RANGE] in [
            ACTIVATION_QUANTIZE_RANGE_DYNAMIC, ACTIVATION_QUANTIZE_RANGE_STATIC
        ], f"Invalid activation quantize range calibration. Supported types: [{ACTIVATION_QUANTIZE_RANGE_DYNAMIC}, {ACTIVATION_QUANTIZE_RANGE_STATIC}]"
        output[ACTIVATION_QUANTIZE_SCHEDULE_OFFSET] = get_scalar_param(sub_param_dict,
                                                                       ACTIVATION_QUANTIZE_SCHEDULE_OFFSET,
                                                                       ACTIVATION_QUANTIZE_SCHEDULE_OFFSET_DEFAULT)
    else:
        output[ACTIVATION_QUANTIZATION_ENABLED] = ACTIVATION_QUANTIZATION_ENABLED_DEFAULT
        output[ACTIVATION_QUANTIZE_TYPE] = ACTIVATION_QUANTIZE_TYPE_DEFAULT
        output[ACTIVATION_QUANTIZE_RANGE] = ACTIVATION_QUANTIZE_RANGE_DEFAULT
        output[ACTIVATION_QUANTIZE_SCHEDULE_OFFSET] = ACTIVATION_QUANTIZE_SCHEDULE_OFFSET_DEFAULT
    return output


def get_activation_quantization_different_groups(param_dict):
    output = {}
    sub_param_dict = param_dict[DIFFERENT_GROUPS]

    def get_params(name, group_dict):
        assert ACTIVATION_QUANTIZE_BITS in group_dict.keys(
        ), f"{ACTIVATION_QUANTIZE_BITS} must be specified for activation quantization group {name}"
        return group_dict

    for k, v in sub_param_dict.items():
        output[k] = {}
        output[k][DIFFERENT_GROUPS_PARAMETERS] = get_params(k, sub_param_dict[k][DIFFERENT_GROUPS_PARAMETERS])
        output[k][DIFFERENT_GROUPS_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_MODULE_SCOPE,
                                                                    DIFFERENT_GROUPS_MODULE_SCOPE_DEFAULT)
        output[k][DIFFERENT_GROUPS_RELATED_MODULE_SCOPE] = get_scalar_param(
            sub_param_dict[k], DIFFERENT_GROUPS_RELATED_MODULE_SCOPE, DIFFERENT_GROUPS_RELATED_MODULE_SCOPE_DEFAULT)

    return output


def get_sparse_pruning(param_dict):
    output = {}
    if SPARSE_PRUNING not in param_dict.keys():
        param_dict[SPARSE_PRUNING] = {SHARED_PARAMETERS: {}, DIFFERENT_GROUPS: {}}
    sub_param_dict = param_dict[SPARSE_PRUNING]
    # shared parameters
    output[SHARED_PARAMETERS] = get_sparse_pruning_shared_parameters(sub_param_dict)
    # each sub-groups
    if output[SHARED_PARAMETERS][SPARSE_PRUNING_ENABLED] and output[SHARED_PARAMETERS][
            SPARSE_PRUNING_METHOD] != SPARSE_PRUNING_METHOD_SNIP_MOMENTUM:
        assert DIFFERENT_GROUPS in sub_param_dict.keys(
        ), f"Sparse Pruning is enabled and not snip_momentum method, {DIFFERENT_GROUPS} must be specified"
    output[DIFFERENT_GROUPS] = get_sparse_pruning_different_groups(sub_param_dict)
    return output


def get_sparse_pruning_shared_parameters(param_dict):
    output = {}

    if SHARED_PARAMETERS in param_dict.keys():
        sub_param_dict = param_dict[SHARED_PARAMETERS]
        output[SPARSE_PRUNING_ENABLED] = get_scalar_param(sub_param_dict, SPARSE_PRUNING_ENABLED,
                                                          SPARSE_PRUNING_ENABLED_DEFAULT)
        output[SPARSE_PRUNING_METHOD] = get_scalar_param(sub_param_dict, SPARSE_PRUNING_METHOD,
                                                         SPARSE_PRUNING_METHOD_DEFAULT)
        assert output[SPARSE_PRUNING_METHOD] in [
            SPARSE_PRUNING_METHOD_L1, SPARSE_PRUNING_METHOD_TOPK, SPARSE_PRUNING_METHOD_SNIP_MOMENTUM
        ], f"Invalid sparse pruning method. Supported types: [{SPARSE_PRUNING_METHOD_L1}, {SPARSE_PRUNING_METHOD_TOPK}, {SPARSE_PRUNING_METHOD_SNIP_MOMENTUM}]"
        output[SPARSE_PRUNING_SCHEDULE_OFFSET] = get_scalar_param(sub_param_dict, SPARSE_PRUNING_SCHEDULE_OFFSET,
                                                                  SPARSE_PRUNING_SCHEDULE_OFFSET_DEFAULT)
        if output[SPARSE_PRUNING_METHOD] == SPARSE_PRUNING_METHOD_SNIP_MOMENTUM:
            output[SPARSE_PRUNING_BLOCK_PATTERN] = get_scalar_param(sub_param_dict, SPARSE_PRUNING_BLOCK_PATTERN,
                                                                    SPARSE_PRUNING_BLOCK_PATTERN_DEFAULT)
            output[SPARSE_PRUNING_DENSE_RATIO] = get_scalar_param(sub_param_dict, SPARSE_PRUNING_DENSE_RATIO,
                                                                  SPARSE_PRUNING_DENSE_RATIO_DEFAULT)
            assert output[SPARSE_PRUNING_DENSE_RATIO] > 0 and output[
                SPARSE_PRUNING_DENSE_RATIO] < 1, f"Invalid dense_ratio value. Must be less than 1"
            output[SPARSE_PRUNING_SCHEDULE_OFFSET_STRIDE] = get_scalar_param(
                sub_param_dict, SPARSE_PRUNING_SCHEDULE_OFFSET_STRIDE, SPARSE_PRUNING_SCHEDULE_OFFSET_STRIDE_DEFAULT)
            output[SPARSE_PRUNING_EXCLUDED_MODULES] = get_list_param(sub_param_dict, SPARSE_PRUNING_EXCLUDED_MODULES,
                                                                     SPARSE_PRUNING_EXCLUDED_MODULES_DEFAULT)
            output[SPARSE_PRUNING_SCHEDULE_OFFSET_END] = get_scalar_param(sub_param_dict,
                                                                          SPARSE_PRUNING_SCHEDULE_OFFSET_END,
                                                                          output[SPARSE_PRUNING_SCHEDULE_OFFSET])
            assert output[SPARSE_PRUNING_SCHEDULE_OFFSET] <= output[
                SPARSE_PRUNING_SCHEDULE_OFFSET_END], f"Invalid schedule_offset and schedule_offset_end values"
    else:
        output[SPARSE_PRUNING_ENABLED] = SPARSE_PRUNING_ENABLED_DEFAULT
        output[SPARSE_PRUNING_METHOD] = SPARSE_PRUNING_METHOD_DEFAULT
        output[SPARSE_PRUNING_SCHEDULE_OFFSET] = SPARSE_PRUNING_SCHEDULE_OFFSET_DEFAULT
    return output


def get_sparse_pruning_different_groups(param_dict):
    output = {}
    sub_param_dict = param_dict[DIFFERENT_GROUPS]

    def get_params(name, group_dict):
        assert SPARSE_PRUNING_DENSE_RATIO in group_dict.keys(
        ), f"{SPARSE_PRUNING_DENSE_RATIO} must be specified for sparse pruning group {name}"
        return group_dict

    for k, v in sub_param_dict.items():
        output[k] = {}
        output[k][DIFFERENT_GROUPS_PARAMETERS] = get_params(k, sub_param_dict[k][DIFFERENT_GROUPS_PARAMETERS])
        output[k][DIFFERENT_GROUPS_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_MODULE_SCOPE,
                                                                    DIFFERENT_GROUPS_MODULE_SCOPE_DEFAULT)
        output[k][DIFFERENT_GROUPS_RELATED_MODULE_SCOPE] = get_scalar_param(
            sub_param_dict[k], DIFFERENT_GROUPS_RELATED_MODULE_SCOPE, DIFFERENT_GROUPS_RELATED_MODULE_SCOPE_DEFAULT)

    return output


def get_row_pruning(param_dict):
    output = {}
    if ROW_PRUNING not in param_dict.keys():
        param_dict[ROW_PRUNING] = {SHARED_PARAMETERS: {}, DIFFERENT_GROUPS: {}}
    sub_param_dict = param_dict[ROW_PRUNING]
    # shared parameters
    output[SHARED_PARAMETERS] = get_row_pruning_shared_parameters(sub_param_dict)
    # each sub-groups
    if output[SHARED_PARAMETERS][ROW_PRUNING_ENABLED]:
        assert DIFFERENT_GROUPS in sub_param_dict.keys(
        ), f"Row Pruning is enabled, {DIFFERENT_GROUPS} must be specified"
    output[DIFFERENT_GROUPS] = get_row_pruning_different_groups(sub_param_dict)
    return output


def get_row_pruning_shared_parameters(param_dict):
    output = {}
    if SHARED_PARAMETERS in param_dict.keys():
        sub_param_dict = param_dict[SHARED_PARAMETERS]
        output[ROW_PRUNING_ENABLED] = get_scalar_param(sub_param_dict, ROW_PRUNING_ENABLED,
                                                       ROW_PRUNING_ENABLED_DEFAULT)
        output[ROW_PRUNING_METHOD] = get_scalar_param(sub_param_dict, ROW_PRUNING_METHOD, ROW_PRUNING_METHOD_DEFAULT)
        assert output[ROW_PRUNING_METHOD] in [
            ROW_PRUNING_METHOD_L1, ROW_PRUNING_METHOD_TOPK
        ], f"Invalid row pruning method. Supported types: [{ROW_PRUNING_METHOD_L1}, {ROW_PRUNING_METHOD_TOPK}]"
        output[ROW_PRUNING_SCHEDULE_OFFSET] = get_scalar_param(sub_param_dict, ROW_PRUNING_SCHEDULE_OFFSET,
                                                               ROW_PRUNING_SCHEDULE_OFFSET_DEFAULT)
    else:
        output[ROW_PRUNING_ENABLED] = ROW_PRUNING_ENABLED_DEFAULT
        output[ROW_PRUNING_METHOD] = ROW_PRUNING_METHOD_DEFAULT
        output[ROW_PRUNING_SCHEDULE_OFFSET] = ROW_PRUNING_SCHEDULE_OFFSET_DEFAULT
    return output


def get_row_pruning_different_groups(param_dict):
    output = {}
    sub_param_dict = param_dict[DIFFERENT_GROUPS]

    def get_params(name, group_dict):
        assert ROW_PRUNING_DENSE_RATIO in group_dict.keys(
        ), f"{ROW_PRUNING_DENSE_RATIO} must be specified for row pruning group {name}"
        return group_dict

    for k, v in sub_param_dict.items():
        output[k] = {}
        output[k][DIFFERENT_GROUPS_PARAMETERS] = get_params(k, sub_param_dict[k][DIFFERENT_GROUPS_PARAMETERS])
        output[k][DIFFERENT_GROUPS_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_MODULE_SCOPE,
                                                                    DIFFERENT_GROUPS_MODULE_SCOPE_DEFAULT)
        output[k][DIFFERENT_GROUPS_RELATED_MODULE_SCOPE] = get_scalar_param(
            sub_param_dict[k], DIFFERENT_GROUPS_RELATED_MODULE_SCOPE, DIFFERENT_GROUPS_RELATED_MODULE_SCOPE_DEFAULT)
    return output


def get_head_pruning(param_dict):
    output = {}
    if HEAD_PRUNING not in param_dict.keys():
        param_dict[HEAD_PRUNING] = {SHARED_PARAMETERS: {}, DIFFERENT_GROUPS: {}}
    sub_param_dict = param_dict[HEAD_PRUNING]
    # shared parameters
    output[SHARED_PARAMETERS] = get_head_pruning_shared_parameters(sub_param_dict)
    # each sub-groups
    if output[SHARED_PARAMETERS][HEAD_PRUNING_ENABLED]:
        assert DIFFERENT_GROUPS in sub_param_dict.keys(
        ), f"Head Pruning is enabled, {DIFFERENT_GROUPS} must be specified"
    output[DIFFERENT_GROUPS] = get_head_pruning_different_groups(sub_param_dict)
    return output


def get_head_pruning_shared_parameters(param_dict):
    output = {}
    if SHARED_PARAMETERS in param_dict.keys():
        sub_param_dict = param_dict[SHARED_PARAMETERS]
        output[HEAD_PRUNING_ENABLED] = get_scalar_param(sub_param_dict, HEAD_PRUNING_ENABLED,
                                                        HEAD_PRUNING_ENABLED_DEFAULT)
        output[HEAD_PRUNING_METHOD] = get_scalar_param(sub_param_dict, HEAD_PRUNING_METHOD,
                                                       HEAD_PRUNING_METHOD_DEFAULT)
        assert output[HEAD_PRUNING_METHOD] in [
            HEAD_PRUNING_METHOD_L1, HEAD_PRUNING_METHOD_TOPK
        ], f"Invalid head pruning method. Supported types: [{HEAD_PRUNING_METHOD_L1}, {HEAD_PRUNING_METHOD_TOPK}]"
        output[HEAD_PRUNING_SCHEDULE_OFFSET] = get_scalar_param(sub_param_dict, HEAD_PRUNING_SCHEDULE_OFFSET,
                                                                HEAD_PRUNING_SCHEDULE_OFFSET_DEFAULT)
        if output[HEAD_PRUNING_ENABLED]:
            assert HEAD_PRUNING_NUM_HEADS in sub_param_dict.keys(
            ), f"{HEAD_PRUNING_NUM_HEADS} must be specified for head pruning"
            output[HEAD_PRUNING_NUM_HEADS] = sub_param_dict[HEAD_PRUNING_NUM_HEADS]
    else:
        output[HEAD_PRUNING_ENABLED] = HEAD_PRUNING_ENABLED_DEFAULT
        output[HEAD_PRUNING_METHOD] = HEAD_PRUNING_METHOD_DEFAULT
        output[HEAD_PRUNING_SCHEDULE_OFFSET] = HEAD_PRUNING_SCHEDULE_OFFSET_DEFAULT
    return output


def get_head_pruning_different_groups(param_dict):
    output = {}
    sub_param_dict = param_dict[DIFFERENT_GROUPS]

    def get_params(name, group_dict):
        assert HEAD_PRUNING_DENSE_RATIO in group_dict.keys(
        ), f"dense_ratio must be specified for head pruning group {name}"
        return group_dict

    for k, v in sub_param_dict.items():
        output[k] = {}
        output[k][DIFFERENT_GROUPS_PARAMETERS] = get_params(k, sub_param_dict[k][DIFFERENT_GROUPS_PARAMETERS])
        output[k][DIFFERENT_GROUPS_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_MODULE_SCOPE,
                                                                    DIFFERENT_GROUPS_MODULE_SCOPE_DEFAULT)
        output[k][DIFFERENT_GROUPS_RELATED_MODULE_SCOPE] = get_scalar_param(
            sub_param_dict[k], DIFFERENT_GROUPS_RELATED_MODULE_SCOPE, DIFFERENT_GROUPS_RELATED_MODULE_SCOPE_DEFAULT)
    return output


def get_channel_pruning(param_dict):
    output = {}
    if CHANNEL_PRUNING not in param_dict.keys():
        param_dict[CHANNEL_PRUNING] = {SHARED_PARAMETERS: {}, DIFFERENT_GROUPS: {}}
    sub_param_dict = param_dict[CHANNEL_PRUNING]
    # shared parameters
    output[SHARED_PARAMETERS] = get_channel_pruning_shared_parameters(sub_param_dict)
    # each sub-groups
    if output[SHARED_PARAMETERS][CHANNEL_PRUNING_ENABLED]:
        assert DIFFERENT_GROUPS in sub_param_dict.keys(
        ), f"Sparse Pruning is enabled, {DIFFERENT_GROUPS} must be specified"
    output[DIFFERENT_GROUPS] = get_channel_pruning_different_groups(sub_param_dict)
    return output


def get_channel_pruning_shared_parameters(param_dict):
    output = {}
    if SHARED_PARAMETERS in param_dict.keys():
        sub_param_dict = param_dict[SHARED_PARAMETERS]
        output[CHANNEL_PRUNING_ENABLED] = get_scalar_param(sub_param_dict, CHANNEL_PRUNING_ENABLED,
                                                           CHANNEL_PRUNING_ENABLED_DEFAULT)
        output[CHANNEL_PRUNING_METHOD] = get_scalar_param(sub_param_dict, CHANNEL_PRUNING_METHOD,
                                                          CHANNEL_PRUNING_METHOD_DEFAULT)
        assert output[CHANNEL_PRUNING_METHOD] in [
            CHANNEL_PRUNING_METHOD_L1, CHANNEL_PRUNING_METHOD_TOPK
        ], f"Invalid channel pruning method. Supported types: [{CHANNEL_PRUNING_METHOD_L1}, {CHANNEL_PRUNING_METHOD_TOPK}]"
        output[CHANNEL_PRUNING_SCHEDULE_OFFSET] = get_scalar_param(sub_param_dict, CHANNEL_PRUNING_SCHEDULE_OFFSET,
                                                                   CHANNEL_PRUNING_SCHEDULE_OFFSET_DEFAULT)
    else:
        output[CHANNEL_PRUNING_ENABLED] = CHANNEL_PRUNING_ENABLED_DEFAULT
        output[CHANNEL_PRUNING_METHOD] = CHANNEL_PRUNING_METHOD_DEFAULT
        output[CHANNEL_PRUNING_SCHEDULE_OFFSET] = CHANNEL_PRUNING_SCHEDULE_OFFSET_DEFAULT
    return output


def get_channel_pruning_different_groups(param_dict):
    output = {}
    sub_param_dict = param_dict[DIFFERENT_GROUPS]

    def get_params(name, group_dict):
        assert CHANNEL_PRUNING_DENSE_RATIO in group_dict.keys(
        ), f"{CHANNEL_PRUNING_DENSE_RATIO} must be specified for channel pruning group {name}"
        return group_dict

    for k, v in sub_param_dict.items():
        output[k] = {}
        output[k][DIFFERENT_GROUPS_PARAMETERS] = get_params(k, sub_param_dict[k][DIFFERENT_GROUPS_PARAMETERS])
        output[k][DIFFERENT_GROUPS_MODULE_SCOPE] = get_scalar_param(sub_param_dict[k], DIFFERENT_GROUPS_MODULE_SCOPE,
                                                                    DIFFERENT_GROUPS_MODULE_SCOPE_DEFAULT)
        output[k][DIFFERENT_GROUPS_RELATED_MODULE_SCOPE] = get_scalar_param(
            sub_param_dict[k], DIFFERENT_GROUPS_RELATED_MODULE_SCOPE, DIFFERENT_GROUPS_RELATED_MODULE_SCOPE_DEFAULT)

    return output
