"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""
"""
Collection of DeepSpeed configuration utilities
"""

from collections import Counter


def get_scalar_param(param_dict, param_name, param_default_value):
    return param_dict.get(param_name, param_default_value)


def dict_raise_error_on_duplicate_keys(ordered_pairs):
    """Reject duplicate keys."""
    d = dict((k, v) for k, v in ordered_pairs)
    if len(d) != len(ordered_pairs):
        counter = Counter([pair[0] for pair in ordered_pairs])
        keys = [key for key, value in counter.items() if value > 1]
        raise ValueError("Duplicate keys in DeepSpeed config: {}".format(keys))
    return d
