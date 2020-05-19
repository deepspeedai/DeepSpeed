"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""
"""
Collection of DeepSpeed configuration utilities
"""


def get_scalar_param(param_dict, param_name, param_default_value):
    if param_name in param_dict.keys():
        return param_dict[param_name]
    else:
        return param_default_value


def dict_raise_error_on_duplicate_keys(ordered_pairs):
    """Reject duplicate keys."""
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            raise ValueError("Duplicate key in DeepSpeed config: %r" % (k, ))
        else:
            d[k] = v
    return d
