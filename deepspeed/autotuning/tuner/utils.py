# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import numpy as np
import itertools
from ..utils import *
import collections.abc


def index_to_feature(p, dims):
    """convert index form (single integer) to feature form (vector)"""
    feature = []
    for dim in dims:
        feature.append(p % dim)
        p //= dim
    return feature


def feature_to_index(feature, dims):
    """convert feature form (vector) to index form (single integer)"""
    p = 0
    for j, k in enumerate(feature):
        print("j:", "k:", k, "dims", dims[:j])
        p += int(np.prod(dims[:j])) * k
    return p


def dict_to_dims(tuning_space):

    dims = []

    for key, val in tuning_space.items():
        if isinstance(val, dict):
            dims.extend(dict_to_dims(val))
        elif isinstance(val, list):
            dims.append(len(val))
        else:
            dims.append(1)

    return dims


def gen_combinations(d: dict):
    keys, values = d.keys(), d.values()
    for v in values:
        if not isinstance(v, list):
            v = [v]
    values_choices = (gen_combinations(v) if isinstance(v, dict) else get_list(v) for v in values)
    for comb in itertools.product(*values_choices):
        yield dict(zip(keys, comb))


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_to_feature(feature_dict, keys, max_value=None):
    """Extract values from dict"""
    feature = []
    for key, val in feature_dict.items():  # First level
        if key not in keys:
            continue
        if val is None or val == "auto" or key == "autotuning" or val == "":
            continue
        if isinstance(val, dict):
            feature.append(dict_to_feature(val, max_value))
        else:
            feature.append(float(val))

    # normalization, should not matter in tree models
    if max_value is not None:
        norm_feature = []
        for f, mv in zip(feature, max_value):
            norm_feature.append(f / mv)
        feature = norm_feature

    return feature
