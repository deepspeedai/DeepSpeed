# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import json
import numpy as np
import math
from packaging import version as pkg_version

from .config import ElasticityConfig, ElasticityConfigError, ElasticityError, \
    ElasticityIncompatibleWorldSize
from .constants import ELASTICITY, ENABLED, ENABLED_DEFAULT, LATEST_ELASTICITY_VERSION, \
    MINIMUM_DEEPSPEED_VERSION, DEEPSPEED_ELASTICITY_CONFIG
from ..git_version_info import version as __version__
from ..utils import logger

# Thirty eight smallest highly composite numbers. The list should
# be enough to support up to 720K batch size.
HCN_LIST = [
    1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680, 2520, 5040, 7560, 10080, 15120, 20160,
    25200, 27720, 45360, 50400, 55440, 83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280, 720720
]


def get_candidate_batch_sizes(base_list, max_acceptable_batch_size):
    candidate_batch_size = []
    for base in base_list:
        if base >= max_acceptable_batch_size:
            candidate_batch_size.append(base)
        else:
            value = max_acceptable_batch_size // base
            index = np.argmax(np.asarray(HCN_LIST) > value)
            candidate_batch_size.append(HCN_LIST[index - 1] * base)
    candidate_batch_size = list(set(candidate_batch_size))
    logger.info(f"Candidate batch size: {candidate_batch_size}")
    return candidate_batch_size


def get_valid_gpus(batch_size, micro_batches, min_valid_gpus, max_valid_gpus):
    valid_gpus = []
    for micro_batch in micro_batches:
        if batch_size % micro_batch == 0:

            max_gpus = batch_size // micro_batch
            if max_gpus >= min_valid_gpus and max_gpus <= max_valid_gpus:
                valid_gpus.append(max_gpus)

            # find all factors less than max_gpus / 2
            for i in range(1, max_gpus // 2 + 1):
                if i > max_valid_gpus:
                    break
                if i < min_valid_gpus:
                    continue
                if max_gpus % i == 0:
                    valid_gpus.append(i)
    valid_gpus = set(valid_gpus)
    valid_gpus = sorted(list(valid_gpus))
    return valid_gpus


def get_best_candidates(candidate_batch_sizes, micro_batches, min_gpus, max_gpus, prefer_larger):

    max_valid_gpus = 0
    valid_gpus = None
    final_batch_size = int(min(micro_batches))

    for batch_size in candidate_batch_sizes:

        current_valid_gpus = get_valid_gpus(batch_size, micro_batches, min_gpus, max_gpus)

        if (len(current_valid_gpus) > max_valid_gpus or (len(current_valid_gpus) == max_valid_gpus and
                                                         ((prefer_larger and batch_size > final_batch_size) or
                                                          (not prefer_larger and batch_size < final_batch_size)))):
            max_valid_gpus = len(current_valid_gpus)
            valid_gpus = current_valid_gpus
            final_batch_size = batch_size

    return final_batch_size, valid_gpus


def _get_compatible_gpus_v01(micro_batches,
                             max_acceptable_batch_size,
                             min_gpus=None,
                             max_gpus=None,
                             prefer_larger=True):
    '''We use two heuristics to compute the batch size
        1. We use the Lowest Common Multiple of the micro-batches
    as the base batch size and scale it by a HCN such that the result is
    the largest batch size less than the max_acceptable batch size
        2. We use each of the micro batches as a base and scale it
    by a HCN such that the result is the largest batch size less than the
    max_acceptable batch size.

    We then use brute force to count the number of compatible GPU count for
    each of the aforementioned cases, and return the batch size with the most number of
    compatible GPU counts in the min-max GPU range if provided, other wise
    we return the batch size with the most number of total compatible GPU counts.

    Returns:
        final_batch_size
        valid_gpus
    '''
    min_gpus = min_gpus or 1
    max_gpus = max_gpus or max_acceptable_batch_size // min(micro_batches)

    if not all(mb <= max_acceptable_batch_size for mb in micro_batches):
        raise ValueError(f"All micro batches must be less than \
            or equal to max_acceptable_batch_size: {max_acceptable_batch_size}")

    lcm = np.lcm.reduce(micro_batches)

    base_list = []
    base_list.extend(micro_batches)
    base_list.append(lcm)

    candidate_batch_sizes = get_candidate_batch_sizes(base_list, max_acceptable_batch_size)

    final_batch_size, valid_gpus = get_best_candidates(candidate_batch_sizes, micro_batches, min_gpus, max_gpus,
                                                       prefer_larger)

    return final_batch_size, valid_gpus


def _get_compatible_gpus_v02(micro_batches,
                             max_acceptable_batch_size,
                             current_num_gpus,
                             min_gpus=None,
                             max_gpus=None,
                             prefer_larger=True,
                             num_gpus_per_node=1,
                             model_parallel_size=1):
    '''
    Returns:
        final_batch_size
        valid_gpus
        micro-batch size
    '''
    if num_gpus_per_node % model_parallel_size != 0:
        raise ElasticityError(
            f"In Elasticity v0.2, number of GPUs per node:" \
            f"{num_gpus_per_node} should be divisible by " \
            f"model parallel size {model_parallel_size}")

    def get_microbatch(final_batch_size):
        candidate_microbatch = None

        for micro_batch in micro_batches:
            if final_batch_size // current_num_gpus % micro_batch == 0:
                if candidate_microbatch == None:
                    candidate_microbatch = micro_batch
                if prefer_larger and candidate_microbatch < micro_batch:
                    candidate_microbatch = micro_batch
        return candidate_microbatch

    dp_size_per_node = num_gpus_per_node // model_parallel_size

    final_batch_size, valid_world_size = _get_compatible_gpus_v01(
        micro_batches,
        int(max_acceptable_batch_size / dp_size_per_node),
        int(min_gpus / num_gpus_per_node),
        int(max_gpus / num_gpus_per_node),  # Passing number of max nodes as Elasticity v2 works at node level
        prefer_larger=prefer_larger)

    final_batch_size = int(final_batch_size) * dp_size_per_node
    valid_dp_world_size = [i * dp_size_per_node for i in valid_world_size]
    if current_num_gpus // model_parallel_size in valid_dp_world_size:
        candidate_microbatch = get_microbatch(final_batch_size)
        return final_batch_size, valid_dp_world_size, candidate_microbatch

    current_dp_size = (current_num_gpus / num_gpus_per_node) * dp_size_per_node
    candidate_batch_sizes = []
    for micro_batch in micro_batches:
        min_batch_size = micro_batch * current_dp_size

        factor = math.floor(max_acceptable_batch_size / float(min_batch_size))
        candidate_batch_sizes.append(factor * min_batch_size)

    used_microbatch = None
    if prefer_larger:
        candidate_batch_size = max(candidate_batch_sizes)
    else:
        candidate_batch_size = min(candidate_batch_sizes)

    candidate_microbatch = get_microbatch(candidate_batch_size)

    return candidate_batch_size, [int(current_dp_size)], candidate_microbatch


def _compatible_ds_version_check(target_deepspeed_version: str):
    min_version = pkg_version.parse(MINIMUM_DEEPSPEED_VERSION)
    target_version = pkg_version.parse(target_deepspeed_version)

    err_str = f"Target deepspeed version of {target_deepspeed_version} is not compatible " \
        f"with minimum version {MINIMUM_DEEPSPEED_VERSION} supporting elasticity."
    if target_version < min_version:
        raise ElasticityError(err_str)
    return True


def elasticity_enabled(ds_config: dict):
    if ELASTICITY not in ds_config:
        return False
    return ds_config[ELASTICITY].get(ENABLED, ENABLED_DEFAULT)


def ensure_immutable_elastic_config(runtime_elastic_config_dict: dict):
    """
    Ensure the resource scheduler saw the same elastic config we are using at runtime
    """
    if DEEPSPEED_ELASTICITY_CONFIG in os.environ:
        scheduler_elastic_config_dict = json.loads(os.environ[DEEPSPEED_ELASTICITY_CONFIG])
        scheduler_elastic_config = ElasticityConfig(scheduler_elastic_config_dict)
        runtime_elastic_config = ElasticityConfig(runtime_elastic_config_dict)
        err_str = "Elastic config '{}={}' seen by resource scheduler does not match config passed to runtime {}={}"
        if runtime_elastic_config.max_acceptable_batch_size != scheduler_elastic_config.max_acceptable_batch_size:
            raise ElasticityConfigError(
                err_str.format('max_acceptable_batch_size', scheduler_elastic_config.max_acceptable_batch_size,
                               'max_acceptable_batch_size', runtime_elastic_config.max_acceptable_batch_size))
        if runtime_elastic_config.micro_batches != scheduler_elastic_config.micro_batches:
            raise ElasticityConfigError(
                err_str.format('micro_batches', scheduler_elastic_config.micro_batches, 'micro_batches',
                               runtime_elastic_config.micro_batches))
        if runtime_elastic_config.version != scheduler_elastic_config.version:
            raise ElasticityConfigError(
                err_str.format('version', scheduler_elastic_config.version, 'version', runtime_elastic_config.version))
    else:
        logger.warning("Unable to find DEEPSPEED_ELASTICITY_CONFIG environment variable, cannot " \
            "guarantee resource scheduler will scale this job using compatible GPU counts.")


def compute_elastic_config(ds_config: dict, target_deepspeed_version: str, world_size=0, return_microbatch=False):
    """Core deepspeed elasticity API. Given an elastic config (similar to the example below)
    DeepSpeed will compute a total train batch size corresponding valid GPU count list that
    provides a high level of elasticity. Elasticity in this case means we are safe to scale
    the training job up/down across the GPU count list *without* any negative impacts on
    training convergence. This is achievable primarily due to DeepSpeed's gradient accumulation
    feature which allows us to decompose a global training batch size into:
    micro-batch-size * gradient-accumulation-steps * world-size.

    "elasticity": {
        "enabled": true,
        "max_train_batch_size": 2000,
        "micro_batch_sizes": [2,4,6],
        "min_gpus": 1,
        "max_gpus" : 10000
        "min_time": 20
        "version": 0.1
    }

    Intended to be called both by scheduling infrastructure and deepspeed runtime.
    For the same `ds_config` we should return deterministic results.

    Args:
        ds_config (dict): DeepSpeed config dictionary/json
        target_deepspeed_version (str): When called from scheduling
            infrastructure we want to ensure that the target deepspeed version is
            compatible with the elasticity version used in the backend.
        world_size (int, optional): Intended/current DP world size, will do some sanity
            checks to ensure world size is actually valid with the config.
        return_microbatch (bool, optional): whether to return micro batch size or not.

    Raises:
        ElasticityConfigError: Missing required elasticity config or elasticity disabled
        ElasticityError: If target deepspeed version is not compatible with current version

    Returns:
        final_batch_size (int): total batch size used for training
        valid_gpus (list(int)): list of valid GPU counts with this config
        micro_batch_size (int, optional): if world_size is provided will return
            specific micro batch size
    """
    if not isinstance(ds_config, dict):
        raise ValueError("Expected ds_config to be a dictionary but received " \
            f"a {type(ds_config)}, containing: {ds_config}")

    if ELASTICITY not in ds_config:
        raise ElasticityConfigError(f"'{ELASTICITY}' is missing from config json," \
            " please add it if running an elastic training job.")

    elastic_config_dict = ds_config[ELASTICITY]
    if not elastic_config_dict.get(ENABLED, ENABLED_DEFAULT):
        raise ElasticityConfigError("Elasticity is disabled, please enable it " \
            "('enabled':true) if running an elastic training job.")

    elastic_config = ElasticityConfig(elastic_config_dict)
    model_parallel_size = elastic_config.model_parallel_size
    num_gpus_per_node = elastic_config.num_gpus_per_node

    if model_parallel_size > 1 and float(elastic_config.version) != 0.2:
        raise ElasticityConfigError(f"Elasticity V{elastic_config.version} " \
            f"does not support model-parallel training. Given model-parallel size: " \
            f"{model_parallel_size}")

    if float(elastic_config.version) > LATEST_ELASTICITY_VERSION:
        raise ElasticityConfigError("Attempting to run elasticity version " \
            f"{elastic_config.version} but runtime only supports up " \
            f"to {LATEST_ELASTICITY_VERSION}")

    # Ensure target deepspeed version works with intended elasticity version
    if not _compatible_ds_version_check(target_deepspeed_version):
        raise ElasticityError("Unable to run elasticity on target deepspeed version of" \
            f" {target_deepspeed_version}, currently {__version__}")

    if float(elastic_config.version) == 0.1:
        final_batch_size, valid_gpus = _get_compatible_gpus_v01(
            micro_batches=elastic_config.micro_batches,
            max_acceptable_batch_size=elastic_config.max_acceptable_batch_size,
            min_gpus=elastic_config.min_gpus,
            max_gpus=elastic_config.max_gpus,
            prefer_larger=elastic_config.prefer_larger_batch_size)
        # ensure batch size is int dtype
        final_batch_size = int(final_batch_size)
    elif float(elastic_config.version) == 0.2:
        if world_size != 0:
            current_num_gpus = world_size
        else:
            if "WORLD_SIZE" in os.environ and \
                os.getenv('WORLD_SIZE').isnumeric():
                current_num_gpus = int(os.getenv('WORLD_SIZE'))
            else:
                WORLD_SIZE = os.getenv('WORLD_SIZE')
                raise ElasticityConfigError(
                    'Elasticity V 0.2 needs WORLD_SIZE '\
                    'to compute valid batch size. '\
                    'Either give it as argument to function compute_elastic_config '\
                    'or set it as an environment variable. '\
                    f'Value of WORLD_SIZE as environment variable is {WORLD_SIZE}')

        final_batch_size, valid_gpus, candidate_microbatch_size = _get_compatible_gpus_v02(
            micro_batches=elastic_config.micro_batches,
            max_acceptable_batch_size=elastic_config.max_acceptable_batch_size,
            current_num_gpus=current_num_gpus,
            min_gpus=elastic_config.min_gpus,
            max_gpus=elastic_config.max_gpus,
            prefer_larger=elastic_config.prefer_larger_batch_size,
            num_gpus_per_node=num_gpus_per_node,
            model_parallel_size=model_parallel_size)
        # ensure batch size is int dtype
        final_batch_size = int(final_batch_size)
    else:
        raise NotImplementedError(f"Unable to find elastic logic for version: {elastic_config.version}")

    logger.info(f"Valid World Size (GPUs / Model Parallel Size): {valid_gpus}")

    if world_size > 0:
        if world_size not in valid_gpus:
            raise ElasticityIncompatibleWorldSize(f"World size ({world_size}) is not valid " \
        f"with the current list of valid GPU counts: {valid_gpus}")

        # Pick largest valid micro batch size
        micro_batch_size = None
        for mbsz in sorted(list(set(elastic_config.micro_batches)), reverse=True):
            if final_batch_size // world_size % mbsz == 0:
                micro_batch_size = mbsz
                break
        assert micro_batch_size is not None, "Unable to find divisible micro batch size" \
            f" world_size={world_size}, final_batch_size={final_batch_size}, and " \
            f" micro_batches={elastic_config.micro_batches}."
        return final_batch_size, valid_gpus, micro_batch_size

    if return_microbatch:
        # Pick a valid micro batch size
        if float(elastic_config.version) == 0.2:
            return final_batch_size, valid_gpus, candidate_microbatch_size
        else:
            micro_batch_size = None
            for mbsz in sorted(list(set(elastic_config.micro_batches)), reverse=True):
                if final_batch_size // world_size % mbsz == 0:
                    micro_batch_size = mbsz
                    break
            assert micro_batch_size is not None, "Unable to find divisible micro batch size" \
                    f" world_size={world_size}, final_batch_size={final_batch_size}, and " \
                    f" micro_batches={elastic_config.micro_batches}."
            return final_batch_size, valid_gpus, micro_batch_size

    return final_batch_size, valid_gpus
