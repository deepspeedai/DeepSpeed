"""
Copyright 2020 The Microsoft DeepSpeed Team
"""

import json
from .constants import *


class ElasticityError(Exception):
    """
    Base exception for all elasticity related errors
    """
    pass


class ElasticityConfigError(ElasticityError):
    """
    Elasticity configuration error
    """
    pass


class ElasticityIncompatibleWorldSize(ElasticityError):
    """
    Attempting to run a world size that is incompatible with a given elastic config
    """
    pass


class ElasticityConfig:
    """
    Elastic config object, constructed from a param dictionary that only contains elastic
    config parameters, example below:

    If elasticity is enabled, user must specify (at least) max_train_batch_size
    and micro_batch_sizes.

    {
        "enabled": true,
        "max_train_batch_size": 2000,
        "micro_batch_sizes": [2,4,6],
        "min_gpus": 1,
        "max_gpus" : 10000
        "min_time": 20
        "ignore_non_elastic_batch_info": false
        "version": 0.1
    }
    """
    def __init__(self, param_dict):
        self.enabled = param_dict.get(ENABLED, ENABLED_DEFAULT)
        if self.enabled:
            if MAX_ACCEPTABLE_BATCH_SIZE in param_dict:
                self.max_acceptable_batch_size = param_dict[MAX_ACCEPTABLE_BATCH_SIZE]
            else:
                raise ElasticityConfigError(
                    f"Elasticity config missing {MAX_ACCEPTABLE_BATCH_SIZE}")
            if MICRO_BATCHES in param_dict:
                self.micro_batches = param_dict[MICRO_BATCHES]
            else:
                raise ElasticityConfigError(f"Elasticity config missing {MICRO_BATCHES}")
        else:
            self.max_acceptable_batch_size = param_dict.get(
                MAX_ACCEPTABLE_BATCH_SIZE,
                MAX_ACCEPTABLE_BATCH_SIZE_DEFAULT)
            self.micro_batches = param_dict.get(MICRO_BATCHES, MICRO_BATCHES_DEFAULT)

        if not isinstance(self.micro_batches, list):
            raise ElasticityConfigError(
                f"Elasticity expected value of {MICRO_BATCHES} to be a "
                f"list of micro batches, instead is: {type(self.micro_batches)}, containing: {self.micro_batches}"
            )

        if not all(map(lambda m: isinstance(m, int), self.micro_batches)):
            raise ElasticityConfigError(
                f"Elasticity expected {MICRO_BATCHES} to only contain a list of integers, "
                f"instead contains: f{self.micro_batches}")

        if not all(map(lambda m: m > 0, self.micro_batches)):
            raise ElasticityConfigError(
                f"Elasticity expected {MICRO_BATCHES} to only contain positive integers, "
                f"instead contains: f{self.micro_batches}")

        self.min_gpus = param_dict.get(MIN_GPUS, MIN_GPUS_DEFAULT)
        self.max_gpus = param_dict.get(MAX_GPUS, MAX_GPUS_DEFAULT)
        if self.min_gpus < 1 or self.max_gpus < 1:
            raise ElasticityConfigError(
                "Elasticity min/max gpus must be > 0, "
                f"given min_gpus: {self.min_gpus}, max_gpus: {self.max_gpus}")
        if self.max_gpus < self.min_gpus:
            raise ElasticityConfigError(
                "Elasticity min_gpus cannot be greater than max_gpus, "
                f"given min_gpus: {self.min_gpus}, max_gpus: {self.max_gpus}")

        self.min_time = param_dict.get(MIN_TIME, MIN_TIME_DEFAULT)
        if self.min_time < 0:
            raise ElasticityConfigError(
                f"Elasticity min time needs to be >= 0: given {self.min_time}")

        self.version = param_dict.get(VERSION, VERSION_DEFAULT)
        self.prefer_larger_batch_size = param_dict.get(PREFER_LARGER_BATCH,
                                                       PREFER_LARGER_BATCH_DEFAULT)
        self.ignore_non_elastic_batch_info = param_dict.get(
            IGNORE_NON_ELASTIC_BATCH_INFO,
            IGNORE_NON_ELASTIC_BATCH_INFO_DEFAULT)

    def repr(self):
        return self.__dict__

    def __repr__(self):
        return json.dumps(self.__dict__, sort_keys=True, indent=4)
