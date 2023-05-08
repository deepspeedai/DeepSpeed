# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys

from deepspeed.autotuning.constants import *
from deepspeed.autotuning.utils import write_experiments
from deepspeed.utils import logger


class BaseTuner:

    def __init__(self, exps, resource_manager, metric):
        self.all_exps = exps
        self.rm = resource_manager
        self.best_iter = 0
        self.best_exp = None
        self.best_metric_val = None
        self.metric = metric if metric else AUTOTUNING_METRIC_DEFAULT
        logger.info(f"total number of exps =  {len(self.all_exps)}")

    def has_next(self):
        """Whether there exists more configurations for evaluation"""
        if len(self.all_exps) > 0:
            return True
        else:
            return False

    def next_batch(self, sample_size):
        """Select the next batch of configurations for evaluation"""
        raise NotImplementedError

    def update(self):
        """"Update the tuner with what configurations have been evaluated and their performance results"""

    def tune(self, sample_size=1, n_trials=1000, early_stopping=None):
        i = 0
        try:
            while i < n_trials and self.has_next():
                # Select the next batch of configuration for evaluation
                sampled_exps = self.next_batch(sample_size)
                # Generate experiments for measurement of performance
                exp_paths = write_experiments(sampled_exps, self.rm.exps_dir)
                self.rm.schedule_experiments(exp_paths)
                self.rm.run()
                exp, metric_val = self.rm.parse_results(self.metric)
                if self.best_exp == None or self.best_metric_val == None or (metric_val
                                                                             and metric_val > self.best_metric_val):
                    # logger.info(f"tuner finds better = {exp}")
                    self.best_exp = exp
                    self.best_metric_val = metric_val
                    self.best_iter = i

                i += len(sampled_exps)

                # Update the tuner with evaluated performance results
                self.update()

                self.rm.clear()

                # Early stop if no more promising configurations are likely to be found
                if early_stopping and i >= self.best_iter + early_stopping:
                    logger.info(
                        f"Tuner early stopped at iteration {i}. Best iteration is {self.best_iter}. Early stopping threshold is {early_stopping}"
                    )
                    break
            return i
        except:
            logger.info("Tuner Error:", sys.exc_info()[0])
            return i
