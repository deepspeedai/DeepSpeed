# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import hjson

from ..constants import AUTOTUNING, AUTOTUNING_METRIC_PATH
from .base_tuner import BaseTuner
from .cost_model import XGBoostCostModel
from .utils import *
from ..utils import *
import numbers
from ..constants import AUTOTUNING_METRIC_LATENCY

INIT_NUM = 2


class ModelBasedTuner(BaseTuner):
    """Exploring the search space with a cost model"""

    def __init__(self, exps: list, resource_manager, metric, tuning_space):
        super().__init__(exps, resource_manager, metric)
        self.tuning_space = tuning_space
        self.best_iter = 0

        self.all_configs = [e['ds_config'] for e in exps]
        self.num_all_configs = len(self.all_configs)

        self.dims = dict_to_dims(self.tuning_space)

        logger.info(f"Create config dim: {self.dims}, all configs: {self.num_all_configs}")

        self.visited = set([])

        self.trials = []
        self.trial_pt = 0

        init_num = min(INIT_NUM, self.num_all_configs)

        for _ in range(init_num):
            exp_feature = np.random.randint(self.num_all_configs)
            exp_feature = 0
            while exp_feature in self.visited:
                exp_feature = np.random.randint(self.num_all_configs)
            self.trials.append(exp_feature)
            self.visited.add(exp_feature)

        self.cost_model = XGBoostCostModel("rank")

        self.evaluated_configs = []
        self.evaluated_perf = []

        self.train_ct = 0

        self.random_exploration_ratio = 0.2  # do random exploration

    def find_estimated_top_configs(self):
        """Use the cost model to predict the estimated performance of configurations and find the top ones for the next round of evaluation"""

        configs = []

        for c in self.all_configs:
            flattened_ds_config = flatten(c)
            feature_val = []
            for k, v in flattened_ds_config.items():
                if isinstance(v, numbers.Number):
                    feature_val.append(v)
            configs.append(feature_val)
        # print(configs)
        # TODO the current implementation requires that all configs have the same shape.
        configs = np.array(configs, dtype=np.float32)
        estimates = self.cost_model.predict(configs)

        n = len(estimates)
        top_idx = np.argsort(estimates)
        top_idx_ret = top_idx if self.metric == AUTOTUNING_METRIC_LATENCY else top_idx[::-1][:n]

        # top_configs = [self.all_configs[i] for i in top_idx]

        return top_idx_ret

    def next_batch(self, sample_size):
        sampled_batch = []

        counter = 0
        while counter < sample_size:

            if len(self.visited) >= self.num_all_configs:
                break

            while self.trial_pt < len(self.trials):
                logger.debug(f"trials: {self.trials}")
                # Select top promising trials
                index = self.trials[self.trial_pt]
                if index not in self.visited:
                    break
                self.trial_pt += 1

            # To avoid over-exploitation, randomly select one that has not been explored.
            rand = np.random.rand()
            if rand < self.random_exploration_ratio:
                # Do normal selection
                feature = np.random.choice(self.trials)
                while index in self.visited:
                    index = np.random.randint(self.num_all_configs)

            # Need to track both the sampled configs and indices

            sampled_batch.append(self.all_exps[index])
            self.visited.add(index)
            counter += 1

        return sampled_batch

    def has_next(self):
        return len(self.visited) < self.num_all_configs

    def update(self):
        for exp_id, (exp, err) in self.rm.finished_experiments.items():
            feature_val = []
            if err:
                logger.info(
                    f"Skipping exp_id = {exp_id}, exp_name = {exp['name']}, the experiment did not run successfully with error = {err}, thus a metrics.txt does not exist for it. Please check the stderr.log in {exp['result_dir']}"
                )
                ds_config = exp["ds_config"]
                flattened_ds_config = flatten(ds_config)
                for k, v in flattened_ds_config.items():
                    if isinstance(v, numbers.Number):
                        feature_val.append(v)
                self.evaluated_configs.append(feature_val)
                self.evaluated_perf.append(0.0)
                continue

            p = exp["ds_config"][AUTOTUNING][AUTOTUNING_METRIC_PATH]
            with open(p, 'r') as f:
                results = hjson.load(f)
                curr_iter = results[self.metric]
                logger.debug(f"parsing the results for {exp_id}， Result is {curr_iter}")

                ds_config = exp["ds_config"]
                flattened_ds_config = flatten(ds_config)
                for k, v in flattened_ds_config.items():
                    if isinstance(v, numbers.Number):
                        feature_val.append(v)
                self.evaluated_configs.append(feature_val)
                self.evaluated_perf.append(curr_iter)

        logger.debug(f"**Evaluated configs: {len(self.evaluated_configs)}, evaluated perf: {self.evaluated_perf}")

        self.cost_model.fit(self.evaluated_configs, self.evaluated_perf)

        estimated_top_configs = self.find_estimated_top_configs()

        self.trials = estimated_top_configs
        self.trial_pt = 0
        self.train_ct += 1
