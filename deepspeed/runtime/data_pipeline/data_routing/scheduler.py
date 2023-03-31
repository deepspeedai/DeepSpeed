# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math

from deepspeed.utils import logger
# from deepspeed.runtime.lr_schedules import WarmupLR
from ..constants import *

#####based on the paper random-ltd: https://arxiv.org/abs/2211.11586


class BaseScheduler(object):

    def __init__(self):
        self.state = {}

    def __fixed_root_get_value(self, global_steps, root_degree=None):
        s_state = self.state[RANDOM_LTD_SCHEDULE_CONFIG]
        if root_degree is None:
            root_degree = s_state['root_degree']
        next_seq = (float(global_steps) / s_state[RANDOM_LTD_REQUIRE_STEP])**(1.0 / root_degree)
        next_seq = math.floor(next_seq * (self.state[RANDOM_LTD_MAX_VALUE] - self.state[RANDOM_LTD_MIN_VALUE]) +
                              self.state[RANDOM_LTD_MIN_VALUE])
        next_seq -= (next_seq % s_state[RANDOM_LTD_INCREASE_STEP])
        next_seq = min(next_seq, self.state[RANDOM_LTD_MAX_VALUE])
        return next_seq

    def get_value(self, global_steps):
        if self.state[RANDOM_LTD_SCHEDULER_TYPE] == 'fixed_linear':
            return self.__fixed_root_get_value(global_steps, 1)
        else:
            raise RuntimeError('Unsupported random LTD schedule type')


class RandomLTDScheduler(BaseScheduler):

    def __init__(self, config):
        super().__init__()
        self.model_layer_num = config[RANDOM_LTD_TOTAL_LAYER_NUM]
        self.random_ltd_layer_num = config[RANDOM_LTD_LAYER_NUM]
        self.config_schedule = config[RANDOM_LTD_SCHEDULER]
        self.global_batch_size = config[RANDOM_LTD_GLOBAL_BATCH_SIZE]
        self.reset_to_init()

        if config[RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE][RANDOM_LTD_LAYER_TOKEN_LR_ENABLED]:
            logger.warning("**********Work In Progress************")
            raise NotImplementedError

        self.state[RANDOM_LTD_CONSUMED_LAYER_TOKENS] = 0

        # self.first_step = True
    def get_total_layer_tokens(self, train_iters):
        for step in range(train_iters):
            self.update_seq(step)
        return self.state[RANDOM_LTD_CONSUMED_LAYER_TOKENS]

    def reset_to_init(self):
        if self.config_schedule is not None:
            self.state[RANDOM_LTD_MIN_VALUE] = self.config_schedule[RANDOM_LTD_MIN_VALUE]
            self.state[RANDOM_LTD_MAX_VALUE] = self.config_schedule[RANDOM_LTD_MAX_VALUE]
            self.state[RANDOM_LTD_CURRENT_VALUE] = self.config_schedule[RANDOM_LTD_MIN_VALUE]
            self.state[RANDOM_LTD_SCHEDULE_CONFIG] = self.config_schedule[RANDOM_LTD_SCHEDULE_CONFIG]
            self.state[RANDOM_LTD_SCHEDULER_TYPE] = self.config_schedule[RANDOM_LTD_SCHEDULER_TYPE]
        self.state[RANDOM_LTD_CONSUMED_LAYER_TOKENS] = 0
        self.state[RANDOM_LTD_CURR_STEP] = -1

    def get_current_seq(self):
        return self.state[RANDOM_LTD_CURRENT_VALUE]

    def set_current_seq(self, seq_length):
        self.state[RANDOM_LTD_CURRENT_VALUE] = seq_length

    def get_random_ltd_layer_num(self):
        return self.random_ltd_layer_num

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def update_seq(self, global_steps):
        if self.state[RANDOM_LTD_CURRENT_VALUE] < self.state[RANDOM_LTD_MAX_VALUE]:
            self.state[RANDOM_LTD_CURRENT_VALUE] = self.get_value(global_steps)
        if global_steps != self.state[RANDOM_LTD_CURR_STEP]:
            self.state[RANDOM_LTD_CONSUMED_LAYER_TOKENS] += self.global_batch_size*(self.state[RANDOM_LTD_CURRENT_VALUE] * self.random_ltd_layer_num \
                + self.state[RANDOM_LTD_MAX_VALUE] * (self.model_layer_num - self.random_ltd_layer_num))
            self.state[RANDOM_LTD_CURR_STEP] = global_steps

    def state_dict(self):
        return {
            RANDOM_LTD_CONSUMED_LAYER_TOKENS: self.state[RANDOM_LTD_CONSUMED_LAYER_TOKENS],
            RANDOM_LTD_CURR_STEP: self.state[RANDOM_LTD_CURR_STEP],
            RANDOM_LTD_CURRENT_VALUE: self.state[RANDOM_LTD_CURRENT_VALUE],
            RANDOM_LTD_MIN_VALUE: self.state[RANDOM_LTD_MIN_VALUE],
            RANDOM_LTD_MAX_VALUE: self.state[RANDOM_LTD_MAX_VALUE],
        }

    def load_state_dict(self, state_dict):
        self.state[RANDOM_LTD_CONSUMED_LAYER_TOKENS] = state_dict[RANDOM_LTD_CONSUMED_LAYER_TOKENS]
        self.state[RANDOM_LTD_CURR_STEP] = state_dict[RANDOM_LTD_CURR_STEP]
        self.state[RANDOM_LTD_CURRENT_VALUE] = state_dict[RANDOM_LTD_CURRENT_VALUE]
        self.state[RANDOM_LTD_MIN_VALUE] = state_dict[RANDOM_LTD_MIN_VALUE]
        self.state[RANDOM_LTD_MAX_VALUE] = state_dict[RANDOM_LTD_MAX_VALUE]
