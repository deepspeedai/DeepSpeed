# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
from deepspeed.utils import logger
from .constants import *


class CurriculumScheduler(object):

    def __init__(self, config):
        super().__init__()
        self.state = {}
        assert CURRICULUM_LEARNING_MIN_DIFFICULTY in config, \
            f"Curriculum learning requires the config '{CURRICULUM_LEARNING_MIN_DIFFICULTY}'"
        assert CURRICULUM_LEARNING_MAX_DIFFICULTY in config, \
            f"Curriculum learning requires the config '{CURRICULUM_LEARNING_MAX_DIFFICULTY}'"
        assert CURRICULUM_LEARNING_SCHEDULE_TYPE in config, \
            f"Curriculum learning requires the config '{CURRICULUM_LEARNING_SCHEDULE_TYPE}'"
        self.state[CURRICULUM_LEARNING_MIN_DIFFICULTY] = config[CURRICULUM_LEARNING_MIN_DIFFICULTY]
        self.state[CURRICULUM_LEARNING_MAX_DIFFICULTY] = config[CURRICULUM_LEARNING_MAX_DIFFICULTY]
        self.state[CURRICULUM_LEARNING_CURRENT_DIFFICULTY] = config[CURRICULUM_LEARNING_MIN_DIFFICULTY]
        self.state[CURRICULUM_LEARNING_SCHEDULE_TYPE] = config[CURRICULUM_LEARNING_SCHEDULE_TYPE]
        self.first_step = True
        if config[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_FIXED_DISCRETE:
            """
            The schedule_config is a list of difficulty and a list of max
            step belonging to each difficulty. Example json config:
            "schedule_config": {
              "difficulty": [1,2,3],
              "max_step": [5,10]
            }
            The "max_step" has one less element than "difficulty", because
            the last difficulty will be used for all following steps.
            The self.state[CURRICULUM_LEARNING_SCHEDULE_CONFIG] is a dictionary of
            difficulty : [max step for this difficulty, next difficulty].
            """
            assert CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY in config[CURRICULUM_LEARNING_SCHEDULE_CONFIG], \
                f"Curriculum learning with fixed_discrete schedule requires the schedule_config '{CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY}'"
            assert CURRICULUM_LEARNING_SCHEDULE_MAX_STEP in config[CURRICULUM_LEARNING_SCHEDULE_CONFIG], \
                f"Curriculum learning with fixed_discrete schedule requires the schedule_config '{CURRICULUM_LEARNING_SCHEDULE_MAX_STEP}'"
            assert len(config[CURRICULUM_LEARNING_SCHEDULE_CONFIG][CURRICULUM_LEARNING_SCHEDULE_MAX_STEP]) > 0
            assert len(config[CURRICULUM_LEARNING_SCHEDULE_CONFIG][CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY]) > 0
            assert len(config[CURRICULUM_LEARNING_SCHEDULE_CONFIG][CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY]) == len(
                config[CURRICULUM_LEARNING_SCHEDULE_CONFIG][CURRICULUM_LEARNING_SCHEDULE_MAX_STEP]) + 1
            self.state[CURRICULUM_LEARNING_SCHEDULE_CONFIG] = config[CURRICULUM_LEARNING_SCHEDULE_CONFIG]
        elif config[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_FIXED_ROOT:
            """
            The schedule_config includes:
            total_curriculum_step: how many steps the curriculum learning takes to go
            from min difficulty to max difficulty.
            difficulty_step: the difficulty level determined every time must
            be a multiple of this difficulty_step. This is used to determine
            the step of difficulty increase, and to ensure the use of NVIDIA
            Tensor Core acceleration (requires multiple of 8 (FP16) or
            16 (INT8)).
            root_degree: the degree of the root function. Degree of 2 means
            square root and degree of 3 means cube root. Degree of 1 is
            equivalent to linear.
            "schedule_config": {
              "total_curriculum_step": 30000,
              "difficulty_step": 8,
              "root_degree": 2
            }
            """
            assert CURRICULUM_LEARNING_SCHEDULE_TOTAL_STEP in config[CURRICULUM_LEARNING_SCHEDULE_CONFIG], \
                f"Curriculum learning with fixed_root schedule requires the schedule_config '{CURRICULUM_LEARNING_SCHEDULE_TOTAL_STEP}'"
            assert CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP in config[CURRICULUM_LEARNING_SCHEDULE_CONFIG], \
                f"Curriculum learning with fixed_root schedule requires the schedule_config '{CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP}'"
            assert CURRICULUM_LEARNING_SCHEDULE_ROOT_DEGREE in config[CURRICULUM_LEARNING_SCHEDULE_CONFIG], \
                f"Curriculum learning with fixed_root schedule requires the schedule_config '{CURRICULUM_LEARNING_SCHEDULE_ROOT_DEGREE}'"
            if config[CURRICULUM_LEARNING_SCHEDULE_CONFIG][CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP] % 8 != 0:
                logger.warning(
                    f'When using seqlen metric, the difficulty_step for curriculum learning has to be multiple of 8 (for FP16 data) or 16 (for INT8 data) to enable NVIDIA Tensor Core acceleration. Disregard this warning if this is unrelated to your metric/hardware.'
                )
            self.state[CURRICULUM_LEARNING_SCHEDULE_CONFIG] = config[CURRICULUM_LEARNING_SCHEDULE_CONFIG]
        elif config[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_FIXED_LINEAR:
            """
            The schedule_config is the same as CURRICULUM_LEARNING_SCHEDULE_FIXED_ROOT but without the
            root_degree.
            "schedule_config": {
              "total_curriculum_step": 30000,
              "difficulty_step": 8
            }
            """
            assert CURRICULUM_LEARNING_SCHEDULE_TOTAL_STEP in config[CURRICULUM_LEARNING_SCHEDULE_CONFIG], \
                f"Curriculum learning with fixed_linear schedule requires the schedule_config '{CURRICULUM_LEARNING_SCHEDULE_TOTAL_STEP}'"
            assert CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP in config[CURRICULUM_LEARNING_SCHEDULE_CONFIG], \
                f"Curriculum learning with fixed_linear schedule requires the schedule_config '{CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP}'"
            if config[CURRICULUM_LEARNING_SCHEDULE_CONFIG][CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP] % 8 != 0:
                logger.warning(
                    f'When using seqlen metric, the difficulty_step for curriculum learning has to be multiple of 8 (for FP16 data) or 16 (for INT8 data) to enable NVIDIA Tensor Core acceleration. Disregard this warning if this is unrelated to your metric/hardware.'
                )
            self.state[CURRICULUM_LEARNING_SCHEDULE_CONFIG] = config[CURRICULUM_LEARNING_SCHEDULE_CONFIG]
        elif config[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_CUSTOM:
            """
            Fully customized schedule. User need to provide a custom schedule
            function by using the set_custom_curriculum_learning_schedule API
            in deepspeed/runtime/engine.py
            """
            self.custom_get_difficulty = None
        else:
            raise RuntimeError('Unsupported curriculum schedule type')

    def get_current_difficulty(self):
        return self.state[CURRICULUM_LEARNING_CURRENT_DIFFICULTY]

    def set_current_difficulty(self, difficulty):
        self.state[CURRICULUM_LEARNING_CURRENT_DIFFICULTY] = difficulty

    def set_custom_get_difficulty(self, schedule_function):
        self.custom_get_difficulty = schedule_function

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def __fixed_discrete_get_difficulty(self, global_steps):
        s_state = self.state[CURRICULUM_LEARNING_SCHEDULE_CONFIG]
        if global_steps > s_state[CURRICULUM_LEARNING_SCHEDULE_MAX_STEP][-1]:
            return s_state[CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY][-1]
        for i in range(len(s_state[CURRICULUM_LEARNING_SCHEDULE_MAX_STEP])):
            if global_steps <= s_state[CURRICULUM_LEARNING_SCHEDULE_MAX_STEP][i]:
                return s_state[CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY][i]

    def __fixed_root_get_difficulty(self, global_steps, root_degree=None):
        s_state = self.state[CURRICULUM_LEARNING_SCHEDULE_CONFIG]
        if root_degree is None:
            root_degree = s_state[CURRICULUM_LEARNING_SCHEDULE_ROOT_DEGREE]
        next_difficulty = (float(global_steps) / s_state[CURRICULUM_LEARNING_SCHEDULE_TOTAL_STEP])**(1.0 / root_degree)
        next_difficulty = math.floor(
            next_difficulty *
            (self.state[CURRICULUM_LEARNING_MAX_DIFFICULTY] - self.state[CURRICULUM_LEARNING_MIN_DIFFICULTY]) +
            self.state[CURRICULUM_LEARNING_MIN_DIFFICULTY])
        next_difficulty -= (next_difficulty % s_state[CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP])
        next_difficulty = min(next_difficulty, self.state[CURRICULUM_LEARNING_MAX_DIFFICULTY])
        return next_difficulty

    def get_difficulty(self, global_steps):
        if self.state[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_FIXED_DISCRETE:
            return self.__fixed_discrete_get_difficulty(global_steps)
        elif self.state[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_FIXED_LINEAR:
            return self.__fixed_root_get_difficulty(global_steps, 1)
        elif self.state[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_FIXED_ROOT:
            return self.__fixed_root_get_difficulty(global_steps)
        elif self.state[CURRICULUM_LEARNING_SCHEDULE_TYPE] == CURRICULUM_LEARNING_SCHEDULE_CUSTOM:
            return self.custom_get_difficulty(global_steps)
        else:
            raise RuntimeError('Unsupported curriculum schedule type')

    def update_difficulty(self, global_steps):
        if self.state[CURRICULUM_LEARNING_CURRENT_DIFFICULTY] < self.state[CURRICULUM_LEARNING_MAX_DIFFICULTY]:
            self.state[CURRICULUM_LEARNING_CURRENT_DIFFICULTY] = self.get_difficulty(global_steps)
        return self.state[CURRICULUM_LEARNING_CURRENT_DIFFICULTY]
