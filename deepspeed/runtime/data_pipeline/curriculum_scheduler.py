'''
Copyright 2021 The Microsoft DeepSpeed Team
'''
import math
from deepspeed.utils import logger


class CurriculumScheduler(object):
    def __init__(self, config):
        super().__init__()
        self.state = {}
        assert "curriculum_type" in config, "Curriculum learning requires the config 'curriculum_type'"
        assert "min_difficulty" in config, "Curriculum learning requires the config 'min_difficulty'"
        assert "max_difficulty" in config, "Curriculum learning requires the config 'max_difficulty'"
        assert "schedule_type" in config, "Curriculum learning requires the config 'schedule_type'"
        self.state['min_difficulty'] = config['min_difficulty']
        self.state['max_difficulty'] = config['max_difficulty']
        self.state['current_difficulty'] = config['min_difficulty']
        self.state['schedule_type'] = config['schedule_type']
        self.first_step = True
        if config['schedule_type'] == 'fixed_discrete':
            """
            The schedule_config is a list of difficulty and a list of max
            step belonging to each difficulty. Example json config:
            "schedule_config": {
              "difficulty": [1,2,3],
              "max_step": [5,10]
            }
            The "max_step" has one less element than "difficulty", because
            the last difficulty will be used for all following steps.
            The self.state['schedule'] is a dictionary of
            difficulty : [max step for this difficulty, next difficulty].
            """
            assert "difficulty" in config['schedule_config'], "Curriculum learning with fixed_discrete schedule requires the schedule_config 'difficulty'"
            assert "max_step" in config['schedule_config'], "Curriculum learning with fixed_discrete schedule requires the schedule_config 'max_step'"
            assert len(config['schedule_config']['max_step']) > 0
            assert len(config['schedule_config']['difficulty']) > 0
            assert len(config['schedule_config']['difficulty']) == len(
                config['schedule_config']['max_step']) + 1
            self.state['schedule'] = config['schedule_config']
        elif config['schedule_type'] == 'fixed_root':
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
            assert "total_curriculum_step" in config['schedule_config'], "Curriculum learning with fixed_root schedule requires the schedule_config 'total_curriculum_step'"
            assert "difficulty_step" in config['schedule_config'], "Curriculum learning with fixed_root schedule requires the schedule_config 'difficulty_step'"
            assert "root_degree" in config['schedule_config'], "Curriculum learning with fixed_root schedule requires the schedule_config 'root_degree'"
            if config['schedule_config']['difficulty_step'] % 8 != 0:
                logger.warning(
                    f'The difficulty_step for curriculum learning has to be multiple of 8 (for FP16 data) or 16 (for INT8 data) to enable NVIDIA Tensor Core acceleration. Disregard this warning if this is unrelated to your hardware.'
                )
            self.state['schedule'] = config['schedule_config']
        elif config['schedule_type'] == 'fixed_linear':
            """
            The schedule_config is the same as 'fixed_root' but without the
            root_degree.
            "schedule_config": {
              "total_curriculum_step": 30000,
              "difficulty_step": 8
            }
            """
            assert "total_curriculum_step" in config['schedule_config'], "Curriculum learning with fixed_linear schedule requires the schedule_config 'total_curriculum_step'"
            assert "difficulty_step" in config['schedule_config'], "Curriculum learning with fixed_linear schedule requires the schedule_config 'difficulty_step'"
            if config['schedule_config']['difficulty_step'] % 8 != 0:
                logger.warning(
                    f'The difficulty_step for curriculum learning has to be multiple of 8 (for FP16 data) or 16 (for INT8 data) to enable NVIDIA Tensor Core acceleration. Disregard this warning if this is unrelated to your hardware.'
                )
            self.state['schedule'] = config['schedule_config']
        else:
            raise RuntimeError('Unsupported curriculum schedule type')

    def get_current_difficulty(self):
        return self.state['current_difficulty']

    def set_current_difficulty(self, difficulty):
        self.state['current_difficulty'] = difficulty

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def __fixed_discrete_get_difficulty(self, global_steps):
        s_state = self.state['schedule']
        if global_steps > s_state['max_step'][-1]:
            return s_state['difficulty'][-1]
        for i in range(len(s_state['max_step'])):
            if global_steps <= s_state['max_step'][i]:
                return s_state['difficulty'][i]

    def __fixed_root_get_difficulty(self, global_steps, root_degree=None):
        s_state = self.state['schedule']
        if root_degree is None:
            root_degree = s_state['root_degree']
        next_difficulty = (float(global_steps) /
                           s_state['total_curriculum_step'])**(1.0 / root_degree)
        next_difficulty = math.floor(
            next_difficulty *
            (self.state['max_difficulty'] - self.state['min_difficulty']) +
            self.state['min_difficulty'])
        next_difficulty -= (next_difficulty % s_state['difficulty_step'])
        next_difficulty = min(next_difficulty, self.state['max_difficulty'])
        return next_difficulty

    def get_difficulty(self, global_steps):
        if self.state['schedule_type'] == 'fixed_discrete':
            return self.__fixed_discrete_get_difficulty(global_steps)
        elif self.state['schedule_type'] == 'fixed_linear':
            return self.__fixed_root_get_difficulty(global_steps, 1)
        elif self.state['schedule_type'] == 'fixed_root':
            return self.__fixed_root_get_difficulty(global_steps)
        else:
            raise RuntimeError('Unsupported curriculum schedule type')

    def update_difficulty(self, global_steps):
        if self.state['current_difficulty'] < self.state['max_difficulty']:
            self.state['current_difficulty'] = self.get_difficulty(global_steps)
        return self.state['current_difficulty']
