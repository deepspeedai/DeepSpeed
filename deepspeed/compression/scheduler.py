# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .compress import get_module_name
from .constants import *
from .helper import recursive_getattr
from deepspeed.utils import logger


class compression_scheduler():
    '''
    Used to schedule different compression methods
    '''

    def __init__(self, model, compression_config):
        self.model = model
        self.compression_config = compression_config
        self.make_init()
        self.training_steps = 0
        self.weight_quantization_enabled = False

        self.verbose = {
            WEIGHT_QUANTIZATION: False,
            ACTIVATION_QUANTIZATION: False,
            SPARSE_PRUNING: False,
            HEAD_PRUNING: False,
            ROW_PRUNING: False,
            CHANNEL_PRUNING: False
        }

    def make_init(self):
        self.different_compression_methods = {}
        for method, method_content in self.compression_config.items():
            if LAYER_REDUCTION in method:
                continue
            self.different_compression_methods[method] = {
                TECHNIQUE_ENABLED: False,
                SHARED_PARAMETERS: None,
                DIFFERENT_GROUPS: []
            }
            exist_module_name = set()
            shared_parameters = method_content[SHARED_PARAMETERS]
            self.different_compression_methods[method][TECHNIQUE_ENABLED] = shared_parameters[TECHNIQUE_ENABLED]
            self.different_compression_methods[method][SHARED_PARAMETERS] = shared_parameters

            for group_name, method_parameters in method_content[DIFFERENT_GROUPS].items():
                module_name_list = []
                for key_word in method_parameters[DIFFERENT_GROUPS_MODULE_SCOPE]:
                    module_name, exist_module_name = get_module_name(group_name,
                                                                     self.model,
                                                                     key_word,
                                                                     exist_module_name,
                                                                     verbose=False)
                    module_name_list.extend(module_name)
                if module_name_list:
                    self.different_compression_methods[method][DIFFERENT_GROUPS].append(
                        [group_name, module_name_list,
                         method_parameters.copy().pop('params')])

    def check_weight_quantization(self):
        # check weight quantization
        wq = self.different_compression_methods[WEIGHT_QUANTIZATION]
        if not wq[TECHNIQUE_ENABLED]:
            return
        else:
            shared_parameters = wq[SHARED_PARAMETERS]
            if self.training_steps >= shared_parameters[TECHNIQUE_SCHEDULE_OFFSET]:
                for group_name, module_name_list, method_parameters in wq[DIFFERENT_GROUPS]:
                    for module_name in module_name_list:
                        module = recursive_getattr(self.model, module_name)
                        module.weight_quantization_enabled = True

                if not self.verbose[WEIGHT_QUANTIZATION]:
                    logger.info(f'Weight quantization is enabled at step {self.training_steps}')
                    self.weight_quantization_enabled = True
                    self.verbose[WEIGHT_QUANTIZATION] = True

    def check_activation_quantization(self):
        # check activation quantization
        aq = self.different_compression_methods[ACTIVATION_QUANTIZATION]
        if not aq[TECHNIQUE_ENABLED]:
            return
        else:
            shared_parameters = aq[SHARED_PARAMETERS]
            if self.training_steps >= shared_parameters[TECHNIQUE_SCHEDULE_OFFSET]:
                for group_name, module_name_list, method_parameters in aq[DIFFERENT_GROUPS]:
                    for module_name in module_name_list:
                        module = recursive_getattr(self.model, module_name)
                        module.activation_quantization_enabled = True
                if not self.verbose[ACTIVATION_QUANTIZATION]:
                    logger.info(f'Activation quantization is enabled at step {self.training_steps}')
                    self.verbose[ACTIVATION_QUANTIZATION] = True

    def check_sparse_pruning(self):
        # check sparse pruning
        sp = self.different_compression_methods[SPARSE_PRUNING]
        if not sp[TECHNIQUE_ENABLED]:
            return
        else:
            shared_parameters = sp[SHARED_PARAMETERS]
            if shared_parameters[TECHNIQUE_SCHEDULE_OFFSET] <= self.training_steps <= shared_parameters[
                    TECHNIQUE_SCHEDULE_OFFSET_END]:
                for group_name, module_name_list, method_parameters in sp[DIFFERENT_GROUPS]:
                    for module_name in module_name_list:
                        module = recursive_getattr(self.model, module_name)
                        module.sparse_pruning_enabled = True
                if not self.verbose[SPARSE_PRUNING]:
                    logger.info(f'Sparse pruning is enabled at step {self.training_steps}')
                    self.verbose[SPARSE_PRUNING] = True

    def check_head_pruning(self):
        # check head pruning
        hp = self.different_compression_methods[HEAD_PRUNING]
        if not hp[TECHNIQUE_ENABLED]:
            return
        else:
            shared_parameters = hp[SHARED_PARAMETERS]
            if self.training_steps >= shared_parameters[TECHNIQUE_SCHEDULE_OFFSET]:
                for group_name, module_name_list, method_parameters in hp[DIFFERENT_GROUPS]:
                    for module_name in module_name_list:
                        module = recursive_getattr(self.model, module_name)
                        module.head_pruning_enabled = True
                if not self.verbose[HEAD_PRUNING]:
                    logger.info(f'Head pruning is enabled at step {self.training_steps}')
                    self.verbose[HEAD_PRUNING] = True

    def check_row_pruning(self):
        # check row pruning
        rp = self.different_compression_methods[ROW_PRUNING]
        if not rp[TECHNIQUE_ENABLED]:
            return
        else:
            shared_parameters = rp[SHARED_PARAMETERS]
            if self.training_steps >= shared_parameters[TECHNIQUE_SCHEDULE_OFFSET]:
                for group_name, module_name_list, method_parameters in rp[DIFFERENT_GROUPS]:
                    for module_name in module_name_list:
                        module = recursive_getattr(self.model, module_name)
                        module.row_pruning_enabled = True
                if not self.verbose[ROW_PRUNING]:
                    logger.info(f'Row pruning is enabled at step {self.training_steps}')
                    self.verbose[ROW_PRUNING] = True

    def check_channel_pruning(self):
        # check channel pruning
        cp = self.different_compression_methods[CHANNEL_PRUNING]
        if not cp[TECHNIQUE_ENABLED]:
            return
        else:
            shared_parameters = cp[SHARED_PARAMETERS]
            if self.training_steps >= shared_parameters[TECHNIQUE_SCHEDULE_OFFSET]:
                for group_name, module_name_list, method_parameters in cp[DIFFERENT_GROUPS]:
                    for module_name in module_name_list:
                        module = recursive_getattr(self.model, module_name)
                        module.channel_pruning_enabled = True
                if not self.verbose[CHANNEL_PRUNING]:
                    logger.info(f'Channel pruning is enabled at step {self.training_steps}')
                    self.verbose[CHANNEL_PRUNING] = True

    def check_all_modules(self):
        # check all different compression methods we have
        self.check_weight_quantization()
        self.check_activation_quantization()
        self.check_sparse_pruning()
        self.check_head_pruning()
        self.check_row_pruning()
        self.check_channel_pruning()

    def step(self, step_zero_check=False):
        if not step_zero_check:
            self.training_steps += 1
        self.check_all_modules()
