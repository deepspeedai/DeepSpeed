"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from deepspeed.runtime.config_utils import get_scalar_param, DeepSpeedConfigObject

#########################################
#  DeepSpeed Activation Checkpointing
#########################################
# Activation Checkpointing Allows to save memory by only keeping a select few
#activations for the backpropagation.
ACTIVATION_CHKPT_FORMAT = '''
Activation Checkpointing should be configured as:
"session_params": {
  "activation_checkpointing": {
    "partitioned_activations": [true|false],
    "number_checkpoints": 100,
    "contiguous_memory_optimization": [true|false],
    "cpu_checkpointing": [true|false]
    "profile": [true|false],
    "synchronize_checkpoint_boundary": [true|false],
    }
}
'''

ACT_CHKPT_PARTITION_ACTIVATIONS = 'partition_activations'
ACT_CHKPT_PARTITION_ACTIVATIONS_DEFAULT = False

ACT_CHKPT_NUMBER_CHECKPOINTS = 'number_checkpoints'
ACT_CHKPT_NUMBER_CHECKPOINTS_DEFAULT = None

ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION = 'contiguous_memory_optimization'
ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION_DEFAULT = False

ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY = 'synchronize_checkpoint_boundary'
ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY_DEFAULT = False

ACT_CHKPT_PROFILE = 'profile'
ACT_CHKPT_PROFILE_DEFAULT = False

ACT_CHKPT_CPU_CHECKPOINTING = 'cpu_checkpointing'
ACT_CHKPT_CPU_CHECKPOINTING_DEFAULT = False

ACT_CHKPT = 'activation_checkpointing'

ACT_CHKPT_DEFAULT = {
    ACT_CHKPT_PARTITION_ACTIVATIONS: ACT_CHKPT_PARTITION_ACTIVATIONS_DEFAULT,
    ACT_CHKPT_NUMBER_CHECKPOINTS: ACT_CHKPT_NUMBER_CHECKPOINTS_DEFAULT,
    ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION:
    ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION_DEFAULT,
    ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY:
    ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY_DEFAULT,
    ACT_CHKPT_PROFILE: ACT_CHKPT_PROFILE_DEFAULT,
    ACT_CHKPT_CPU_CHECKPOINTING: ACT_CHKPT_CPU_CHECKPOINTING_DEFAULT
}


class DeepSpeedActivationCheckpointingConfig(DeepSpeedConfigObject):
    def __init__(self, param_dict):
        super(DeepSpeedActivationCheckpointingConfig, self).__init__()

        self.partition_activations = None
        self.contiguous_memory_optimization = None
        self.cpu_checkpointing = None
        self.number_checkpoints = None
        self.synchronize_checkpoint_boundary = None
        self.profile = None

        if ACT_CHKPT in param_dict.keys():
            act_chkpt_config_dict = param_dict[ACT_CHKPT]
        else:
            act_chkpt_config_dict = ACT_CHKPT_DEFAULT

        self._initialize(act_chkpt_config_dict)

    def _initialize(self, act_chkpt_config_dict):
        self.partition_activations = get_scalar_param(
            act_chkpt_config_dict,
            ACT_CHKPT_PARTITION_ACTIVATIONS,
            ACT_CHKPT_PARTITION_ACTIVATIONS_DEFAULT)

        self.contiguous_memory_optimization = get_scalar_param(
            act_chkpt_config_dict,
            ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION,
            ACT_CHKPT_CONTIGUOUS_MEMORY_OPTIMIZATION_DEFAULT)

        self.cpu_checkpointing = get_scalar_param(act_chkpt_config_dict,
                                                  ACT_CHKPT_CPU_CHECKPOINTING,
                                                  ACT_CHKPT_CPU_CHECKPOINTING_DEFAULT)

        self.number_checkpoints = get_scalar_param(act_chkpt_config_dict,
                                                   ACT_CHKPT_NUMBER_CHECKPOINTS,
                                                   ACT_CHKPT_NUMBER_CHECKPOINTS_DEFAULT)

        self.profile = get_scalar_param(act_chkpt_config_dict,
                                        ACT_CHKPT_PROFILE,
                                        ACT_CHKPT_PROFILE_DEFAULT)

        self.synchronize_checkpoint_boundary = get_scalar_param(
            act_chkpt_config_dict,
            ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY,
            ACT_CHKPT_SYNCHRONIZE_CHECKPOINT_BOUNDARY_DEFAULT)
