'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
import sys
import types

from . import ops

from .runtime.engine import DeepSpeedEngine
from .runtime.engine import ADAM_OPTIMIZER, LAMB_OPTIMIZER
from .runtime.pipe.engine import PipelineEngine
from .runtime.lr_schedules import add_tuning_arguments
from .runtime.config import DeepSpeedConfig
from .runtime.activation_checkpointing import checkpointing
from .ops.transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig
from .utils import log_dist

from .pipe import PipelineModule

from .git_version_info import version, git_hash, git_branch


def _parse_version(version_str):
    '''Parse a version string and extract the major, minor, and patch versions.'''
    import re
    matched = re.search('^(\d+)\.(\d+)\.(\d+)', version_str)
    return int(matched.group(1)), int(matched.group(2)), int(matched.group(3))


# Export version information
__version__ = version
__version_major__, __version_minor__, __version_patch__ = _parse_version(__version__)
__git_hash__ = git_hash
__git_branch__ = git_branch

# Provide backwards compatability with old deepspeed.pt module structure, should hopefully not be used
pt = types.ModuleType('pt', 'dummy pt module for backwards compatability')
deepspeed = sys.modules[__name__]
setattr(deepspeed, 'pt', pt)
setattr(deepspeed.pt, 'deepspeed_utils', deepspeed.runtime.utils)
sys.modules['deepspeed.pt'] = deepspeed.pt
sys.modules['deepspeed.pt.deepspeed_utils'] = deepspeed.runtime.utils
setattr(deepspeed.pt, 'deepspeed_config', deepspeed.runtime.config)
sys.modules['deepspeed.pt.deepspeed_config'] = deepspeed.runtime.config


def initialize(args,
               model,
               optimizer=None,
               model_parameters=None,
               training_data=None,
               lr_scheduler=None,
               mpu=None,
               dist_init_required=None,
               collate_fn=None,
               config_params=None):
    """Initialize the DeepSpeed Engine.

    Arguments:
        args: a dictionary containing local_rank and deepspeed_config
            file location

        model: Required: nn.module class before apply any wrappers

        optimizer: Optional: a user defined optimizer, this is typically used instead of defining
            an optimizer in the DeepSpeed json config.

        model_parameters: Optional: An iterable of torch.Tensors or dicts.
            Specifies what Tensors should be optimized.

        training_data: Optional: Dataset of type torch.utils.data.Dataset

        lr_scheduler: Optional: Learning Rate Scheduler Object. It should define a get_lr(),
            step(), state_dict(), and load_state_dict() methods

        mpu: Optional: A model parallelism unit object that implements
            get_{model,data}_parallel_{rank,group,world_size}()

        dist_init_required: Optional: None will auto-initialize torch.distributed if needed,
            otherwise the user can force it to be initialized or not via boolean.

        collate_fn: Optional: Merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.

    Returns:
        A tuple of ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``

        * ``engine``: DeepSpeed runtime engine which wraps the client model for distributed training.

        * ``optimizer``: Wrapped optimizer if a user defined ``optimizer`` is supplied, or if
          optimizer is specified in json config else ``None``.

        * ``training_dataloader``: DeepSpeed dataloader if ``training_data`` was supplied,
          otherwise ``None``.

        * ``lr_scheduler``: Wrapped lr scheduler if user ``lr_scheduler`` is passed, or
          if ``lr_scheduler`` specified in JSON configuration. Otherwise ``None``.
    """
    log_dist("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(
        __version__,
        __git_hash__,
        __git_branch__),
             ranks=[0])

    if not isinstance(model, PipelineModule):
        engine = DeepSpeedEngine(args=args,
                                 model=model,
                                 optimizer=optimizer,
                                 model_parameters=model_parameters,
                                 training_data=training_data,
                                 lr_scheduler=lr_scheduler,
                                 mpu=mpu,
                                 dist_init_required=dist_init_required,
                                 collate_fn=collate_fn,
                                 config_params=config_params)
    else:
        assert mpu is None, "mpu must be None with pipeline parallelism"
        engine = PipelineEngine(args=args,
                                model=model,
                                optimizer=optimizer,
                                model_parameters=model_parameters,
                                training_data=training_data,
                                lr_scheduler=lr_scheduler,
                                mpu=model.mpu(),
                                dist_init_required=dist_init_required,
                                collate_fn=collate_fn,
                                config_params=config_params)

    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler
    ]
    return tuple(return_items)


def _add_core_arguments(parser):
    r"""Helper (internal) function to update an argument parser with an argument group of the core DeepSpeed arguments.
        The core set of DeepSpeed arguments include the following:
        1) --deepspeed: boolean flag to enable DeepSpeed
        2) --deepspeed_config <json file path>: path of a json configuration file to configure DeepSpeed runtime.

        This is a helper function to the public add_config_arguments()

    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    group = parser.add_argument_group('DeepSpeed', 'DeepSpeed configurations')

    group.add_argument(
        '--deepspeed',
        default=False,
        action='store_true',
        help=
        'Enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)')

    group.add_argument('--deepspeed_config',
                       default=None,
                       type=str,
                       help='DeepSpeed json configuration file.')

    group.add_argument(
        '--deepscale',
        default=False,
        action='store_true',
        help=
        'Deprecated enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)'
    )

    group.add_argument('--deepscale_config',
                       default=None,
                       type=str,
                       help='Deprecated DeepSpeed json configuration file.')

    group.add_argument(
        '--deepspeed_mpi',
        default=False,
        action='store_true',
        help=
        "Run via MPI, this will attempt to discover the necessary variables to initialize torch "
        "distributed from the MPI environment")

    return parser


def add_config_arguments(parser):
    r"""Update the argument parser to enabling parsing of DeepSpeed command line arguments.
        The set of DeepSpeed arguments include the following:
        1) --deepspeed: boolean flag to enable DeepSpeed
        2) --deepspeed_config <json file path>: path of a json configuration file to configure DeepSpeed runtime.

    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    parser = _add_core_arguments(parser)

    return parser
