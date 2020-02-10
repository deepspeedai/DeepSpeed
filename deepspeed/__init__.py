'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

from deepspeed.pt.deepspeed_light import DeepSpeedLight
from deepspeed.pt.deepspeed_light import ADAM_OPTIMIZER, LAMB_OPTIMIZER
from deepspeed.pt.deepspeed_lr_schedules import add_tuning_arguments

try:
    from deepspeed.git_version_info import git_hash, git_branch
except ImportError:
    git_hash = None
    git_branch = None

# Export version information
__version_major__ = 0
__version_minor__ = 1
__version_patch__ = 0
__version__ = '.'.join(
    map(str,
        [__version_major__,
         __version_minor__,
         __version_patch__]))
__git_hash__ = git_hash
__git_branch__ = git_branch


def initialize(args,
               model,
               optimizer=None,
               model_parameters=None,
               training_data=None,
               lr_scheduler=None,
               mpu=None,
               dist_init_required=True,
               collate_fn=None):
    r"""Initialize the DeepSpeed Engine.

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
            get_model/data_parallel_group/rank/size()

        dist_init_required: Optional: Initializes torch.distributed

        collate_fn: Optional: Merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.

    Return:
        The following tuple is returned by this function.
        tuple: engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler

        engine: DeepSpeed runtime engine which wraps the client model for distributed training.

        engine.optimizer: Wrapped optimizer if a user defined optimizer is passed or
            if optimizer is specified in json config else None.

        engine.training_dataloader: DeepSpeed dataloader if training data was passed else None.

        engine.lr_scheduler: Wrapped lr scheduler if user lr scheduler is passed
            or if lr scheduler specified in json config else None.


    """
    print("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(
        __version__,
        __git_hash__,
        __git_branch__),
          flush=True)

    engine = DeepSpeedLight(args=args,
                            model=model,
                            optimizer=optimizer,
                            model_parameters=model_parameters,
                            training_data=training_data,
                            lr_scheduler=lr_scheduler,
                            mpu=mpu,
                            dist_init_required=dist_init_required,
                            collate_fn=collate_fn)

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

    group.add_argument('--deepspeed',
                       default=False,
                       action='store_true',
                       help='Enable DeepSpeed')

    group.add_argument('--deepspeed_config',
                       default=None,
                       type=str,
                       help='DeepSpeed json configuration file.')

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
