'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

from deepspeed.pt.deepspeed_light import DeepSpeedLight
from deepspeed.pt.deepspeed_light import ADAM_OPTIMIZER, LAMB_OPTIMIZER
from deepspeed.pt.deepspeed_lr_schedules import add_tuning_arguments

try:
    from deepspeed.version_info import git_hash, git_branch
except ImportError:
    git_hash = None
    git_branch = None

__version__ = 0.1
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

        model_parameters: Optional: An iterable of torch.Tensor s or dicts.
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
        tuple: engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler

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


def add_core_arguments(parser):
    r"""Adds argument group for enabling deepspeed and providing deepspeed config file

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
    r"""Updates the parser to parse DeepSpeed arguments

    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    parser = add_core_arguments(parser)

    return parser
