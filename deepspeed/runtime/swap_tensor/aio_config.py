'''
Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
'''

from deepspeed.runtime.config_utils import get_scalar_param
from deepspeed.runtime.swap_tensor.constants import *

AIO_DEFAULT_DICT = {
    AIO_BLOCK_SIZE: AIO_BLOCK_SIZE_DEFAULT,
    AIO_QUEUE_DEPTH: AIO_QUEUE_DEPTH_DEFAULT,
    AIO_THREAD_COUNT: AIO_THREAD_COUNT_DEFAULT,
    AIO_SINGLE_SUBMIT: AIO_SINGLE_SUBMIT_DEFAULT,
    AIO_OVERLAP_EVENTS: AIO_OVERLAP_EVENTS_DEFAULT
}


def get_aio_config(param_dict):
    if AIO in param_dict.keys() and param_dict[AIO] is not None:
        aio_dict = param_dict[AIO]
        return {
            AIO_BLOCK_SIZE:
            get_scalar_param(aio_dict,
                             AIO_BLOCK_SIZE,
                             AIO_BLOCK_SIZE_DEFAULT),
            AIO_QUEUE_DEPTH:
            get_scalar_param(aio_dict,
                             AIO_QUEUE_DEPTH,
                             AIO_QUEUE_DEPTH_DEFAULT),
            AIO_THREAD_COUNT:
            get_scalar_param(aio_dict,
                             AIO_THREAD_COUNT,
                             AIO_THREAD_COUNT_DEFAULT),
            AIO_SINGLE_SUBMIT:
            get_scalar_param(aio_dict,
                             AIO_SINGLE_SUBMIT,
                             AIO_SINGLE_SUBMIT_DEFAULT),
            AIO_OVERLAP_EVENTS:
            get_scalar_param(aio_dict,
                             AIO_OVERLAP_EVENTS,
                             AIO_OVERLAP_EVENTS_DEFAULT)
        }

    return AIO_DEFAULT_DICT
