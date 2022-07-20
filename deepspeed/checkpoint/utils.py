import os
from .constants import (MODEL_FILE_PREFIX,
                        MODEL_FILE_SUFFIX,
                        OPTIM_FILE_SUFFIX,
                        ZERO_FILE_PREFIX)


def get_model_ckpt_name_for_rank(base_folder, mp_rank_str):
    ckpt_name = os.path.join(
        base_folder,
        MODEL_FILE_PREFIX + mp_rank_str + MODEL_FILE_SUFFIX,
    )
    return ckpt_name


def get_zero_ckpt_name_for_rank(base_folder, dp_rank, mp_rank):
    zero_prefix = f'{ZERO_FILE_PREFIX}{dp_rank}'
    mp_rank_string = f'_{MODEL_FILE_PREFIX}{mp_rank:02d}'
    zero_ckpt_name = os.path.join(
        base_folder,
        zero_prefix + mp_rank_string + OPTIM_FILE_SUFFIX,
    )
    return zero_ckpt_name


def get_layer_ckpt_name_for_rank(base_folder, layer_id, tp_rank):
    ckpt_file = f'{layer_id}-model_{tp_rank:02d}{MODEL_FILE_SUFFIX}'
    ckpt_path = os.path.join(base_folder, ckpt_file)
    return ckpt_path
