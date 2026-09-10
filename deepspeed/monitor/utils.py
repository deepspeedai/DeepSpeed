# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from packaging import version as pkg_version


def check_tb_availability():
    try:
        import tensorboard  # noqa: F401 # type: ignore
    except ImportError:
        print('If you want to use tensorboard logging, please `pip install tensorboard`')
        raise


def check_wandb_availability():
    try:
        import wandb  # noqa: F401 # type: ignore
    except ImportError:
        print(
            'If you want to use wandb logging, please `pip install wandb` and follow the instructions at https://docs.wandb.ai/quickstart'
        )
        raise


def check_comet_availability():
    try:
        import comet_ml
        comet_version = pkg_version.parse(comet_ml.__version__)
        if comet_version < pkg_version.Version("3.41.0"):
            raise ImportError("`comet_ml` must have at least version 3.41.0")
    except ImportError:
        print('If you want to use comet logging, please `pip install "comet_ml>=3.41.0"`')
        raise


def check_trackio_availability():
    try:
        import trackio  # noqa: F401 # type: ignore
    except ImportError:
        print(
            'If you want to use Trackio logging, please `pip install trackio` and follow the instructions at https://github.com/huggingface/trackio'
        )
        raise