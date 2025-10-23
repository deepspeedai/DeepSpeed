# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import multiprocessing as mp
from .ds_aio_args import get_validated_args
from .io_engine import io_engine_multiprocessing


def ds_io_main():
    print('Testing DeepNVMe python frontend')

    args = get_validated_args()
    mp.set_start_method('spawn', force=True)
    multiprocess_function = io_engine_multiprocessing
    multiprocess_function(args, args.read)


if __name__ == "__main__":
    ds_io_main()
