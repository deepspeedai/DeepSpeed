# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

SCRIPT_PREFIX = '_aio_bench'
WRITE_OP_DESC = 'write'
READ_OP_DESC = 'read'
READ_IO_DIR = f'{SCRIPT_PREFIX}_{READ_OP_DESC}_io'
WRITE_IO_DIR = f'{SCRIPT_PREFIX}_{WRITE_OP_DESC}_io'
BENCH_LOG_DIR = f'{SCRIPT_PREFIX}_logs'
READ_LOG_DIR = f'{SCRIPT_PREFIX}_{READ_OP_DESC}_logs'
WRITE_LOG_DIR = f'{SCRIPT_PREFIX}_{WRITE_OP_DESC}_logs'
