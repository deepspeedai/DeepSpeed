# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


#########################################
# Validation modes
#########################################
class ValidationMode:
    WARN = "WARN"
    IGNORE = "IGNORE"
    FAIL = "FAIL"


#########################################
# Checkpoint config params
#########################################
# "checkpoint": {tag_validation=["Ignore"|"Warn"|"Fail"]}
CHECKPOINT_FORMAT = '''
"checkpoint": {
  "tag_validation": [Ignore|Warn|Fail],
  "checkpoint_serialization": False,
  "writer": {
    "type": [mock|python|fast],
    "decoupled": [True|False]
    "io_buffer_size": 64e6,
    "io_buffer_double": True,
    "show_statistics": False,
    "data_parallel": [replica|socket|machine],
    "io_multiplier": 1,
  }
}
'''
CHECKPOINT = "checkpoint"
CHECKPOINT_TAG_VALIDATION = "tag_validation"
CHECKPOINT_TAG_VALIDATION_DEFAULT = ValidationMode.WARN
CHECKPOINT_TAG_VALIDATION_MODES = [ValidationMode.WARN, ValidationMode.IGNORE, ValidationMode.FAIL]

CHECKPOINT_SERIALIZATION = "checkpoint_serialization"
CHECKPOINT_SERIALIZATION_DEFAULT = True

CHECKPOINT_WRITER = "writer"
CHECKPOINT_WRITER_DEFAULT = None

CHECKPOINT_WRITER_TYPE = "type"


class CheckpointWriterType:
    MOCK = "MOCK"
    PYTHON = "PYTHON"
    FAST = "FAST"


CHECKPOINT_WRITER_TYPE_DEFAULT = CheckpointWriterType.FAST
CHECKPOINT_WRITER_TYPES = [CheckpointWriterType.MOCK, CheckpointWriterType.PYTHON, CheckpointWriterType.FAST]

CHECKPOINT_IO_BUFFER_SIZE = "io_buffer_size"
CHECKPOINT_IO_BUFFER_SIZE_DEFAULT = 64 * (1024**2)

CHECKPOINT_IO_BUFFER_DOUBLE = "io_buffer_double"
CHECKPOINT_IO_BUFFER_DOUBLE_DEFAULT = True

CHECKPOINT_IO_MULTIPLIER = "io_multiplier"
CHECKPOINT_IO_MULTIPLIER_DEFAULT = 1

CHECKPOINT_IO_STATISTICS = "show_statistics"
CHECKPOINT_IO_STATISTICS_DEFAULT = False

CHECKPOINT_DATA_PARALLEL = "data_parallel"
CHECKPOINT_DATA_PARALLEL_DEFAULT = None


class CheckpointDataParallel:
    REPLICA = "REPLICA"
    SOCKET = "SOCKET"
    MACHINE = "MACHINE"


CHECKPOINT_DATA_PARALLEL_UNITS = [
    CheckpointDataParallel.REPLICA, CheckpointDataParallel.SOCKET, CheckpointDataParallel.MACHINE
]

CHECKPOINT_WRITER_DECOUPLED = "decoupled"
CHECKPOINT_WRITER_DECOUPLED_DEFAULT = False
