# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import tempfile

PDSH_LAUNCHER = 'pdsh'
PDSH_MAX_FAN_OUT = 1024

OPENMPI_LAUNCHER = 'openmpi'
MPICH_LAUNCHER = 'mpich'
IMPI_LAUNCHER = 'impi'
SLURM_LAUNCHER = 'slurm'
MVAPICH_LAUNCHER = 'mvapich'
# /tmp doesn't exist on Windows; use the platform's actual temp directory instead.
MVAPICH_TMP_HOSTFILE = os.path.join(tempfile.gettempdir(), 'deepspeed_mvapich_hostfile')

ELASTIC_TRAINING_ID_DEFAULT = "123456789"
