# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import pytest
from deepspeed.utils.zero_to_fp32 import get_optim_files


@pytest.mark.parametrize('num_checkpoints', [1, 2, 12, 24])
def test_get_optim_files(tmpdir, num_checkpoints):
    saved_files = []
    for i in range(num_checkpoints):
        file_name = "zero_" + str(i) + "_optim_states.pt"
        path_name = os.path.join(tmpdir, file_name)
        saved_files.append(path_name)
        with open(path_name, "w") as f:
            f.write(file_name)
    loaded_files = get_optim_files(tmpdir)
    for lf, sf in zip(loaded_files, saved_files):
        assert lf == sf
