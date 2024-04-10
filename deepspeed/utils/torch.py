# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from packaging import version as pkg_version

import torch


def required_torch_version(min_version=None, max_version=None):
    assert min_version or max_version, "Must provide a min_version or max_version argument"

    torch_version = pkg_version.parse(torch.__version__)

    if min_version and pkg_version.parse(str(min_version)) > torch_version:
        return False

    if max_version and pkg_version.parse(str(max_version)) < torch_version:
        return False

    return True
