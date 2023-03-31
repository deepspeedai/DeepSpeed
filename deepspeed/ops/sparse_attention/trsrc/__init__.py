# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys
import os


def _build_file_index(directory, suffix='.tr'):
    """Build an index of source files and their basenames in a given directory.

    Args:
        directory (string): the directory to index
        suffix (string): index files with this suffix

    Returns:
        list: A list of tuples of the form [(basename, absolute path), ...]
    """

    index = []

    for fname in os.listdir(directory):
        if fname.endswith(suffix):
            basename = fname[:fname.rfind(suffix)]  # strip the suffix
            path = os.path.join(directory, fname)
            index.append((basename, path))

    return index


# Go over all local source files and parse them as strings
_module = sys.modules[_build_file_index.__module__]
_directory = os.path.dirname(os.path.realpath(__file__))
for name, fname in _build_file_index(_directory):
    with open(fname, 'r') as fin:
        setattr(_module, name, fin.read())
