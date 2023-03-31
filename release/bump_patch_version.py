# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from packaging import version as pkg_version

with open('../version.txt') as fd:
    version = pkg_version.parse(fd.read())

with open('../version.txt', 'w') as fd:
    fd.write(f'{version.major}.{version.minor}.{version.micro + 1}\n')

print(f'{version} -> {version.major}.{version.minor}.{version.micro + 1}')
