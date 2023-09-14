# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
from packaging import version as pkg_version

parser = argparse.ArgumentParser()

parser.add_argument("--new_version", type=str, help="The new version being published.")

args = parser.parse_args()

new_version = pkg_version.parse(args.new_version)

with open('./version.txt') as fd:
    current_version = pkg_version.parse(fd.read())

# Valid version are those where the major/minor/micro are incremented by no more than one from the existing release, and the less significant values are reset to 0.
valid_major_update = pkg_version.Version(f'{current_version.major + 1}.0.0')
valid_minor_update = pkg_version.Version(f'{current_version.major}.{current_version.minor + 1}.0')
valid_micro_update = pkg_version.Version(
    f'{current_version.major}.{current_version.minor}.{current_version.micro + 1}')

valid_versions = [valid_major_update, valid_minor_update, valid_micro_update]

if new_version not in valid_versions:
    raise Exception(f'{new_version} is an invalid version. Valid versions are {valid_versions}.\n')
