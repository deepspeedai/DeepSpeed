# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
from packaging import version as pkg_version

parser = argparse.ArgumentParser()

parser.add_argument("--current_version",
                    type=str,
                    help="The current version being published to help set the next version.")

args = parser.parse_args()

current_version = pkg_version.parse(args.current_version)

with open('./version.txt', 'w') as fd:
    fd.write(f'{current_version.major}.{current_version.minor}.{current_version.micro + 1}\n')

print(f'{current_version} -> {current_version.major}.{current_version.minor}.{current_version.micro + 1}')
