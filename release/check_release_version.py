# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
from packaging import version as pkg_version

parser = argparse.ArgumentParser()

parser.add_argument("--release_version", type=str, help="The new version being published.")

args = parser.parse_args()

release_version = pkg_version.parse(args.release_version)

with open('./version.txt') as fd:
    repo_version = pkg_version.parse(fd.read())

assert repo_version == release_version, f"{repo_version=} does not match {release_version=}, unable to proceed"
