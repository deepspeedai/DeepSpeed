#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from __future__ import annotations
'''Copyright The Microsoft DeepSpeed Team'''
"""
Checks each file in sys.argv for the string "torch.distributed".
Modified from https://github.com/jlebar/pre-commit-hooks/blob/master/check_do_not_submit.py
"""

import subprocess
import sys


def err(s: str) -> None:
    print(s, file=sys.stderr)


# There are many ways we could search for the string "torch.distributed", but `git
# grep --no-index` is nice because
#  - it's very fast (as compared to iterating over the file in Python)
#  - we can reasonably assume it's available on all machines
#  - unlike plain grep, which is slower and has different flags on MacOS versus
#    Linux, git grep is always the same.
res = subprocess.run(
    ["git", "grep", "-Hn", "--no-index", r"torch\.distributed", *sys.argv[1:]],
    capture_output=True,
)
if res.returncode == 0:
    err('Error: The string "torch.distributed" was found. Please replace all calls to torch.distributed with "deepspeed.comm"'
        )
    err(res.stdout.decode("utf-8"))
    sys.exit(1)
elif res.returncode == 2:
    err(f"Error invoking grep on {', '.join(sys.argv[1:])}:")
    err(res.stderr.decode("utf-8"))
    sys.exit(2)
