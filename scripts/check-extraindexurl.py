#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from __future__ import annotations
'''Copyright The Microsoft DeepSpeed Team'''
"""
Checks each file in sys.argv for the string "--extra-index-url".
Modified from https://github.com/jlebar/pre-commit-hooks/blob/master/check_do_not_submit.py
"""

import subprocess
import sys


def err(s: str) -> None:
    print(s, file=sys.stderr)


print(*sys.argv[1:])

# There are many ways we could search for the string "--extra-index-url", but `git
# grep --no-index` is nice because
#  - it's very fast (as compared to iterating over the file in Python)
#  - we can reasonably assume it's available on all machines
#  - unlike plain grep, which is slower and has different flags on MacOS versus
#    Linux, git grep is always the same.
res = subprocess.run(
    ["git", "grep", "-Hn", "--no-index", "-e", r"--extra-index-url", *sys.argv[1:]],
    capture_output=True,
)
if res.returncode == 0:
    err('Error: The string "--extra-index-url" was found.\nPlease replace all calls to --extra-index-url with "--index-url"'
        )
    err(res.stdout.decode("utf-8"))
    sys.exit(1)
elif res.returncode == 2:
    err(f"Error invoking grep on {', '.join(sys.argv[1:])}:")
    err(res.stderr.decode("utf-8"))
    sys.exit(2)
