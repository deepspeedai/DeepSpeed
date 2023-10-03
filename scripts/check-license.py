#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from __future__ import annotations
'''Copyright The Microsoft DeepSpeed Team'''
"""
Modified from https://github.com/jlebar/pre-commit-hooks/blob/master/check_do_not_submit.py
"""

import subprocess
import sys


def err(s: str) -> None:
    print(s, file=sys.stderr)


COPYRIGHT = [
    (r"^# Copyright (c) Microsoft Corporation.$", r"^\/\/ Copyright (c) Microsoft Corporation.$"),
    (r"^# SPDX-License-Identifier: Apache-2.0$", r"^\/\/ SPDX-License-Identifier: Apache-2.0$"),
    (r"^# DeepSpeed Team$", r"^\/\/ DeepSpeed Team$"),
]

success = True
failures = []
for f in sys.argv[1:]:
    for copyright_line in COPYRIGHT:
        cmd = ["git", "grep", "--quiet"]
        for line in copyright_line:
            cmd.extend(["-e", line])
        cmd.append(f)
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode == 1:
            success = False
            failures.append(f)
            break
        elif res.returncode == 2:
            err(f"Error invoking grep on {', '.join(sys.argv[1:])}:")
            err(res.stderr.decode("utf-8"))
            sys.exit(2)

if not success:
    err(f'{failures}: Missing license at top of file')
    err(res.stdout.decode("utf-8"))
    sys.exit(1)
