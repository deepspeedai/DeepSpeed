#!/usr/bin/env python3
from __future__ import annotations
'''Copyright The Microsoft DeepSpeed Team'''
"""
Modified from https://github.com/jlebar/pre-commit-hooks/blob/master/check_do_not_submit.py
"""

import subprocess
import sys


def err(s: str) -> None:
    print(s, file=sys.stderr)


success = True
failures = []
for f in sys.argv[1:]:
    res = subprocess.run(
        ["git",
         "grep",
         "--quiet",
         "-e",
         r"Copyright .* DeepSpeed Team",
         f],
        capture_output=True)
    if res.returncode == 1:
        success = False
        failures.append(f)
    elif res.returncode == 2:
        err(f"Error invoking grep on {', '.join(sys.argv[1:])}:")
        err(res.stderr.decode("utf-8"))
        sys.exit(2)

if not success:
    err(f'{failures}: Missing license at top of file')
    err(res.stdout.decode("utf-8"))
    sys.exit(1)
