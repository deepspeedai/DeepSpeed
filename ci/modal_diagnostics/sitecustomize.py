# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import faulthandler
import json
import os
import signal
import sys
from pathlib import Path

_STACK_FILE = None

root = os.environ.get("DS_DIAGNOSTICS_DIR")
if root and hasattr(signal, "SIGUSR1"):
    try:
        stack_dir = Path(root) / "stacks"
        stack_dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        _STACK_FILE = (stack_dir / f"stack-{pid}.log").open("a", encoding="utf-8")
        faulthandler.enable(file=_STACK_FILE, all_threads=True)
        faulthandler.register(signal.SIGUSR1, file=_STACK_FILE, all_threads=True, chain=False)
        marker = {"argv0": Path(sys.argv[0]).name, "pid": pid}
        (stack_dir / f"registered-{pid}.json").write_text(
            json.dumps(marker, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError):
        pass
