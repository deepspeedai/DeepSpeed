# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import faulthandler
import json
import multiprocessing.util
import os
import signal
import sys
from pathlib import Path

_STACK_FILE = None


def _register_stack_handler():
    global _STACK_FILE
    root = os.environ.get("DS_DIAGNOSTICS_DIR")
    if not root or not hasattr(signal, "SIGUSR1"):
        return
    try:
        stack_dir = Path(root) / "stacks"
        stack_dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        stack_file = (stack_dir / f"stack-{pid}.log").open("a", encoding="utf-8")
        faulthandler.enable(file=stack_file, all_threads=True)
        faulthandler.unregister(signal.SIGUSR1)
        faulthandler.register(signal.SIGUSR1, file=stack_file, all_threads=True, chain=False)
        previous = _STACK_FILE
        _STACK_FILE = stack_file
        if previous is not None:
            previous.close()
        marker = {"argv0": Path(sys.argv[0]).name, "pid": pid}
        (stack_dir / f"registered-{pid}.json").write_text(
            json.dumps(marker, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError):
        pass


_register_stack_handler()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_register_stack_handler)
multiprocessing.util.register_after_fork(_register_stack_handler, lambda _: _register_stack_handler())
