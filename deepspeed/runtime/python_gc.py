# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Process-wide Python cyclic garbage collection control."""

import gc
import os
from threading import RLock

from deepspeed.utils import logger


class PythonGCManager:
    """Coordinate process-wide GC state across multiple DeepSpeed engines."""

    def __init__(self):
        self._lock = RLock()
        self._active_engines = 0
        self._restore_enabled = False
        self._generation = 0
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(before=self._before_fork,
                                after_in_parent=self._after_fork_parent,
                                after_in_child=self._after_fork_child)

    def _before_fork(self):
        self._lock.acquire()

    def _after_fork_parent(self):
        self._lock.release()

    def _after_fork_child(self):
        if self._active_engines > 0 and self._restore_enabled and not gc.isenabled():
            gc.enable()
        self._active_engines = 0
        self._restore_enabled = False
        self._generation += 1
        self._lock = RLock()

    def acquire(self):
        with self._lock:
            if self._active_engines == 0:
                self._restore_enabled = gc.isenabled()
                if self._restore_enabled:
                    collected = gc.collect()
                    gc.disable()
                    logger.info("Disabled automatic Python cyclic GC after collecting %d objects", collected)
            self._active_engines += 1
            return self._generation

    def release(self, generation):
        with self._lock:
            if generation != self._generation:
                return
            if self._active_engines == 0:
                return
            self._active_engines -= 1
            restore_enabled = self._active_engines == 0 and self._restore_enabled
            if self._active_engines == 0:
                self._restore_enabled = False
            if restore_enabled and not gc.isenabled():
                gc.enable()
                logger.info("Restored automatic Python cyclic GC")

    def collect(self):
        """Run an explicit collection without changing the automatic-GC policy."""
        return gc.collect()

    @property
    def active_engines(self):
        with self._lock:
            return self._active_engines


python_gc_manager = PythonGCManager()
