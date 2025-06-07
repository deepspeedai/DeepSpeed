# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .constants import *

BASE_STAT_KEYS = [
    CLOSE_COUNT_KEY, FILENO_COUNT_KEY, FLUSH_COUNT_KEY, WRITE_COUNT_KEY, WRITE_BYTES_KEY, WRITE_SEC_KEY,
    WRITE_SPEED_KEY
]


class BaseFileWriter(object):

    def __init__(self, file_path):
        self._file_path = file_path
        self._stats = {k: 0 for k in BASE_STAT_KEYS}

    def close(self):
        pass

    def fileno(self):
        pass

    def flush(self):
        pass

    def write(self, buffer):
        pass

    def file_path(self):
        return self._file_path

    def _incr_stats(self, key, incr=1):
        self._stats[key] += incr

    def _dump_state(self):
        if self._stats[WRITE_SEC_KEY] > 0:
            self._stats[WRITE_SPEED_KEY] = (self._stats[WRITE_BYTES_KEY] / self._stats[WRITE_SEC_KEY] / (1024**3))
        state = self._stats
        state[FILE_PATH_KEY] = self.file_path()
        print(f'stats = {self._stats}')
