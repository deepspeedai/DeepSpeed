# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
from .constants import *
from .base_file_writer import BaseFileWriter


class PyFileWriter(BaseFileWriter):

    def __init__(self, file_path):
        super(PyFileWriter, self).__init__(file_path)
        self._fp = open(file_path, 'wb')

    def close(self):
        self._incr_stats(CLOSE_COUNT_KEY)
        self._fp.close()

    def fileno(self):
        self._incr_stats(FILENO_COUNT_KEY)
        return INVALID_FD  # self._fp.fileno()

    def flush(self):
        self._incr_stats(FLUSH_COUNT_KEY)
        self._fp.flush()

    def write(self, buffer):
        st = time.time()
        self._fp.write(buffer)
        self._incr_stats(WRITE_SEC_KEY, time.time() - st)
        self._incr_stats(WRITE_COUNT_KEY)
        self._incr_stats(WRITE_BYTES_KEY, len(buffer))
        return len(buffer)
