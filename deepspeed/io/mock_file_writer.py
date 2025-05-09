# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .constants import *
from .base_file_writer import BaseFileWriter
from .utils import obj_serialization_details


class MockFileWriter(BaseFileWriter):

    def __init__(self, file_path):
        super(MockFileWriter, self).__init__(file_path)
        self._fp = open(file_path, 'wb')
        self._stats[SAVE_STORAGE_KEY] = 0
        self._stats[SAVE_STORAGE_BYTES_KEY] = 0
        self._get_serialization_details = obj_serialization_details()

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
        return self._write(len(buffer))

    def save_torch_storage_object_list(self, storage_obj_list, save_size):
        num_bytes = sum([self._save_torch_storage_object(obj, save_size) for obj in storage_obj_list])
        return num_bytes

    def _save_torch_storage_object(self, storage_obj, save_size):
        details = self._get_serialization_details(storage_obj)
        self._incr_stats(SAVE_STORAGE_KEY)
        self._incr_stats(SAVE_STORAGE_BYTES_KEY, details.size)
        num_written_bytes = self._write(STORAGE_OBJ_SIZE) if save_size else 0
        return num_written_bytes + self._write(details.size)

    def _write(self, num_bytes):
        self._incr_stats(WRITE_COUNT_KEY)
        self._incr_stats(WRITE_BYTES_KEY, num_bytes)
        return num_bytes
