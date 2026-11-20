# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import threading
from collections import deque

from deepspeed.accelerator import get_accelerator


class ReadyBatch:
    """Consecutive independent Ready jobs owned by one shared optimizer executor future."""

    def __init__(self, optimizer, item):
        self.optimizer = optimizer
        self.lock = threading.Lock()
        self.pending = deque()
        self.bounds = {}
        self.window = (item["grad_buffer_index"], item["cancel_version"], item["target_step"])
        self.closed = False
        self.active = None
        self.future = None
        self.append(item)

    def append(self, item):
        window = (item["grad_buffer_index"], item["cancel_version"], item["target_step"])
        group = item["sub_group_id"]
        ranges = item["ranges"]
        first = ranges[0][0]
        last_offset, last_size = ranges[-1]
        last = last_offset + last_size
        with self.lock:
            if self.closed or window != self.window:
                return False
            previous = self.bounds.get(group)
            if previous is not None:
                # Reentrant edits and clipping can regenerate a range. An intersecting
                # span stays in FIFO order behind this batch, even if it contains gaps.
                if first < previous[1] and last > previous[0]:
                    return False
                first, last = min(first, previous[0]), max(last, previous[1])
            self.bounds[group] = (first, last)
            self.pending.append(item)
            return True

    def discard(self):
        with self.lock:
            self.closed = True
            self.pending.clear()
            if self.active is not None:
                self.active.cancel()

    def _take(self, wait):
        with self.lock:
            if not self.pending:
                if wait:
                    self.closed = True
                return None
            item = self.pending[0]
            if not wait:
                event = item["last_copy_event"]
                commit = self.optimizer._pending_state_updates_by_subgroup.get(item["sub_group_id"])
                if commit is not None and not commit.done():
                    return None
                if event is not None:
                    if not event.query():
                        return None
                    # This job owns its event reference. A successful query already
                    # observed the D2H completion required by the CPU kernel.
                    item["last_copy_event"] = None
            self.pending.popleft()
        return self.optimizer._bucketwise_chunk_update_worker(item, defer=True)

    def _take_eligible(self):
        return self._take(wait=False)

    def run(self):
        stream = self.optimizer._bucketwise_h2d_stream
        started = False
        try:
            while True:
                work = self._take(wait=True)
                if work is None:
                    with self.lock:
                        if self.closed:
                            break
                    continue
                with self.lock:
                    if self.closed:
                        break
                    self.active = work
                started = True
                work.run(self._take_eligible)
            if started:
                # All helpers have returned from copy submission, including any prefix
                # of a cancelled job. One stream event protects all their pinned slots.
                event = get_accelerator().Event()
                event.record(stream)
                return event
        except BaseException:
            stream.synchronize()
            raise
        finally:
            with self.lock:
                self.closed = True
                self.active = None
                self.pending.clear()
