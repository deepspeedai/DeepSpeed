# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from deepspeed.utils.logging import logger

_GB = 1024**3
# Emit an INFO checkpoint each time cumulative pinned memory reaches the next
# power-of-two multiple of this base. Doubling keeps the number of checkpoints
# logarithmic in the total, so large offload paths (e.g. per-parameter shards
# pinned at init) produce a few readable milestones instead of flooding the log.
_CHECKPOINT_BASE_GB = 32


def _fmt_bytes(num_bytes: int) -> str:
    kb = 1024
    mb = kb * 1024
    gb = mb * 1024
    if num_bytes >= gb:
        return f"{num_bytes / gb:.3f} GB"
    if num_bytes >= mb:
        return f"{num_bytes / mb:.2f} MB"
    if num_bytes >= kb:
        return f"{num_bytes / kb:.1f} KB"
    return f"{num_bytes} B"


class _PinnedMemoryTracker:
    """Process-wide total of host memory pinned through the accelerator's
    ``pin_memory``. Pinned memory is page-locked: it cannot be swapped out and
    counts against the host memlock limit (``ulimit -l``). The running total is
    a useful hint when diagnosing host out-of-memory errors, which often surface
    far from the call site that consumed the resident-RAM budget.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._bytes = 0
        self._calls = 0
        self._next_checkpoint = _CHECKPOINT_BASE_GB * _GB

    def track(self, num_bytes: int) -> None:
        self._bytes += num_bytes
        self._calls += 1
        logger.debug(f"pin_memory: +{_fmt_bytes(num_bytes)} "
                     f"(call #{self._calls}, running total: {_fmt_bytes(self._bytes)})")
        while self._bytes >= self._next_checkpoint:
            msg = (f"[pinned-memory checkpoint] crossed {_fmt_bytes(self._next_checkpoint)}: "
                   f"{_fmt_bytes(self._bytes)} pinned across {self._calls} allocations")
            logger.info(msg)
            self._next_checkpoint *= 2

    def log_summary(self, tag: str = "") -> None:
        prefix = f"[pinned-memory {tag}] " if tag else "[pinned-memory] "
        logger.info(f"{prefix}{_fmt_bytes(self._bytes)} pinned across {self._calls} allocations")


_tracker = _PinnedMemoryTracker()


def track_pinned_memory(num_bytes: int) -> None:
    _tracker.track(num_bytes)


def pinned_memory_summary(tag: str = "") -> None:
    _tracker.log_summary(tag)
