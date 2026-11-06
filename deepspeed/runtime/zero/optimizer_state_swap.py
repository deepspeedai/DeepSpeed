# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""ZeRO-1/2's FP32 optimizer state in pinned host memory outside the optimizer step.

``zero_optimization.optimizer_state_swap``. DeepSpeed's ZeRO-1/2 optimizer keeps, on every GPU, its FP32 partition of
the master weights (``single_partition_of_fp32_groups[i]``) and the optimizer's state for it (Adam's ``exp_avg`` and
``exp_avg_sq``). They are read only by the step, yet they hold the GPU memory through the forward and backward, where a
long-sequence job needs it for activations. With the swap on, those tensors live on the GPU only from the end of the
backward to the end of the step:

- at the gradient accumulation boundary, the end-of-backward reduction epilogue
  (``overlapping_partition_gradients_reduce_epilogue``) starts copying them back on a side stream, so the copy overlaps
  the round trip until the step;
- ``step`` waits for those copies, runs unchanged, then starts the copies out and frees the GPU memory.

``fraction`` ``f`` in (0, 1] moves only the first tensors (masters and their state, in the optimizer's order) that hold
a fraction ``f`` of those bytes, whole tensors at a time, and keeps the rest on the GPU: less to fetch before each step,
for a peak that has room. The masters exist from the optimizer's construction, the moments from its first step: the
swap evicts the masters when the optimizer is built (so the first forward runs without them too), and chooses the set
again at the eviction where the moments first appear.


The tensors are freed by resizing their storage to 0 and restored by resizing it back, not by replacing them: DeepSpeed
keeps views into the masters (the ``_hp_mapping`` fragments), which must stay valid. The caching allocator holds a freed
block until the copy stream recorded on it has finished, so the next forward can reuse the memory safely. The swap
replaces ``zero_optimization.offload_optimizer``, which moves the step itself to the CPU; the two are refused together.
Checkpoints and the other paths that read or replace the state outside the step work on the host copies instead of
fetching them: ``DeepSpeedEngine.save_checkpoint`` and ``load_checkpoint``, and the optimizer's own ``load_state_dict``,
``refresh_fp32_params``, ``update_lp_params`` and ``_restore_from_bit16_weights``, run with each evicted master's
``.data`` and each evicted state entry set to its pinned host copy (``begin_host_hold``), so a save writes the host
copies and a load writes into them, and the evicted state never comes back to the GPU, where it may not fit. A load
that replaces the optimizer's state tensors keeps the GPU tensors it replaces (the ``_hp_mapping`` views point into
them) and copies the new values into their host copies. ``optimizer_state_resident`` holds the whole state on the GPU
for code that needs it there (``safe_get_full_fp32_param`` and friends), and a universal checkpoint load uses it.

Muon's momentum stays on the GPU: ZeRO folds each gradient into it in the end-of-backward epilogue
(``get_flat_partition``) and commits it at the start of ``step``, before the swap has fetched anything; the masters of
the Muon groups and the auxiliary Adam's state still move.
"""

import contextlib
import functools
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from deepspeed.accelerator import get_accelerator


def _is_muon_group(optimizer: Any, group_no: int) -> bool:
    is_muon = getattr(optimizer, "_is_muon_group", None)
    groups = getattr(optimizer, "bit16_groups", None)
    return is_muon is not None and groups is not None and bool(groups[group_no]) and is_muon(groups[group_no])


def group_state_tensors(optimizer: Any, group_no: int) -> List[torch.Tensor]:
    """The tensors of param group ``group_no`` the swap moves: its master partition, then its optimizer state tensors,
    except Muon's momentum, which ZeRO reads outside the step."""
    master = optimizer.single_partition_of_fp32_groups[group_no]
    tensors = [master]
    for key, value in optimizer.optimizer.state.get(master, {}).items():
        if not (torch.is_tensor(value) and value.device == master.device and value.numel() > 1):
            continue
        if key == "momentum_buffer" and _is_muon_group(optimizer, group_no):
            continue
        tensors.append(value)
    return tensors


def state_tensors(optimizer: Any) -> List[torch.Tensor]:
    """Every tensor the swap can move, in order: each master partition, then its optimizer state tensors."""
    tensors = []
    for group_no in range(len(optimizer.single_partition_of_fp32_groups)):
        tensors.extend(group_state_tensors(optimizer, group_no))
    return tensors


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def select(tensors: Iterable[torch.Tensor], fraction: float) -> set:
    """Ids of the first whole tensors whose bytes reach ``fraction`` of the total (all of them at 1)."""
    tensors = list(tensors)
    budget = fraction * sum(_nbytes(t) for t in tensors)
    chosen, total = set(), 0
    for tensor in tensors:
        if total >= budget:
            break
        chosen.add(id(tensor))
        total += _nbytes(tensor)
    return chosen


class AcceleratorStreams:
    """The device calls the swap makes; tests substitute a recording fake to check the bookkeeping on the CPU."""

    pin_memory = True

    def new_stream(self, device):
        return get_accelerator().Stream(device=device)

    def current_stream(self):
        return get_accelerator().current_stream()

    def use(self, stream):
        return get_accelerator().stream(stream)

    def new_event(self):
        return get_accelerator().Event()

    def record_stream(self, tensor, stream) -> None:
        tensor.record_stream(stream)

    def synchronize_stream(self, stream) -> None:
        stream.synchronize()


class _HostCopies:
    """What both swaps share: the pinned host copies, and the hold that serves the evicted state from them."""

    def _host_copy(self, tensor: torch.Tensor) -> torch.Tensor:
        host = self.host.get(id(tensor))
        if host is None or host.numel() != tensor.numel() or host.dtype != tensor.dtype:
            host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=self._streams.pin_memory)
            self.host[id(tensor)] = host
        return host

    def _evict_now(self, tensor: torch.Tensor) -> None:
        """Evict one tensor outside the step, synchronously."""
        self._host_copy(tensor).copy_(tensor.detach())
        tensor.untyped_storage().resize_(0)

    def _fetch_now(self, tensor: torch.Tensor) -> None:
        tensor.untyped_storage().resize_(_nbytes(tensor))
        tensor.detach().copy_(self.host[id(tensor)])

    def _prune_host_copies(self, live: Iterable[torch.Tensor]) -> None:
        live_ids = {id(t) for t in live}
        for key in [key for key in self.host if key not in live_ids]:
            del self.host[key]

    def begin_host_hold(self) -> list:
        """Until ``end_host_hold``, each evicted master's ``.data`` is its pinned host copy and each evicted state entry
        of the optimizer is its host copy, so code that saves or loads the state works on the host without the state
        coming back to the GPU. Returns what ``end_host_hold`` restores, and the state entries before the hold."""
        self.sync_copies_out()
        evicted = self.evicted_tensors()
        stash, before = [], {}
        state = self.optimizer.optimizer.state
        for group_no, master in enumerate(self.optimizer.single_partition_of_fp32_groups):
            entries = state.get(master, {})
            for key, value in list(entries.items()):
                before[(group_no, key)] = value
                if torch.is_tensor(value) and id(value) in evicted:
                    stash.append((group_no, key, value, self.host[id(value)]))
                    entries[key] = self.host[id(value)]
            if id(master) in evicted:
                stash.append((group_no, None, master.data, self.host[id(master)]))
                master.data = self.host[id(master)]
        return stash, before

    def end_host_hold(self, held, reselect: bool = False) -> None:
        """Put the GPU tensors back. A load may have written into the host copies, or replaced the state tensors: the
        new values then go into the host copy of the tensor they replace, which stays (the ``_hp_mapping`` views point
        into it), and a tensor without one gets an evicted GPU tensor of its own; nothing comes back to the GPU."""
        stash, before = held
        state = self.optimizer.optimizer.state
        masters = self.optimizer.single_partition_of_fp32_groups
        replaced, evicted_masters = {}, set()
        for group_no, key, gpu, host in stash:
            if key is not None:
                replaced[(group_no, key)] = (gpu, host)
                continue
            master = masters[group_no]
            if master.data.data_ptr() != host.data_ptr():
                host.copy_(master.data)
            master.data = gpu
            evicted_masters.add(group_no)
        unused = []
        for group_no, master in enumerate(masters):
            entries = state.get(master)
            if not entries:
                continue
            for key, value in list(entries.items()):
                old = replaced.pop((group_no, key), None)
                if old is not None:
                    gpu, host = old
                    if value is host:
                        entries[key] = gpu
                        continue
                    if torch.is_tensor(value) and value.shape == gpu.shape and value.dtype == gpu.dtype:
                        host.copy_(value)
                        entries[key] = gpu
                        continue
                    unused.append(gpu)
                elif value is before.get((group_no, key)):
                    continue  # untouched, and on the GPU
                if not (group_no in evicted_masters and torch.is_tensor(value) and value.is_floating_point()
                        and value.numel() > 1):
                    continue
                if key == "momentum_buffer" and _is_muon_group(self.optimizer, group_no):
                    entries[key] = value.to(master.device)  # Muon's momentum stays on the GPU
                else:
                    gpu = torch.empty(value.shape, dtype=value.dtype, device=master.device)
                    gpu.untyped_storage().resize_(0)
                    self._host_copy(gpu).copy_(value)
                    entries[key] = gpu
        unused.extend(gpu for gpu, _ in replaced.values())
        for gpu in unused:
            self.host.pop(id(gpu), None)
        if reselect:
            self.after_host_load()


def _at_accumulation_boundary(optimizer: Any) -> bool:
    boundary = getattr(optimizer, "is_gradient_accumulation_boundary", True)
    return bool(boundary() if callable(boundary) else boundary)


class OptimizerStateSwap(_HostCopies):
    """The whole-state swap of one ZeRO optimizer: which tensors move, their pinned host copies, and whether they are
    on the GPU now. The optimizer calls the ``after_*`` / ``before_*`` hooks from its own methods."""

    def __init__(self,
                 fraction: float,
                 tensors: Callable[[], List[torch.Tensor]],
                 streams: Optional[Any] = None,
                 optimizer: Any = None) -> None:
        if not 0 < fraction <= 1:
            raise ValueError(f"optimizer_state_swap.fraction must be in (0, 1], got {fraction!r}")
        self.fraction = fraction
        self.optimizer = optimizer  # the host holds (_HostCopies) relink its state
        self._all_tensors = tensors
        self._streams = streams or AcceleratorStreams()
        self.swapped: Optional[set] = None  # ids of the tensors this swap moves, chosen at an evict
        self._known = 0  # how many state tensors that choice saw: the moments appear at the first step
        self.host: Dict[int, torch.Tensor] = {}  # id(tensor) -> pinned host copy, allocated once and reused
        self.resident = True
        self.bytes = 0  # bytes moved by the last eviction
        self._stream = None
        self._fetched = None  # event recorded after the last fetch's copies
        self._hold_evicted = False

    def tensors(self) -> List[torch.Tensor]:
        """The tensors this swap moves; numel and device survive the storage resize, so the set is stable."""
        tensors = self._all_tensors()
        if self.swapped is None or (self.resident and len(tensors) != self._known):
            self.swapped = select(tensors, self.fraction)
            self._known = len(tensors)
        return [t for t in tensors if id(t) in self.swapped]

    def _side_stream(self, device):
        if self._stream is None:
            self._stream = self._streams.new_stream(device)
        return self._stream

    def evict(self) -> None:
        """Copy the tensors to their host buffers on the side stream, then free their GPU storage."""
        if not self.resident:
            return
        tensors = self.tensors()
        if not tensors:
            return
        stream = self._side_stream(tensors[0].device)
        stream.wait_stream(self._streams.current_stream())
        self.bytes = 0
        with self._streams.use(stream):
            for tensor in tensors:
                host = self.host.get(id(tensor))
                if host is None or host.numel() != tensor.numel():
                    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=self._streams.pin_memory)
                    self.host[id(tensor)] = host
                host.copy_(tensor.detach(), non_blocking=True)
                self._streams.record_stream(tensor, stream)
                self.bytes += _nbytes(tensor)
        for tensor in tensors:
            tensor.untyped_storage().resize_(0)
        self.resident = False

    def fetch(self) -> None:
        """Reallocate the tensors' storage (on the compute stream) and start the copies back on the side stream."""
        if self.resident:
            return
        tensors = self.tensors()
        stream = self._side_stream(tensors[0].device)
        for tensor in tensors:
            tensor.untyped_storage().resize_(_nbytes(tensor))
        stream.wait_stream(self._streams.current_stream())
        with self._streams.use(stream):
            for tensor in tensors:
                tensor.detach().copy_(self.host[id(tensor)], non_blocking=True)
                self._streams.record_stream(tensor, stream)
        self._fetched = self._streams.new_event()
        self._fetched.record(stream)
        self.resident = True

    def wait(self) -> None:
        """Make the compute stream wait for the last fetch's copies."""
        if self._fetched is not None:
            self._streams.current_stream().wait_event(self._fetched)
            self._fetched = None

    # Hooks, in the order the optimizer calls them.

    def install(self) -> None:
        """The masters exist from the optimizer's construction: off the GPU before the first forward."""
        self.evict()

    def after_backward_epilogue(self, optimizer: Any) -> None:
        if _at_accumulation_boundary(optimizer):  # the last backward before the step
            self.fetch()

    def before_step(self) -> None:
        self.fetch()  # a no-op when the epilogue already started it
        self.wait()

    def after_step(self) -> None:
        self.evict()

    def all_resident(self) -> bool:
        return self.resident

    def begin_hold(self) -> None:
        """The tensors on the GPU until ``end_hold``, evicted again then if they were evicted before."""
        self._hold_evicted = not self.resident
        self.fetch()
        self.wait()

    def end_hold(self, reselect: bool = False) -> None:
        """``reselect``: the hold may have replaced the optimizer's state tensors (a checkpoint load), so the moved set
        and its host buffers are chosen again before the next eviction."""
        if reselect:
            self.swapped = None
            self.host.clear()
        if self._hold_evicted:
            self.evict()

    @contextlib.contextmanager
    def held(self, *, reselect: bool = False):
        self.begin_hold()
        try:
            yield
        finally:
            self.end_hold(reselect)

    def evicted_tensors(self) -> set:
        if self.resident:
            return set()
        return {id(t) for t in self.tensors() if t.untyped_storage().nbytes() == 0}

    def sync_copies_out(self) -> None:
        if self._stream is not None:
            self._streams.synchronize_stream(self._stream)

    def after_host_load(self) -> None:
        """After a load, choose the moved set again (the moments may be new) and evict or fetch to match it."""
        tensors = self._all_tensors()
        chosen = select(tensors, self.fraction)
        for tensor in tensors:
            evicted = tensor.untyped_storage().nbytes() == 0
            if id(tensor) in chosen and not evicted:
                self._evict_now(tensor)
            elif id(tensor) not in chosen and evicted:
                self._fetch_now(tensor)
        self.swapped, self._known = chosen, len(tensors)
        self.resident = not chosen
        self._fetched = None
        self._prune_host_copies(tensors)


def holds_state(reselect: bool = False, of: Callable[[Any], Any] = lambda owner: owner, on_device=False):
    """Decorator for a method that reads or replaces the FP32 state outside the step: the swap of ``of(self)`` (the ZeRO
    optimizer itself, or an engine's ``optimizer``) serves the evicted state from its host copies for the call
    (``on_device``, a bool or a function of ``self``: holds it on the GPU instead). A no-op for an optimizer without the
    swap, or another optimizer type."""

    def decorator(method):

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            device = on_device(self) if callable(on_device) else on_device
            with held(of(self), reselect=reselect, on_device=device):
                return method(self, *args, **kwargs)

        return wrapper

    return decorator


def held(optimizer: Any, reselect: bool = False, on_device: bool = False):
    """``holds_state``'s context for one call: the swap of ``optimizer`` serves the evicted state from its host copies
    (or holds it on the GPU, ``on_device``); a null context for an optimizer without the swap. For methods that must
    stay undecorated (the engine's checkpoint entry points, whose source callers verify)."""
    hold = getattr(optimizer, "_optimizer_state_hold", None)
    return contextlib.nullcontext() if hold is None else hold(on_device=on_device, reselect=reselect)


def build(optimizer: Any, config: Any, streams: Optional[Any] = None):
    """The swap ``config`` (``DeepSpeedZeroOptimizerStateSwapConfig``) asks for on this ZeRO-1/2 optimizer. Nothing is
    moved until ``install``."""
    return OptimizerStateSwap(float(config.fraction), lambda: state_tensors(optimizer), streams, optimizer=optimizer)


def describe(config: Any) -> str:
    return "fetch at the end of the backward"
