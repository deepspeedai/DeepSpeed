# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""State used to record and replay AutoEP routing decisions."""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum

import torch


class RoutingReplayMode(str, Enum):
    """Modes understood by :class:`RoutingReplay`."""

    OFF = "off"
    RECORD = "record"
    REPLAY = "replay"


class RoutingReplay:
    """Per-model routing state for checkpoint and rollout replay.

    A replay object is deliberately attached to one model.  Keeping the
    queues out of module globals prevents an actor and a reference model in
    the same process from consuming each other's routes.
    """

    def __init__(self):
        self.mode = RoutingReplayMode.OFF
        self._recorded = defaultdict(deque)
        self._replay = {}
        self._replay_offsets = defaultdict(int)

    def clear(self) -> None:
        """Drop recorded and imported routes and reset read cursors."""
        self._recorded.clear()
        self._replay.clear()
        self._replay_offsets.clear()

    def set_mode(self, mode: RoutingReplayMode | str) -> None:
        """Set the active mode without changing stored route data."""
        self.mode = RoutingReplayMode(mode)

    @property
    def is_replaying(self) -> bool:
        """Whether routers should consume stored route IDs."""
        return self.mode == RoutingReplayMode.REPLAY

    @property
    def is_recording(self) -> bool:
        """Whether routers should append selected route IDs."""
        return self.mode == RoutingReplayMode.RECORD

    def set_replay_data(self, layer_name: str, selected_experts: torch.Tensor) -> None:
        """Register external routes for one layer.

        ``selected_experts`` must be an integer tensor shaped ``[tokens, top_k]``.
        The tensor is detached and copied so callers can reuse their rollout
        buffer immediately after this call.
        """
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError("layer_name must be a non-empty string")
        if not torch.is_tensor(selected_experts) or selected_experts.ndim != 2:
            raise ValueError("selected_experts must be a rank-2 tensor shaped [tokens, top_k]")
        if selected_experts.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("selected_experts must use an integer dtype")
        self._replay[layer_name] = selected_experts.detach().to(device="cpu").contiguous().clone()
        self._replay_offsets[layer_name] = 0

    def _record(self, layer_name: str, selected_experts: torch.Tensor) -> None:
        self._recorded[layer_name].append(selected_experts.detach().to(device="cpu").contiguous().clone())

    def next_routes(
        self,
        layer_name: str,
        tokens: int,
        top_k: int,
        device: torch.device,
        num_experts: int | None = None,
    ) -> torch.Tensor:
        """Return the next stored routes for a router forward."""
        if layer_name not in self._replay:
            queued = self._recorded.get(layer_name)
            if not queued:
                raise RuntimeError(f"No recorded routes available for AutoEP layer '{layer_name}'")
            routes = queued[0]
        else:
            routes = self._replay[layer_name]
            offset = self._replay_offsets[layer_name]
            routes = routes[offset:offset + tokens]

        if routes.shape != (tokens, top_k):
            raise ValueError(f"Routes for AutoEP layer '{layer_name}' have shape {tuple(routes.shape)}, "
                             f"expected {(tokens, top_k)}")
        if num_experts is not None and routes.numel() > 0:
            if routes.min().item() < 0 or routes.max().item() >= num_experts:
                raise ValueError(f"Replayed routes for '{layer_name}' contain an expert outside "
                                 f"[0, {num_experts})")
            if top_k > 1:
                sorted_routes = routes.sort(dim=-1).values
                if torch.any(sorted_routes[:, 1:] == sorted_routes[:, :-1]):
                    raise ValueError(f"Replayed routes for '{layer_name}' contain duplicate experts")

        if layer_name in self._replay:
            self._replay_offsets[layer_name] += tokens
        else:
            queued.popleft()
        return routes.to(device=device, dtype=torch.long)

    def apply(self, layer_name: str, selected_experts: torch.Tensor) -> torch.Tensor:
        """Record or replace a router's selected expert IDs."""
        if self.mode == RoutingReplayMode.RECORD:
            self._record(layer_name, selected_experts)
            return selected_experts
        if self.mode == RoutingReplayMode.REPLAY:
            return self.next_routes(layer_name, selected_experts.shape[0], selected_experts.shape[1],
                                    selected_experts.device)
        return selected_experts

    @contextmanager
    def recording(self, *, clear: bool = True):
        """Record one or more model forwards."""
        previous = self.mode
        if clear:
            self._recorded.clear()
            self._replay.clear()
            self._replay_offsets.clear()
        self.mode = RoutingReplayMode.RECORD
        try:
            yield self
        finally:
            self.mode = previous

    @contextmanager
    def replaying(self):
        """Replay recorded or imported routes for model forwards."""
        previous = self.mode
        self.mode = RoutingReplayMode.REPLAY
        try:
            yield self
        finally:
            self.mode = previous


def attach_routing_replay(model, replay: RoutingReplay, *, prefix: str = "") -> int:
    """Attach ``replay`` to every AutoEP router below ``model``.

    Returns the number of routers attached.  This helper keeps the public
    setup independent of the private replacement traversal used by AutoEP.
    """
    count = 0
    for name, module in model.named_modules():
        router = getattr(module, "router", None)
        if router is None or not hasattr(router, "set_routing_replay"):
            continue
        layer_name = f"{prefix}.{name}".strip(".") or "root"
        router.set_routing_replay(replay, layer_name)
        count += 1
    return count
