# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""CUDA graph capture for HybridEngine token generation.

Generation in the HybridEngine is bound by CPU work rather than by the GPU: a
single decode step issues on the order of a thousand kernel launches, and the
kernels finish well before the host can queue the next step. Capturing the decode
forward into a CUDA graph replaces all of those launches with one replay.

Two properties of the inference kernels shape this design.

First, a single graph cannot serve every decode position. The kernels read the
current sequence length from a host-side counter
(``InferenceContext::current_tokens()``) and pass it to kernels as a launch
parameter. Capture freezes launch parameters, so a graph captured at one position
would keep reading and writing that same position. Host code *does* run during
capture, so capturing one graph per position records the correct offsets into
each.

Second, that counter is advanced from host code and is not reachable from Python.
Replay runs no host code, so it leaves the counter behind. An eager decode step
after a replay would therefore use a stale sequence length and silently corrupt
the KV cache. Eager and replayed decode steps must never be mixed within one
sequence, which is why the decision is made once per sequence, up front, in
``begin_sequence``.
"""

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger


def _tensor_kwargs(kwargs):
    """Names of the keyword arguments that hold a tensor."""
    return [name for name, value in kwargs.items() if torch.is_tensor(value)]


class DecodeGraphCache:
    """Dispatches generation forwards to a per-position CUDA graph.

    Args:
        forward: The original ``forward``, used both for capture and as the
            eager fallback.
        max_positions: Upper bound on how many decode positions may be captured,
            which bounds the memory the cache can consume.
    """

    def __init__(self, forward, max_positions):
        self._forward = forward
        self._max_positions = max_positions

        self._graphs = {}
        self._static_kwargs = {}
        self._static_outputs = {}
        self._pool = None

        self._position = 0
        self._captured_length = None
        self._use_graphs = False
        self._disabled = False

    @property
    def captured_positions(self):
        return len(self._graphs)

    def invalidate(self):
        """Drop every captured graph and the memory pool backing them."""
        self._graphs.clear()
        self._static_kwargs.clear()
        self._static_outputs.clear()
        self._pool = None
        self._captured_length = None

    def begin_sequence(self, num_decode_steps):
        """Decide once, before any decode step, whether this sequence uses graphs.

        Args:
            num_decode_steps: How many single-token forwards this generate call
                will make, or ``None`` when that is not known ahead of time.

        A sequence runs entirely on graphs or entirely eagerly. Deciding here
        rather than per step is what keeps the host-side sequence counter
        consistent: a capture pass advances it on every step, and a replay pass
        never reads it.
        """
        self._position = 0

        if self._disabled or num_decode_steps is None or num_decode_steps <= 0:
            self._use_graphs = False
            return

        if num_decode_steps > self._max_positions:
            self._use_graphs = False
            return

        # A different generation length needs its own set of graphs, because each
        # graph has both its sequence offset and its attention-mask width baked in.
        if self._captured_length is not None and self._captured_length != num_decode_steps:
            logger.info(f"HybridEngine CUDA graph: generation length changed "
                        f"{self._captured_length} -> {num_decode_steps}, recapturing.")
            self.invalidate()

        self._captured_length = num_decode_steps
        self._use_graphs = True

    def _capture(self, position, args, kwargs):
        """Record a graph for ``position`` and keep its static input buffers."""
        static = {name: kwargs[name].clone() for name in _tensor_kwargs(kwargs)}
        capture_kwargs = dict(kwargs)
        capture_kwargs.update(static)

        graph = get_accelerator().create_graph()
        if self._pool is None:
            with get_accelerator().capture_to_graph(graph):
                output = self._forward(*args, **capture_kwargs)
            self._pool = graph.pool()
        else:
            with get_accelerator().capture_to_graph(graph, pool=self._pool):
                output = self._forward(*args, **capture_kwargs)

        self._graphs[position] = graph
        self._static_kwargs[position] = static
        self._static_outputs[position] = output

    def _replay(self, position, kwargs):
        for name, buffer in self._static_kwargs[position].items():
            buffer.copy_(kwargs[name])
        get_accelerator().replay_graph(self._graphs[position])
        return self._static_outputs[position]

    def _shapes_match(self, position, kwargs):
        captured = self._static_kwargs[position]
        if set(captured) != set(_tensor_kwargs(kwargs)):
            return False
        return all(captured[name].shape == kwargs[name].shape for name in captured)

    def _fall_back(self, reason, args, kwargs):
        """Give up on graphs for good and run eagerly.

        Only safe before any replay has happened in the current sequence, which
        is why this is reachable only from the capture path.
        """
        logger.warning(f"HybridEngine CUDA graph disabled: {reason}")
        self.invalidate()
        self._disabled = True
        self._use_graphs = False
        return self._forward(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        is_decode_step = (input_ids is not None and input_ids.dim() == 2 and input_ids.shape[1] == 1)

        if not self._use_graphs or not is_decode_step:
            return self._forward(*args, **kwargs)

        position = self._position
        self._position += 1

        if position >= self._max_positions:
            # begin_sequence() bounds the length, so this means the caller ran
            # longer than it declared. Replays have already happened, so eager
            # execution would read a stale sequence counter.
            raise RuntimeError(f"HybridEngine CUDA graph: generation exceeded the declared length "
                               f"({self._max_positions} decode steps). Set hybrid_engine.max_out_tokens "
                               f"to cover the longest generation, or disable hybrid_engine.enable_cuda_graph.")

        if position in self._graphs:
            if self._shapes_match(position, kwargs):
                return self._replay(position, kwargs)
            return self._fall_back("input shapes changed mid-sequence", args, kwargs)

        try:
            self._capture(position, args, kwargs)
        except Exception as err:
            return self._fall_back(f"capture failed at position {position}: {err}", args, kwargs)

        # Capture records the work without running it, so the first execution must
        # come from a replay. That is also what fills the KV cache for this position.
        return self._replay(position, kwargs)


def decode_steps_from_generate_kwargs(kwargs):
    """How many single-token forwards a generate call will make, if knowable.

    HF emits one prompt forward that produces the first new token, then one
    forward per remaining token. Only a pinned length is usable here: with an
    open-ended limit the sequence may stop early, and a sequence that runs
    *longer* than its captured graphs cannot fall back safely.
    """
    max_new = kwargs.get("max_new_tokens")
    min_new = kwargs.get("min_new_tokens")
    if max_new is None or min_new is None or max_new != min_new:
        return None
    return int(max_new) - 1


def validate_cuda_graph_support(config, zero_stage):
    """Return the reason CUDA graphs cannot be used, or ``None`` if they can."""
    if not get_accelerator().is_available() or get_accelerator().device_name() != "cuda":
        return "CUDA graphs require the CUDA accelerator"

    if zero_stage == 3:
        # Under ZeRO-3 the inference containers hold no persistent weights; the
        # parameters are gathered into fresh buffers for each generate call. A
        # graph would keep replaying whichever buffers existed at capture time,
        # which is silently wrong rather than merely slow.
        return "CUDA graphs are not supported with ZeRO stage 3"

    if config.release_inference_cache:
        # Releasing the workspace frees the buffers the graphs write into.
        return "CUDA graphs are not supported with release_inference_cache"

    if config.inference_tp_size > 1:
        return "CUDA graphs are not supported with inference_tp_size > 1"

    return None
