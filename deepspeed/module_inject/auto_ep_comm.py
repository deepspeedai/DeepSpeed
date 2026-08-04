# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Selectable transports for the AutoEP expert all-to-all.

The dispatch and combine collectives are the largest single cost in an AutoEP
step, and a measured replay of real routing on 16 H100s across two nodes put
NCCL at 195.8 ms of payload all-to-all per step against DeepEP's 86.3 ms. This
module is what lets that be switched without the MoE layer knowing which
transport it is using.

Selection is by environment variable and defaults to the NCCL path, so a job
that sets nothing behaves exactly as before:

    DEEPSPEED_AUTOEP_COMM_BACKEND=nccl     (default)
    DEEPSPEED_AUTOEP_COMM_BACKEND=deepep

A backend is asked for once per layer and reused, because DeepEP's buffers are
sized at construction and are expensive to rebuild.
"""

from __future__ import annotations

import os

import torch

from deepspeed.utils import logger

_BACKEND_ENV = "DEEPSPEED_AUTOEP_COMM_BACKEND"
_NUM_SMS_ENV = "DEEPSPEED_AUTOEP_COMM_SMS"
NCCL_BACKEND = "nccl"
# Names the library, not its version: v2 is the only path implemented, and
# nothing about the name would have to change if that ever grew.
DEEPEP_BACKEND = "deepep"
AVAILABLE_BACKENDS = (NCCL_BACKEND, DEEPEP_BACKEND)
# GIN, and so DeepEP, does not exist in any form below this NCCL version.
NCCL_GIN_MIN_VERSION = (2, 30, 4)
# Chosen by sweeping whole training steps rather than the collective alone.
# On 16 H100s across two nodes the step took 340, 311, 353, 360 and 391 ms at 8,
# 12, 16, 24 and 32 SMs. Twelve is the point where the collective is already as
# fast as it gets while the expert GEMM still has the SMs it needs: at 8 the
# collective itself degrades, and above 12 the step grows because communication
# takes SMs the rest of the step was using.
DEFAULT_COMM_SMS = 12


def _qps_for_sms(num_sms: int) -> int:
    """Queue pairs to reserve for a given SM count.

    One per SM plus a small margin for the control path. This is deliberately
    smaller than DeepEP's automatic choice, which assumes it is the only thing
    on the fabric.
    """
    return num_sms + 4


def configured_backend() -> str:
    """The transport this process should use for expert all-to-all.

    An unset variable selects NCCL, so a job that opts into nothing keeps the
    behaviour it had before this existed.
    """
    raw = os.environ.get(_BACKEND_ENV)
    if raw is None or not raw.strip():
        return NCCL_BACKEND
    name = raw.strip().lower()
    if name not in AVAILABLE_BACKENDS:
        raise ValueError(f"{_BACKEND_ENV}={raw!r} is not one of {AVAILABLE_BACKENDS}")
    return name


def configured_num_sms() -> int:
    """SM budget for communication; 0 lets the backend decide.

    Worth setting because the collective competes with the expert GEMM for
    SMs: measurements showed the GEMM running 1.21-1.25x slower whenever a
    collective was in flight.
    """
    raw = os.environ.get(_NUM_SMS_ENV)
    if raw is None or not raw.strip():
        return 0
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{_NUM_SMS_ENV}={raw!r} is not an integer") from error


def _import_deep_ep():
    """Import DeepEP, explaining the environment it needs when it is absent.

    DeepEP is an optional dependency with prerequisites a cluster either meets
    or does not, and the failures it produces otherwise are opaque: a missing
    GIN-capable NCCL surfaces as an assertion inside buffer construction rather
    than as anything naming NCCL. Since this backend is only ever reached by
    explicit opt-in, the person who opted in is the one who can act on this.
    """
    try:
        import deep_ep
    except ImportError as error:
        raise ImportError(
            f"{_BACKEND_ENV}={DEEPEP_BACKEND} requires the deep_ep package, which is not installed. It also "
            "requires NCCL 2.30.4 or newer built with GIN support: the transport is unavailable below that "
            "version regardless of the network. Unset "
            f"{_BACKEND_ENV} to use the default NCCL all-to-all, which has no such requirement.") from error

    nccl_version = _nccl_version()
    if nccl_version is not None and nccl_version < NCCL_GIN_MIN_VERSION:
        installed = ".".join(str(part) for part in nccl_version)
        minimum = ".".join(str(part) for part in NCCL_GIN_MIN_VERSION)
        # Deliberately a warning. This reports the NCCL that torch bundles and
        # loads through its own RPATH, which DeepEP need not be using: DeepEP
        # links the NCCL it was built against, and that pairing has been
        # observed working while torch reported an older one. Refusing to start
        # on this signal would block a configuration already known to run.
        logger.warning(
            f"torch reports NCCL {installed}, older than the {minimum} that GIN requires. DeepEP links its own "
            "NCCL, so this is only a problem if it also resolves to the older one; a failure inside buffer "
            f"construction is the symptom. Unset {_BACKEND_ENV} to fall back to the default all-to-all.")
    return deep_ep


def _nccl_version() -> tuple[int, ...] | None:
    """The NCCL version torch is linked against, or None if unknowable."""
    try:
        return tuple(torch.cuda.nccl.version())  #ignore-cuda
    except Exception:
        # Not being able to tell is not a reason to block a run that might work.
        return None


class DeepEPExchange:
    """Wraps a DeepEP v2 ``ElasticBuffer`` for one MoE layer.

    Only v2 is supported. The legacy v1 ``Buffer`` moves data over NVSHMEM and
    IBGDA instead of NCCL, which needs either the NVreg_EnableStreamMemOPs
    driver parameter or the GDRCopy device, and it reports markedly lower
    internode bandwidth -- the case this backend exists to improve.

    DeepEP has no separate backward entry points. The gradient of a combine is
    a dispatch and the gradient of a dispatch is a combine, both replayed
    against the handle the forward dispatch produced, so the handle has to
    survive from forward to backward.
    """

    def __init__(self, ep_group, num_experts: int, top_k: int, hidden_size: int, num_max_tokens_per_rank: int):
        deep_ep = _import_deep_ep()

        self.deep_ep = deep_ep
        # Queue pairs are the scarce resource here. Left automatic, DeepEP
        # claims 65 to 129 of them, which is fine in a process that does
        # nothing else but fails in a training step where ZeRO and the
        # data-parallel groups have already taken their share. Asking for only
        # what the chosen SM count needs keeps the request proportionate.
        num_sms = configured_num_sms() or DEFAULT_COMM_SMS
        self.buffer = deep_ep.ElasticBuffer(
            ep_group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden_size,
            num_topk=top_k,
            use_fp8_dispatch=False,
            num_allocated_qps=_qps_for_sms(num_sms),
            # Required once the EP group spans nodes: it splits the ranks into
            # an NVLink domain and an RDMA domain rather than assuming a single
            # flat NVLink domain.
            allow_hybrid_mode=True,
            explicitly_destroy=True,
        )
        self.num_sms = num_sms
        self.num_qps = self.buffer.get_theoretical_num_qps(self.num_sms)
        self.num_experts = num_experts
        # The handle the last dispatch produced. Combine and both backward
        # passes replay against it, so it has to outlive the dispatch call.
        self.last_handle = None
        # The routing weights that arrived with the last dispatch, kept so the
        # layer can decide whether to apply them before or after the experts.
        self.last_recv_weights = None

    def dispatch(self, tokens: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor):
        """Send tokens to their experts, returning rows, weights and handle.

        The weights travel with the tokens because the reduction that uses
        them happens on the receiving side, after the experts have run.
        """
        recv_x, _, recv_weights, handle, _ = self.buffer.dispatch(
            tokens,
            topk_idx=topk_idx.to(self.deep_ep.topk_idx_t),
            # DeepEP reduces in float32, and the router's scores may be bf16.
            topk_weights=topk_weights.float(),
            num_experts=self.num_experts,
            # Group arrivals by expert rather than by source rank. The
            # grouped GEMM walks contiguous per-expert ranges, so the default
            # source-major layout has the right number of rows in an order the
            # GEMM cannot use.
            do_expand=True,
            # No per-expert padding: the counts that become the GEMM's group
            # offsets have to describe the rows that are actually there.
            expert_alignment=1,
            num_sms=self.num_sms,
            num_qps=self.num_qps,
        )
        # The returned event only holds anything when the call was made with
        # async_with_compute_stream; a synchronous result is already usable.
        self.last_handle = handle
        self.last_recv_weights = recv_weights
        return recv_x, recv_weights, handle

    def dispatch_with_handle(self, tokens: torch.Tensor, handle) -> torch.Tensor:
        recv_x, _, _, _, _ = self.buffer.dispatch(
            tokens,
            handle=handle,
            num_sms=self.num_sms,
            num_qps=self.num_qps,
        )
        return recv_x

    def combine(self, rows: torch.Tensor, handle, topk_weights=None) -> torch.Tensor:
        """Reduce expert outputs back to their tokens.

        Passing ``topk_weights`` makes DeepEP weight each expert's output as it
        reduces, which is the same weighted sum the NCCL path performs
        separately after its combine.
        """
        combined, _, _ = self.buffer.combine(rows, handle=handle, topk_weights=topk_weights, num_sms=self.num_sms)
        return combined

    def destroy(self) -> None:
        self.buffer.destroy()


def _conform_rows(tensor: torch.Tensor, shape) -> torch.Tensor:
    """Trim or zero-extend ``tensor`` to ``shape``'s row count.

    DeepEP returns whole buffers sized for the worst case, but autograd checks
    a gradient against the exact input it corresponds to. Rows beyond the ones
    that carried tokens hold no gradient, so trimming discards nothing and
    extending contributes nothing.
    """
    rows = shape[0]
    if tensor.shape[0] == rows:
        return tensor
    if tensor.shape[0] > rows:
        return tensor[:rows]
    extended = tensor.new_zeros((rows, tensor.shape[1]))
    extended[:tensor.shape[0]] = tensor
    return extended


class _DeepEPDispatch(torch.autograd.Function):
    """Forward dispatch whose backward is the matching combine."""

    @staticmethod
    def forward(ctx, exchange: DeepEPExchange, tokens: torch.Tensor, topk_idx: torch.Tensor,
                topk_weights: torch.Tensor):
        received, recv_weights, handle = exchange.dispatch(tokens, topk_idx, topk_weights)
        ctx.exchange = exchange
        ctx.handle = handle
        ctx.tokens_shape = tokens.shape
        ctx.recv_weights = recv_weights
        return received

    @staticmethod
    def backward(ctx, grad_received):
        grad_tokens = ctx.exchange.combine(grad_received.contiguous(), ctx.handle)
        return None, _conform_rows(grad_tokens, ctx.tokens_shape), None, None


class _DeepEPCombine(torch.autograd.Function):
    """Combine whose backward is the matching dispatch, on the same handle."""

    @staticmethod
    def forward(ctx, exchange: DeepEPExchange, rows: torch.Tensor, handle, topk_weights):
        ctx.exchange = exchange
        ctx.handle = handle
        # The forward input was trimmed to the rows that actually arrived,
        # while the backward dispatch hands back a whole buffer. Autograd
        # requires the gradient to match the input it is the gradient of.
        ctx.rows_shape = rows.shape
        return exchange.combine(rows, handle, topk_weights)

    @staticmethod
    def backward(ctx, grad_combined):
        grad_rows = ctx.exchange.dispatch_with_handle(grad_combined.contiguous(), ctx.handle)
        return None, _conform_rows(grad_rows, ctx.rows_shape), None, None


def deepep_dispatch(exchange: DeepEPExchange, tokens: torch.Tensor, topk_idx: torch.Tensor,
                    topk_weights: torch.Tensor):
    received = _DeepEPDispatch.apply(exchange, tokens, topk_idx, topk_weights)
    return received, exchange


def deepep_combine(exchange: DeepEPExchange, rows: torch.Tensor, handle, topk_weights=None) -> torch.Tensor:
    return _DeepEPCombine.apply(exchange, rows, handle, topk_weights)
