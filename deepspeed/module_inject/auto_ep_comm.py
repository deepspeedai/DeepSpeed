# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Selectable transports for the AutoEP expert all-to-all.

The dispatch and combine collectives are the largest single cost in an AutoEP
step, and a measured replay of real routing on 16 H100s across two nodes put
the collective path at 195.8 ms of payload all-to-all per step against DeepEP's
86.3 ms. This module is what lets that be switched without the MoE layer
knowing which transport it is using.

Selection lives in the ``expert_parallel`` section of the DeepSpeed config and
defaults to the collective path, so a job that sets nothing behaves exactly as
before::

    "expert_parallel": {"comm_backend": "comm"}      # default
    "expert_parallel": {"comm_backend": "deepep"}

A backend is asked for once per layer and reused, because DeepEP's buffers are
sized at construction and are expensive to rebuild.
"""

from __future__ import annotations

import torch

from deepspeed.utils import logger

# Names the transport, not the library behind it: the default path goes through
# deepspeed.comm, which is NCCL on CUDA but not on every accelerator.
COMM_BACKEND = "comm"
# Names the library, not its version: v2 is the only path implemented, and
# nothing about the name would have to change if that ever grew.
DEEPEP_BACKEND = "deepep"
AVAILABLE_BACKENDS = (COMM_BACKEND, DEEPEP_BACKEND)
# GIN, and so DeepEP, does not exist in any form below this NCCL version.
NCCL_GIN_MIN_VERSION = (2, 30, 4)
# The config's comm_num_sm default. Chosen by timing whole training steps
# rather than the collective alone, because the budget trades one against the
# other: the collective and the expert GEMM draw from the same SMs.
#
# On 16 H100s across two nodes, taking the fastest of three runs per budget and
# rotating the budgets through every position within one job, a step took
# 272.4 ms at 8 SMs and 256.9 ms at 12. Larger budgets were slower again, the
# collective having taken SMs the rest of the step was using. Interference on a
# shared cluster is one-sided, so the fastest run is the least contaminated
# estimate; a mean of two runs put 8 ahead until a single disturbed run was
# identified as the cause.
DEFAULT_COMM_SMS = 12


def _qps_for_sms(num_sms: int, qp_margin: int) -> int:
    """Queue pairs to reserve for a given SM count.

    One per SM plus a margin for the control path. This is deliberately smaller
    than DeepEP's automatic choice, which assumes it is the only thing on the
    fabric: in a training step ZeRO and the data-parallel groups have already
    taken their share, and asking for DeepEP's default exhausts them.
    """
    return num_sms + qp_margin


# Every buffer built in this process, in construction order. DeepEP buffers are
# constructed with explicitly_destroy, so nothing reclaims them on its own, and
# destroying them is collective: every rank has to do it in the same order.
_LIVE_EXCHANGES: list["DeepEPExchange"] = []

# DeepEP's dispatch kernel handles bfloat16 and fp8, not fp16. A half-precision
# run would otherwise reach an assertion inside the kernel.
SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


def assert_dtype_supported(dtype: torch.dtype) -> None:
    """Reject dtypes DeepEP's kernels cannot dispatch."""
    if dtype not in SUPPORTED_DTYPES:
        raise TypeError(f'comm_backend="{DEEPEP_BACKEND}" does not support {dtype}: DeepEP\'s dispatch kernel '
                        'handles bfloat16, not fp16. Train in bfloat16, or set comm_backend="comm" to use the '
                        "default all-to-all, which has no such restriction.")


def destroy_all_exchanges() -> None:
    """Release every DeepEP buffer this process built.

    Collective, and ordered by construction, so every rank tears the same
    buffers down in the same order. Worth calling at the end of training: the
    buffers ask DeepEP not to reclaim them, so nothing else will.
    """
    for exchange in list(_LIVE_EXCHANGES):
        exchange.destroy()


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
            f'comm_backend="{DEEPEP_BACKEND}" requires the deep_ep package, which is not installed. It also '
            "requires NCCL 2.30.4 or newer built with GIN support: the transport is unavailable below that "
            'version regardless of the network. Set comm_backend="comm" to use the default all-to-all, which '
            "has no such requirement.") from error

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
            'construction is the symptom. Set comm_backend="comm" to fall back to the default all-to-all.')
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

    def __init__(self,
                 ep_group,
                 num_experts: int,
                 top_k: int,
                 hidden_size: int,
                 num_max_tokens_per_rank: int,
                 num_sms: int = DEFAULT_COMM_SMS,
                 qp_margin: int = 4):
        deep_ep = _import_deep_ep()

        self.deep_ep = deep_ep
        # Queue pairs are the scarce resource here. Left automatic, DeepEP
        # claims 65 to 129 of them, which is fine in a process that does
        # nothing else but fails in a training step where ZeRO and the
        # data-parallel groups have already taken their share. Asking for only
        # what the chosen SM count needs keeps the request proportionate.
        self.buffer = deep_ep.ElasticBuffer(
            ep_group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden_size,
            num_topk=top_k,
            use_fp8_dispatch=False,
            num_allocated_qps=_qps_for_sms(num_sms, qp_margin),
            # Required once the EP group spans nodes: it splits the ranks into
            # an NVLink domain and an RDMA domain rather than assuming a single
            # flat NVLink domain.
            allow_hybrid_mode=True,
            explicitly_destroy=True,
        )
        self.num_sms = num_sms
        self.num_experts = num_experts
        # Recorded so the layer can tell when a later batch outgrows it.
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        # The handle the last dispatch produced. Combine and both backward
        # passes replay against it, so it has to outlive the dispatch call.
        self.last_handle = None
        self.destroyed = False
        # Constructed with explicitly_destroy, so nothing reclaims this buffer
        # on its own. Registering it means a process that never calls the
        # layer's teardown can still release every buffer in one call.
        _LIVE_EXCHANGES.append(self)
        # Buffer construction is collective and allocates fabric resources, so
        # it is where an unsuitable cluster fails, often by killing the process
        # without raising. Recording each one lets a post-mortem tell a buffer
        # that was never built from one that was built and then used.
        logger.info(f"AutoEP DeepEP buffer {len(_LIVE_EXCHANGES)} built: "
                    f"capacity={num_max_tokens_per_rank} sms={num_sms} qps={_qps_for_sms(num_sms, qp_margin)}")

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
        )
        # The returned event only holds anything when the call was made with
        # async_with_compute_stream; a synchronous result is already usable.
        self.last_handle = handle
        return recv_x, recv_weights, handle

    def dispatch_with_handle(self, tokens: torch.Tensor, handle) -> torch.Tensor:
        """Replay a dispatch against a cached handle.

        Used as the backward of a combine, which scatters the combined
        gradient back to the rows that contributed to it.
        """
        recv_x, _, _, _, _ = self.buffer.dispatch(
            tokens,
            handle=handle,
            num_sms=self.num_sms,
        )
        return recv_x

    def combine_with_weight_grad(self, rows: torch.Tensor, handle, weight_grads=None):
        """Combine that also reduces the routing-weight gradient.

        Used as the backward of a dispatch. Dispatch replicates a token's
        routing weight to every rank that expert-owns it, so the adjoint is a
        sum over those copies, which is exactly what combine does to the
        weights it carries.
        """
        combined, combined_weights, _ = self.buffer.combine(rows,
                                                            handle=handle,
                                                            topk_weights=weight_grads,
                                                            num_sms=self.num_sms)
        return combined, combined_weights

    def combine(self, rows: torch.Tensor, handle) -> torch.Tensor:
        """Reduce expert outputs back to the tokens they came from.

        Deliberately does not pass ``topk_weights``. DeepEP's combine does not
        multiply the rows by those weights; it transports and reduces them
        alongside, returning them separately. Handing the routing weights here
        would therefore drop them from the result, so the layer applies them to
        the rows itself.
        """
        combined, _, _ = self.buffer.combine(rows, handle=handle, num_sms=self.num_sms)
        return combined

    def destroy(self) -> None:
        """Release the buffer. Collective, so every rank must call it."""
        if self.destroyed:
            return
        self.destroyed = True
        self.buffer.destroy()
        if self in _LIVE_EXCHANGES:
            _LIVE_EXCHANGES.remove(self)


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
    extended = tensor.new_zeros((rows, ) + tuple(tensor.shape[1:]))
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
        ctx.weights_shape = None if topk_weights is None else topk_weights.shape
        # Dispatch moves the weights alongside the tokens, so the received
        # copies are what downstream code differentiates; returning them makes
        # autograd carry their gradient back to the router gate. Without this
        # the gate silently receives nothing and stops learning.
        return received, recv_weights

    @staticmethod
    def backward(ctx, grad_received, grad_recv_weights):
        grad_tokens, grad_weights = ctx.exchange.combine_with_weight_grad(
            grad_received.contiguous(),
            ctx.handle,
            None if grad_recv_weights is None else grad_recv_weights.contiguous(),
        )
        conformed_weights = None
        if grad_weights is not None and ctx.weights_shape is not None:
            conformed_weights = _conform_rows(grad_weights, ctx.weights_shape).reshape(ctx.weights_shape)
        return None, _conform_rows(grad_tokens, ctx.tokens_shape), None, conformed_weights


class _DeepEPCombine(torch.autograd.Function):
    """Combine whose backward is the matching dispatch, on the same handle."""

    @staticmethod
    def forward(ctx, exchange: DeepEPExchange, rows: torch.Tensor, handle):
        ctx.exchange = exchange
        ctx.handle = handle
        # The backward dispatch hands back a whole buffer, while autograd
        # requires the gradient to match the input it is the gradient of.
        ctx.rows_shape = rows.shape
        return exchange.combine(rows, handle)

    @staticmethod
    def backward(ctx, grad_combined):
        grad_rows = ctx.exchange.dispatch_with_handle(grad_combined.contiguous(), ctx.handle)
        return None, _conform_rows(grad_rows, ctx.rows_shape), None


def deepep_dispatch(exchange: DeepEPExchange, tokens: torch.Tensor, topk_idx: torch.Tensor,
                    topk_weights: torch.Tensor):
    """Dispatch tokens and their routing weights, keeping both differentiable."""
    received, recv_weights = _DeepEPDispatch.apply(exchange, tokens, topk_idx, topk_weights)
    return received, recv_weights, exchange


def deepep_combine(exchange: DeepEPExchange, rows: torch.Tensor, handle) -> torch.Tensor:
    return _DeepEPCombine.apply(exchange, rows, handle)
