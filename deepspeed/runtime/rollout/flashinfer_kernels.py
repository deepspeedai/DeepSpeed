# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""FlashInfer kernel integration for HybridEngineRollout.

Provides a context manager that swaps in FlashInfer-optimized kernels for
attention and sampling during rollout decode, and restores the original
implementations on exit.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_flashinfer_available: Optional[bool] = None


def _check_flashinfer():
    global _flashinfer_available
    if _flashinfer_available is None:
        try:
            import flashinfer as _fi  # noqa: F401
            _flashinfer_available = True
        except ImportError:
            _flashinfer_available = False
    return _flashinfer_available


def flashinfer_sample_top_p(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
    """Sample from logits using FlashInfer's fused top-p sampling kernel."""
    from flashinfer.sampling import top_p_sampling_from_probs
    logits = logits / temperature
    probs = torch.softmax(logits.float(), dim=-1)
    sampled = top_p_sampling_from_probs(probs, top_p)
    return sampled.unsqueeze(-1)


def _register_flashinfer_attention():
    """Register a FlashInfer-backed attention function with HF Transformers.

    This replaces the SDPA attention call with FlashInfer's
    ``single_decode_with_kv_cache`` for decode and a simple matmul fallback
    for prefill.  The function is registered under the name ``"flashinfer"``
    so that switching ``model.config._attn_implementation`` to ``"flashinfer"``
    activates it.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if "flashinfer" in ALL_ATTENTION_FUNCTIONS.valid_keys():
        return

    def _flashinfer_attention_forward(module,
                                      query_states,
                                      key_states,
                                      value_states,
                                      attention_mask,
                                      dropout=0.0,
                                      scaling=None,
                                      sliding_window=None,
                                      **kwargs):
        # Decode path: query has seq_len=1
        if query_states.shape[2] == 1:
            from flashinfer.decode import single_decode_with_kv_cache

            sm_scale = scaling if scaling is not None else (query_states.shape[-1]**-0.5)
            window_left = -1
            if sliding_window is not None:
                window_left = sliding_window - 1

            output = single_decode_with_kv_cache(query_states,
                                                 key_states,
                                                 value_states,
                                                 sm_scale=sm_scale,
                                                 window_left=window_left)
            return output.squeeze(2).transpose(1, 2), None

        # Prefill / longer sequences: delegate to the original SDPA handler
        # which correctly handles GQA (repeat_kv for different q/kv head counts).
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        sdpa_fn = ALL_ATTENTION_FUNCTIONS.get("sdpa")
        if sdpa_fn is not None:
            return sdpa_fn(module,
                           query_states,
                           key_states,
                           value_states,
                           attention_mask,
                           dropout=dropout,
                           scaling=scaling,
                           sliding_window=sliding_window,
                           **kwargs)
        import torch.nn.functional as F
        output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
        )
        return output.transpose(1, 2), None

    ALL_ATTENTION_FUNCTIONS.register("flashinfer", _flashinfer_attention_forward)
    logger.info("FlashInfer attention backend registered with HF Transformers.")


class FlashInferKernelManager:
    """Context manager that enables FlashInfer kernels for rollout decode.

    On ``__enter__``, patches:
      - model.config._attn_implementation → ``"flashinfer"``
      - HybridEngineRollout._sample_top_p → flashinfer_sample_top_p

    On ``__exit__``, restores everything.

    Usage::

        mgr = FlashInferKernelManager(engine, rollout)
        with mgr:
            rollout.generate(request, sampling)
    """

    def __init__(self, engine, rollout):
        self.engine = engine
        self.rollout = rollout
        self._orig_attn_impl = None
        self._orig_sample = None

    def __enter__(self):
        if not _check_flashinfer():
            logger.warning("FlashInfer not available, skipping kernel swap.")
            return self

        try:
            _register_flashinfer_attention()

            module = self.engine.module
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "_attn_implementation"):
                self._orig_attn_impl = config._attn_implementation
                config._attn_implementation = "flashinfer"
                logger.info(f"Switched attention backend: {self._orig_attn_impl} → flashinfer")

            self._orig_sample = self.rollout._sample_top_p
            self.rollout._sample_top_p = staticmethod(flashinfer_sample_top_p).__func__
            logger.info("Switched sampling to FlashInfer top_p_sampling_from_probs.")
        except Exception as e:
            logger.warning(f"FlashInfer kernel setup failed, reverting: {e}")
            self.__exit__(None, None, None)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._orig_attn_impl is not None:
            try:
                module = self.engine.module
                module.config._attn_implementation = self._orig_attn_impl
                logger.info(f"Restored attention backend: {self._orig_attn_impl}")
            except Exception:
                pass
            self._orig_attn_impl = None

        if self._orig_sample is not None:
            self.rollout._sample_top_p = self._orig_sample
            logger.info("Restored original sampling.")
            self._orig_sample = None

        return False
