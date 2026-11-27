# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine that generates with an AutoTP tensor-parallel copy of the training model.

The training engine keeps the model in its ZeRO layout. This engine keeps a second, inference
copy of the same model sharded with AutoTP (``init_inference`` without kernel injection), and
``sync_weights`` refreshes that copy in memory from the training engine's fp32 masters:

  * gather: ``safe_get_full_fp32_param`` reassembles each ZeRO-partitioned master weight.
  * slice:  ``_resolve_autotp_partition`` cuts it into this rank's shard, using the layout
    metadata every AutoTP layer records on its weight (the same function that restores a
    universal checkpoint into AutoTP, so the two cannot disagree on a layout).

One tensor-parallel group spans all ranks, so the ranks generate together: each rank's prompts
are gathered, the union is generated once, and every rank keeps its own samples.

Supported: the ``model.generate()`` path. Not supported yet: graph capture, shared prefill, and
continuous batching.
"""

import time
from copy import deepcopy
from types import SimpleNamespace

import torch

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.universal_checkpoint import _resolve_autotp_partition
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutRequest
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout
from deepspeed.utils import safe_get_full_fp32_param


def _params_with_owner(module):
    """Map each parameter name to (parameter, the module that owns it)."""
    owned = {}
    for module_name, owner in module.named_modules():
        for param_name, param in owner.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            owned[full_name] = (param, owner)
    return owned


@torch.no_grad()
def sync_autotp_weights(train_module, tp_module):
    """Overwrite every parameter of ``tp_module`` with the current fp32 masters of ``train_module``.

    Every rank must call this: the gather is a collective, and ``named_parameters`` yields the
    same order on every rank, so the collectives pair up the same tensor everywhere.
    Returns the number of tensors copied.
    """
    tp_params = _params_with_owner(tp_module)
    copied = 0
    for tp_name, train_param in train_module.named_parameters():
        full = safe_get_full_fp32_param(train_param)
        assert full is not None, f"{tp_name} has no ZeRO fp32 master; is the optimizer a ZeRO optimizer?"
        assert tp_name in tp_params, f"inference model has no parameter {tp_name}"
        tp_param, owner = tp_params[tp_name]

        # Only AutoTP layers carry tp_index; other modules (norms, embeddings) are replicated.
        tp_rank = getattr(owner, "tp_index", 0)
        tp_world_size = getattr(owner, "tp_world_size", 1)
        shard = _resolve_autotp_partition(tp_param, {}, full, tp_rank, tp_world_size)
        if shard is None:
            assert full.shape == tp_param.shape, (
                f"{tp_name}: no AutoTP layout recorded, but full shape {tuple(full.shape)} != "
                f"inference shape {tuple(tp_param.shape)}")
            shard = full
        # copy_ rounds fp32 to the inference dtype the same way ZeRO rounds its low-precision params.
        tp_param.copy_(shard.view(tp_param.shape))
        copied += 1

    assert copied == len(tp_params), f"synced {copied} tensors, but the inference model has {len(tp_params)}"
    return copied


class AutoTPRollout(HybridEngineRollout):
    """Single-turn rollout with an AutoTP copy of the training model, refreshed by ``sync_weights``.

    Args:
        engine: DeepSpeed ZeRO training engine whose ``module`` is the Hugging Face causal LM.
        tokenizer: HuggingFace tokenizer (must have pad_token_id or eos_token_id).
        cfg: Rollout config; ``use_graph_capture`` and ``use_shared_prefill`` must be off.
    """

    name = "autotp"

    def __init__(self, engine, tokenizer, cfg=None):
        super().__init__(engine=engine, tokenizer=tokenizer, cfg=cfg)
        if self.use_graph_capture or self.use_shared_prefill:
            raise NotImplementedError("AutoTP rollout supports the generate() path only; turn off "
                                      "use_graph_capture and use_shared_prefill")
        self.train_engine = engine
        self.tp_size = dist.get_world_size()
        # HybridEngineRollout generates through self.engine.module, so point it at the AutoTP copy.
        self.engine = SimpleNamespace(module=self._build_tp_copy(engine.module))

        # Every rank must sample the same tokens, so all ranks draw from one seed per generate call.
        seed = torch.randint(0, 2**31 - 1, (1, ), device=get_accelerator().current_device_name())
        dist.broadcast(seed, src=0)
        self._seed = seed.item()
        self._generate_calls = 0
        self.last_sync = None
        # The copy starts with random weights; it must never generate before its first sync.
        self.sync_weights(step=0)

    def _build_tp_copy(self, train_model):
        # Imported here: deepspeed/__init__.py imports the runtime, so a top-level import would be circular.
        import deepspeed
        from transformers import AutoModelForCausalLM

        dtype = next(train_model.parameters()).dtype
        # Built on the accelerator with random weights: sync_weights overwrites every tensor, so
        # loading weights from disk here would only cost time and host memory.
        with torch.device(get_accelerator().current_device_name()):
            model = AutoModelForCausalLM.from_config(train_model.config,
                                                     dtype=dtype,
                                                     attn_implementation=train_model.config._attn_implementation)
        model.generation_config = deepcopy(train_model.generation_config)
        inference_engine = deepspeed.init_inference(model.eval(),
                                                    config={
                                                        "dtype": dtype,
                                                        "tensor_parallel": {
                                                            "tp_size": self.tp_size
                                                        },
                                                        "replace_with_kernel_inject": False,
                                                        "enable_cuda_graph": False,
                                                    })
        return inference_engine.module

    def sync_weights(self, step: int) -> None:
        """Copy the training engine's current weights into the AutoTP copy (collective)."""
        accelerator = get_accelerator()
        accelerator.synchronize()
        start = time.perf_counter()
        copied = sync_autotp_weights(self.train_engine.module, self.engine.module)
        accelerator.synchronize()
        self.last_sync = {"step": step, "tensors": copied, "seconds": time.perf_counter() - start}

    @torch.no_grad()
    def generate(self, request, sampling):
        if sampling.continuous_batch_size is not None:
            raise NotImplementedError("AutoTP rollout does not support continuous batching")
        rank = dist.get_rank()
        gathered = [None] * self.tp_size
        dist.all_gather_object(gathered, (request.prompt_ids.cpu(), request.prompt_attention_mask.cpu()))
        union_request = self._left_padded_union(gathered, request.prompt_ids.device)

        batch = self._generate_with_shared_seed(union_request, sampling)
        self._check_same_tokens_on_all_ranks(batch)

        # Generation keeps each prompt's n copies adjacent and in prompt order.
        n = sampling.n_samples_per_prompt
        counts = [prompt_ids.shape[0] for prompt_ids, _ in gathered]
        start = sum(counts[:rank]) * n
        end = start + counts[rank] * n
        # Columns added only because another rank's prompts were longer are all padding here; dropping
        # them returns the [B', T_p + T_r] layout that this rank's own request promises.
        extra_columns = union_request.prompt_ids.shape[1] - request.prompt_ids.shape[1]
        return RolloutBatch(input_ids=batch.input_ids[start:end, extra_columns:],
                            attention_mask=batch.attention_mask[start:end, extra_columns:],
                            response_start_idx=batch.response_start_idx[start:end] - extra_columns)

    def _left_padded_union(self, gathered, device):
        """Stack every rank's left-padded prompts into one request; ranks' prompt widths can differ."""
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("The tokenizer must define pad_token_id or eos_token_id")

        width = max(prompt_ids.shape[1] for prompt_ids, _ in gathered)
        all_ids = []
        all_masks = []
        for prompt_ids, attention_mask in gathered:
            padding = (width - prompt_ids.shape[1], 0)
            all_ids.append(torch.nn.functional.pad(prompt_ids, padding, value=pad_token_id))
            all_masks.append(torch.nn.functional.pad(attention_mask, padding, value=0))
        return RolloutRequest(prompt_ids=torch.cat(all_ids).to(device),
                              prompt_attention_mask=torch.cat(all_masks).to(device))

    def _generate_with_shared_seed(self, request, sampling):
        accelerator = get_accelerator()
        # Restored below so that generating does not change the caller's training randomness.
        cpu_rng_state = torch.get_rng_state()
        device_rng_state = accelerator.get_rng_state()
        seed = self._seed + self._generate_calls
        self._generate_calls += 1
        try:
            torch.manual_seed(seed)
            accelerator.manual_seed(seed)
            return super().generate(request, sampling)
        finally:
            torch.set_rng_state(cpu_rng_state)
            accelerator.set_rng_state(device_rng_state)

    def _check_same_tokens_on_all_ranks(self, batch):
        gathered = [None] * self.tp_size
        input_ids = batch.input_ids.cpu()
        dist.all_gather_object(gathered, input_ids)
        if any(not torch.equal(other, input_ids) for other in gathered):
            raise RuntimeError("AutoTP ranks generated different tokens; the tensor-parallel copy is out of sync")
