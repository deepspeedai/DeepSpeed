# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""vLLM rollout on a disjoint GPU group.

**Topology (intended)**
    * Training ranks 0..N-1 run the student under ZeRO-3 on the first N GPUs.
    * vLLM workers run on the device indices listed in ``cfg.gpus`` (or in
      "shared" mode, alongside training rank 0).
    * The vLLM ``LLM`` handle is constructed **only on training rank 0**.
    * Other training ranks receive generated token ids by broadcast from
      rank 0 (:func:`deepspeed.comm.broadcast_object_list`).

**Weight sync**
    * All training ranks cooperatively gather each ZeRO-3 parameter via
      :class:`deepspeed.runtime.zero.GatheredParameters`.
    * Rank 0 pushes the full tensor to vLLM via ``LLM.collective_rpc(...)``,
      which dispatches to every vLLM worker; each worker uses its own TP rank
      to slice and load.

**KNOWN BLOCKING ISSUE — same-process vLLM under the DeepSpeed launcher**

    vLLM's worker initialisation calls ``new_group(...)`` on the global
    process group as a collective. Under the standard DeepSpeed launcher
    (e.g. ``deepspeed --num_gpus 2``) the world spans **all** training
    ranks, but only rank 0 calls into vLLM. The other training ranks never
    participate in vLLM's collective, so the ``LLM`` constructor hangs
    forever waiting on them.

    This was reproduced with vllm 0.6.6 + deepspeed 0.15.4 + torch 2.5.1; the
    same code-path completes in seconds when ``LLM`` is constructed in a
    process whose world size is 1. Verified by minimal repro (rank 0 LLM
    init blocks; rank 1 idle).

    **Workarounds (none currently implemented):**
      1. Run vLLM in a **separate top-level Python process** with its own
         world (size 1), and have the trainer talk to it over an HTTP or
         RPC channel. This is what TRL and OpenRLHF do for their vLLM
         backends.
      2. Spawn vLLM as a subprocess from rank 0 and tunnel calls through a
         queue. Similar to (1) but lower-level.
      3. Wait for upstream vLLM to expose a flag that skips its internal
         ``new_group`` calls when the caller already owns process-group
         setup.

    Until one of those lands, **the vLLM rollout in this PR is verified at
    the unit-test level only** (see ``tests/test_vllm_stitch.py`` and
    ``tests/test_weight_bridge.py``). The hybrid engine rollout is the
    fully-validated live path. See the project README's "vLLM status"
    section for current state.
"""

import os
from typing import Any, List, Optional

import torch

from opsd.config import RolloutConfig
from opsd.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig
from opsd.weight_bridge import WeightBridge, get_bridge


def _is_rank_zero() -> bool:
    # Deferred so this module remains importable in CPU-only test envs that
    # don't have ``deepspeed`` available (the ``stitch_rollout`` helper below
    # is pure tensor math and is unit-tested without DeepSpeed).
    from deepspeed import comm as dist

    return (not dist.is_initialized()) or dist.get_rank() == 0


def stitch_rollout(
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    responses: List[List[int]],
    pad_id: int,
    n_samples_per_prompt: int,
) -> RolloutBatch:
    """Stitch left-padded prompts and per-sample response token ids into one
    right-padded ``RolloutBatch``.

    This is the only piece of vLLM-side post-processing that doesn't depend
    on a live LLM handle, so we factor it out for CPU unit testing.

    Args:
        prompt_ids: ``[B, T_p]`` left-padded prompts.
        prompt_attention_mask: ``[B, T_p]`` matching attention mask.
        responses: list of length ``B * n_samples_per_prompt``; each element
            is the list of generated token ids for one (prompt, sample).
        pad_id: pad token used for both prompt left-padding and response
            right-padding (typically the tokenizer's ``pad_token_id`` or
            ``eos_token_id``).
        n_samples_per_prompt: number of generated samples per prompt.

    Returns:
        :class:`RolloutBatch` with ``response_start_idx = T_p`` for every
        sample.
    """
    B, T_p = prompt_ids.shape
    n = n_samples_per_prompt
    expected = B * n
    if len(responses) != expected:
        raise ValueError(f"expected {expected} response token-id lists "
                         f"(B={B} * n_samples={n}); got {len(responses)}")

    if responses:
        max_response_len = max(len(r) for r in responses)
    else:
        max_response_len = 0
    T_total = T_p + max_response_len
    device = prompt_ids.device

    out_ids = torch.full((expected, T_total), pad_id, dtype=torch.long, device=device)
    out_attn = torch.zeros((expected, T_total), dtype=prompt_attention_mask.dtype, device=device)

    prompts_expanded = prompt_ids.repeat_interleave(n, dim=0)
    attn_expanded = prompt_attention_mask.repeat_interleave(n, dim=0)
    out_ids[:, :T_p] = prompts_expanded
    out_attn[:, :T_p] = attn_expanded

    for i, resp in enumerate(responses):
        L = len(resp)
        if L == 0:
            continue
        out_ids[i, T_p:T_p + L] = torch.tensor(resp, dtype=torch.long, device=device)
        out_attn[i, T_p:T_p + L] = 1

    response_start_idx = torch.full((expected, ), T_p, dtype=torch.long, device=device)
    return RolloutBatch(input_ids=out_ids, attention_mask=out_attn, response_start_idx=response_start_idx)


class VLLMRollout(RolloutEngine):

    name = "vllm"

    def __init__(
        self,
        cfg: RolloutConfig,
        tokenizer: Any,
        student_engine: Any = None,
        student_model_path: Optional[str] = None,
        arch: Optional[str] = None,
    ):
        if cfg.engine != "vllm":
            raise ValueError(f"RolloutConfig.engine must be 'vllm'; got {cfg.engine!r}")
        if student_model_path is None:
            raise ValueError("VLLMRollout needs student_model_path to initialise the vLLM engine "
                             "(it loads weights from disk at construction time)")

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.student_engine = student_engine
        self._model_path = student_model_path

        self.is_rank_zero = _is_rank_zero()
        self.llm: Optional[Any] = None
        self.bridge: Optional[WeightBridge] = get_bridge(arch) if arch is not None else None

        if self.is_rank_zero:
            self._init_vllm()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _init_vllm(self) -> None:
        # Topology selection:
        #   * cfg.gpus empty  → SHARED: vLLM runs in-process on the same GPU
        #     as training rank 0. Simple; no CUDA visibility tricks. Used for
        #     smoke tests and when vLLM + student fit alongside each other.
        #   * cfg.gpus set    → DISJOINT: vLLM workers are pinned to the
        #     listed devices via CUDA_VISIBLE_DEVICES + a spawn-mode
        #     subprocess executor so the new CUDA context isn't inherited
        #     from the already-initialised rank-0 process.
        shared = not self.cfg.gpus

        prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        prev_mp = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
        if not shared:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.cfg.gpus)
            # Must be set before the vllm import; the value is read at import time.
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        try:
            try:
                from vllm import LLM
            except ImportError as e:
                raise ImportError(f"VLLMRollout requires vllm>={self.cfg.vllm_min_version}. "
                                  f"Install with: pip install 'vllm>={self.cfg.vllm_min_version}'") from e

            llm_kwargs = dict(
                model=self._model_path,
                tensor_parallel_size=self.cfg.tensor_parallel_size,
                gpu_memory_utilization=self.cfg.gpu_memory_utilization,
                dtype=self.cfg.vllm_dtype,
                enforce_eager=self.cfg.vllm_enforce_eager,
            )
            if not shared:
                llm_kwargs["distributed_executor_backend"] = "mp"
            self.llm = LLM(**llm_kwargs)
        finally:
            if prev_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
            if prev_mp is None:
                os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
            else:
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = prev_mp

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        B = int(request.prompt_ids.shape[0])
        n = sampling.n_samples_per_prompt

        if self.is_rank_zero:
            from vllm import SamplingParams

            # We send prompt *token ids* rather than text to vLLM so the
            # generation stays bit-exact with how the trainer tokenised. This
            # avoids any subtle BOS / special-token differences between the
            # trainer's and vLLM's text->id paths.
            prompt_token_ids: List[List[int]] = []
            for i in range(B):
                mask = request.prompt_attention_mask[i].bool()
                ids = request.prompt_ids[i][mask].tolist()
                prompt_token_ids.append(ids)

            sp = SamplingParams(
                n=n,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k if sampling.top_k > 0 else -1,
                max_tokens=sampling.max_new_tokens,
            )
            results = self.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sp, use_tqdm=False)
            responses: List[List[int]] = []
            for r in results:
                for out in r.outputs:
                    responses.append(list(out.token_ids))
        else:
            responses = []

        from deepspeed import comm as dist

        if dist.is_initialized() and dist.get_world_size() > 1:
            obj = [responses]
            dist.broadcast_object_list(obj, src=0)
            responses = obj[0]

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        return stitch_rollout(
            prompt_ids=request.prompt_ids,
            prompt_attention_mask=request.prompt_attention_mask,
            responses=responses,
            pad_id=pad_id,
            n_samples_per_prompt=n,
        )

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def sync_weights_from_student(self, step: int) -> None:
        if self.student_engine is None:
            return
        if self.bridge is None:
            # Best-effort inference of arch from the student model class name.
            model = self.student_engine.module
            cls = type(model).__name__.lower()
            if "qwen3" in cls:
                self.bridge = get_bridge("qwen3")
            elif "qwen2" in cls:
                self.bridge = get_bridge("qwen2")
            else:
                raise RuntimeError(f"Cannot infer weight bridge for student class {cls!r}; "
                                   f"set StudentConfig.arch explicitly")

        from deepspeed.runtime.zero import GatheredParameters

        model = self.student_engine.module
        for name, param in model.named_parameters():
            # GatheredParameters is a no-op when ZeRO stage < 3, and a full
            # all-gather when stage == 3. Either way every rank sees the full
            # tensor inside the context; only rank 0 forwards it to vLLM.
            with GatheredParameters([param], modifier_rank=0):
                if not self.is_rank_zero:
                    continue
                # Sanity-check the param name against the bridge so a renamed
                # parameter trips here (cheap) rather than as a silent layout
                # mismatch inside vLLM later (very hard to debug).
                self.bridge.parallel_kind(name)
                self._push_one_param(name, param.data.detach())

    def _push_one_param(self, name: str, tensor: torch.Tensor) -> None:
        # collective_rpc dispatches to every vLLM worker; pickle handles the
        # tensor transfer. CPU tensors pickle cleanly across process bounds.
        cpu = tensor.contiguous().cpu()
        # vLLM's per-architecture model class exposes ``load_weights`` taking
        # an iterable of (name, tensor) pairs and internally handles QKV /
        # gate_up fusion plus per-rank slicing for tensor parallelism.
        self.llm.collective_rpc("load_weights", args=([(name, cpu)], ))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self.llm is not None:
            del self.llm
            self.llm = None
