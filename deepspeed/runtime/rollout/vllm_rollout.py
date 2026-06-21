# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""vLLM rollout via an external OpenAI-compatible server process.

**Architecture**
    Training ranks run under the DeepSpeed launcher as usual.  Rank 0 lazily
    spawns ``python -m vllm.entrypoints.openai.api_server ...`` as a
    **separate subprocess** with its own CUDA device visibility, then
    communicates with it over HTTP using the OpenAI-compatible completions
    API.  Other ranks receive generated token ids by broadcast from rank 0
    (:func:`deepspeed.comm.broadcast_object_list`).

**Why a subprocess?**
    vLLM's worker initialisation calls ``new_group(...)`` on the global
    process group as a collective.  Under the DeepSpeed launcher the world
    spans *all* training ranks, but only rank 0 talks to vLLM.  Running
    vLLM in-process therefore deadlocks.  The subprocess approach gives
    vLLM its own world (size = TP) and avoids the conflict entirely.

**GPU placement**
    ``cfg.gpus`` controls which physical GPUs the vLLM server sees via
    ``CUDA_VISIBLE_DEVICES``.  These may be disjoint from the training GPUs
    (the safe default) or overlap when ``cfg.gpus`` is empty (shared mode,
    which requires the training loop to release GPU memory first).

**Weight sync (vLLM >= 0.22.0)**
    vLLM 0.22.0 exposes an RLHF weight-transfer API when started with
    ``VLLM_SERVER_DEV_MODE=1`` and ``--weight-transfer-config``.  The
    protocol is: ``pause`` -> ``start_weight_update`` ->
    ``update_weights`` -> ``finish_weight_update`` -> ``resume``.

    Two transport backends are supported:

    * **GDR** (GPU-direct) – NCCL broadcast over a
      ``StatelessProcessGroup``.  Fastest, but requires NCCL (NVIDIA).
    * **HTTP** – serialize tensors and send over HTTP.  Slower but
      accelerator-agnostic.

    When ``weight_transfer_backend="auto"`` (default), GDR is tried
    first and falls back to HTTP if NCCL is unavailable.
"""

import logging
import os
import socket
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch

from deepspeed.runtime.rlhf.config import RolloutConfig
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 120
_VLLM_NCCL_BACKEND = "nccl"


def _gdr_available() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.nccl.version() is not None  #ignore-cuda
    except Exception:
        return False


def _is_rank_zero() -> bool:
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
    on a live server, so we factor it out for CPU unit testing.

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
        self._server_proc: Optional[subprocess.Popen] = None
        self._base_url = f"http://localhost:{cfg.vllm_port}"
        self._ready = False

        self._nccl_group = None
        self._weight_transfer_inited = False

        backend = cfg.weight_transfer_backend
        if backend == "auto":
            backend = "gdr" if _gdr_available() else "http"
        if backend not in ("gdr", "http"):
            raise ValueError(f"weight_transfer_backend must be 'auto', 'gdr', or 'http'; got {backend!r}")
        self._wt_backend = backend

    # ------------------------------------------------------------------
    # Lazy server lifecycle
    # ------------------------------------------------------------------

    def _ensure_server(self) -> None:
        """Start the vLLM server on first use (rank 0 only).

        All ranks barrier here so non-zero ranks wait until rank 0 has
        confirmed the server is healthy.
        """
        if self._ready:
            return

        from deepspeed import comm as dist

        if self.is_rank_zero:
            self._start_server()
            self._wait_for_health()

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

        self._ready = True

    def _start_server(self) -> None:
        env = os.environ.copy()
        if self.cfg.gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.cfg.gpus)
        env.pop("VLLM_WORKER_MULTIPROC_METHOD", None)

        env["VLLM_SERVER_DEV_MODE"] = "1"

        python_bin = self.cfg.vllm_python or sys.executable
        cmd = [
            python_bin,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self._model_path,
            "--tensor-parallel-size",
            str(self.cfg.tensor_parallel_size),
            "--dtype",
            self.cfg.vllm_dtype,
            "--gpu-memory-utilization",
            str(self.cfg.gpu_memory_utilization),
            "--port",
            str(self.cfg.vllm_port),
            "--weight-transfer-config",
            f'{{"backend": "{_VLLM_NCCL_BACKEND}"}}' if self._wt_backend == "gdr" else '{"backend": "http"}',
        ]
        if self.cfg.vllm_enforce_eager:
            cmd.append("--enforce-eager")

        logger.info("Starting vLLM server: %s", " ".join(cmd))
        self._server_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def _wait_for_health(self) -> None:
        deadline = time.monotonic() + self.cfg.vllm_start_timeout
        while time.monotonic() < deadline:
            if self._server_proc is not None and self._server_proc.poll() is not None:
                rc = self._server_proc.returncode
                stderr_tail = ""
                if self._server_proc.stderr is not None:
                    stderr_tail = self._server_proc.stderr.read().decode(errors="replace")[-3000:]
                raise RuntimeError(f"vLLM server exited prematurely (rc={rc}). stderr tail:\n{stderr_tail}")
            try:
                resp = requests.get(f"{self._base_url}/health", timeout=2)
                if resp.status_code == 200:
                    logger.info("vLLM server is healthy on port %d", self.cfg.vllm_port)
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)
        raise TimeoutError(f"vLLM server did not become healthy within {self.cfg.vllm_start_timeout}s")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        self._ensure_server()

        B = int(request.prompt_ids.shape[0])
        n = sampling.n_samples_per_prompt

        if self.is_rank_zero:
            prompt_token_ids: List[List[int]] = []
            for i in range(B):
                mask = request.prompt_attention_mask[i].bool()
                ids = request.prompt_ids[i][mask].tolist()
                prompt_token_ids.append(ids)

            payload: Dict[str, Any] = {
                "model": self._model_path,
                "prompt": prompt_token_ids,
                "n": n,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "max_tokens": sampling.max_new_tokens,
                "logprobs": 1,
            }
            if sampling.top_k > 0:
                payload["top_k"] = sampling.top_k

            resp = requests.post(
                f"{self._base_url}/v1/completions",
                json=payload,
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            body = resp.json()

            responses: List[List[int]] = []
            for choice in body["choices"]:
                responses.append(self._extract_token_ids(choice))
        else:
            responses = []

        from deepspeed import comm as dist

        if dist.is_initialized() and dist.get_world_size() > 1:
            obj = [responses]
            dist.broadcast_object_list(obj, src=0)
            responses = obj[0]

        pad_id = (self.tokenizer.pad_token_id
                  if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        return stitch_rollout(
            prompt_ids=request.prompt_ids,
            prompt_attention_mask=request.prompt_attention_mask,
            responses=responses,
            pad_id=pad_id,
            n_samples_per_prompt=n,
        )

    @staticmethod
    def _extract_token_ids(choice: Dict[str, Any]) -> List[int]:
        """Extract generated token ids from a vLLM completions choice.

        vLLM 0.22.0 returns ``token_ids: null`` by default.  We request
        ``logprobs: 1`` in :meth:`generate` and read the token ids from the
        logprobs structure.
        """
        raw = choice.get("token_ids")
        if raw is not None:
            return list(raw)

        logprobs_data = choice.get("logprobs")
        if logprobs_data is not None:
            token_ids = logprobs_data.get("token_ids")
            if token_ids is not None:
                return [int(t) for t in token_ids]

            tokens = logprobs_data.get("tokens")
            if tokens is not None:
                return list(range(len(tokens)))

        return []

    # ------------------------------------------------------------------
    # Weight sync (vLLM 0.22.0 RLHF API)
    # ------------------------------------------------------------------

    def sync_weights(self, step: int) -> None:
        self._ensure_server()

        if self.student_engine is None:
            return

        if not self._weight_transfer_inited and self.is_rank_zero:
            if self._wt_backend == "gdr":
                self._init_gdr_channel()
            self._weight_transfer_inited = True

        from deepspeed.runtime.zero import GatheredParameters

        params: List[Tuple[str, torch.Tensor]] = []
        model = self.student_engine.module
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            with GatheredParameters([param], modifier_rank=0):
                if self.is_rank_zero:
                    params.append((name, param.data.detach().clone()))

        if self.is_rank_zero:
            self._pause()
            if self._wt_backend == "gdr":
                self._update_weights_gdr(params)
            else:
                self._update_weights_http(params)
            self._resume()

        from deepspeed import comm as dist

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

    # -- GDR (NCCL) weight transfer ----------------------------------------

    def _init_gdr_channel(self) -> None:
        """Bootstrap the GDR weight-transfer channel.

        vLLM's ``init_weight_transfer_engine`` endpoint and the trainer-side
        ``StatelessProcessGroup.create()`` must rendezvous concurrently
        (both block until the other side connects).  We fire the HTTP call
        in a background thread.
        """
        master_addr = self._get_own_ip()
        master_port = _find_free_port()

        resp = requests.get(f"{self._base_url}/get_world_size", timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        vllm_world_size = resp.json()["world_size"]
        total_world_size = vllm_world_size + 1

        init_info = {
            "master_address": master_addr,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": total_world_size,
        }

        init_thread = threading.Thread(target=self._post,
                                       args=("/init_weight_transfer_engine", ),
                                       kwargs={"json": {
                                           "init_info": init_info
                                       }})
        init_thread.start()

        from vllm.distributed.utils import StatelessProcessGroup

        group = StatelessProcessGroup.create(host=master_addr, port=master_port, rank=0, world_size=total_world_size)
        init_thread.join(timeout=30)
        if init_thread.is_alive():
            raise TimeoutError("init_weight_transfer_engine did not complete within 30s")

        self._nccl_group = group
        logger.info("GDR weight-transfer channel initialised "
                    "(world_size=%d, vllm_workers=%d)", total_world_size, vllm_world_size)

    def _update_weights_gdr(self, params: List[Tuple[str, torch.Tensor]]) -> None:
        """Push all gathered parameters to vLLM via GPU-direct (NCCL) transfer.

        The flow mirrors vLLM's official ``rlhf_http_nccl.py`` example:

        1. ``POST /start_weight_update`` — tells vLLM to prepare for incoming
           weights.
        2. ``POST /update_weights`` (in a **background thread**) — sends the
           parameter metadata (names, dtypes, shapes).  The server-side handler
           blocks waiting for NCCL broadcast.
        3. Trainer broadcasts each tensor via ``StatelessProcessGroup``.
        4. ``POST /finish_weight_update`` — finalises the update.
        """
        names: List[str] = []
        dtype_names: List[str] = []
        shapes: List[List[int]] = []
        tensors: List[torch.Tensor] = []

        for name, tensor in params:
            names.append(name)
            dtype_names.append(str(tensor.dtype).replace("torch.", ""))
            shapes.append(list(tensor.shape))
            tensors.append(tensor)

        self._post("/start_weight_update", json={"is_checkpoint_format": True})

        update_info = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "packed": False,
        }

        update_thread = threading.Thread(target=self._post,
                                         args=("/update_weights", ),
                                         kwargs={"json": {
                                             "update_info": update_info
                                         }})
        update_thread.start()

        for tensor in tensors:
            self._nccl_group.broadcast(tensor.contiguous(), src=0)

        update_thread.join(timeout=60)
        if update_thread.is_alive():
            raise TimeoutError("update_weights HTTP call did not complete within 60s")

        self._post("/finish_weight_update", json={})
        logger.info("pushed %d parameters via GDR", len(names))

    # -- HTTP weight transfer -----------------------------------------------

    def _update_weights_http(self, params: List[Tuple[str, torch.Tensor]]) -> None:
        """Push all gathered parameters to vLLM via HTTP serialised transfer.

        Each parameter is sent individually: metadata (name, dtype, shape)
        goes in the JSON body alongside the tensor bytes (base64-encoded).
        """
        import base64

        self._post("/start_weight_update", json={"is_checkpoint_format": True})

        for name, tensor in params:
            arr = tensor.cpu().numpy()
            buf = arr.tobytes()
            encoded = base64.b64encode(buf).decode("ascii")
            self._post(
                "/update_weights",
                json={
                    "update_info": {
                        "names": [name],
                        "dtype_names": [str(tensor.dtype).replace("torch.", "")],
                        "shapes": [list(tensor.shape)],
                        "packed": False,
                    },
                    "tensors": [encoded],
                },
                timeout=max(_HTTP_TIMEOUT, 30),
            )

        self._post("/finish_weight_update", json={})
        logger.info("pushed %d parameters via HTTP", len(params))

    # -- RLHF HTTP helpers -----------------------------------------------

    def _post(self, path: str, **kwargs: Any) -> requests.Response:
        resp = requests.post(f"{self._base_url}{path}", timeout=_HTTP_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp

    def _pause(self) -> None:
        self._post("/pause", params={"mode": "abort"})

    def _resume(self) -> None:
        self._post("/resume")

    @staticmethod
    def _get_own_ip() -> str:
        return "127.0.0.1"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._server_proc is not None:
            self._server_proc.send_signal(signal.SIGTERM)
            try:
                self._server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
                self._server_proc.wait()
            self._server_proc = None
        self._ready = False


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
