# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""CPU-only unit tests for VLLMRollout (no GPU or vLLM server needed).

Tests cover configuration validation, command construction, token-id
extraction from API responses, and utility helpers.
"""

from unittest.mock import MagicMock, patch

import pytest

from deepspeed.runtime.rlhf.config import RolloutConfig


def _make_cfg(**overrides):
    defaults = dict(
        engine="vllm",
        vllm_port=8999,
        gpu_memory_utilization=0.3,
        weight_transfer_backend="http",
    )
    defaults.update(overrides)
    return RolloutConfig(**defaults)


# -- __init__ validation ------------------------------------------------


def test_init_rejects_wrong_engine():
    from deepspeed.runtime.rollout.vllm_rollout import VLLMRollout

    cfg = RolloutConfig(engine="hybrid_engine")
    with pytest.raises(ValueError, match="must be 'vllm'"):
        VLLMRollout(cfg=cfg, tokenizer=MagicMock(), student_model_path="x")


def test_gpus_from_env_var(monkeypatch):
    monkeypatch.setenv("ROLLOUT_VISIBLE_DEVICE", "6,7")
    cfg = RolloutConfig(engine="vllm")
    assert cfg.gpus == [6, 7]


def test_env_var_overrides_json_gpus(monkeypatch):
    monkeypatch.setenv("ROLLOUT_VISIBLE_DEVICE", "6,7")
    cfg = RolloutConfig(engine="vllm", gpus=[0, 1])
    assert cfg.gpus == [6, 7]


def test_no_env_var_keeps_json_gpus(monkeypatch):
    monkeypatch.delenv("ROLLOUT_VISIBLE_DEVICE", raising=False)
    cfg = RolloutConfig(engine="vllm", gpus=[0, 1])
    assert cfg.gpus == [0, 1]


def test_init_requires_student_model_path():
    from deepspeed.runtime.rollout.vllm_rollout import VLLMRollout

    cfg = RolloutConfig(engine="vllm")
    with pytest.raises(ValueError, match="student_model_path"):
        VLLMRollout(cfg=cfg, tokenizer=MagicMock())


def test_init_http_backend():
    from deepspeed.runtime.rollout.vllm_rollout import VLLMRollout

    cfg = _make_cfg(weight_transfer_backend="http")
    rollout = VLLMRollout(cfg=cfg, tokenizer=MagicMock(), student_model_path="test-model")
    assert rollout._wt_backend == "http"


# -- _extract_token_ids -------------------------------------------------


def test_extract_token_ids_prefers_token_ids():
    from deepspeed.runtime.rollout.vllm_rollout import VLLMRollout

    choice = {"token_ids": [10, 20, 30]}
    assert VLLMRollout._extract_token_ids(choice) == [10, 20, 30]


def test_extract_token_ids_from_logprobs_token_ids():
    from deepspeed.runtime.rollout.vllm_rollout import VLLMRollout

    choice = {"logprobs": {"token_ids": [5, 6, 7]}}
    assert VLLMRollout._extract_token_ids(choice) == [5, 6, 7]


def test_extract_token_ids_from_logprobs_tokens_fallback():
    from deepspeed.runtime.rollout.vllm_rollout import VLLMRollout

    choice = {"logprobs": {"tokens": ["a", "b"]}}
    assert VLLMRollout._extract_token_ids(choice) == [0, 1]


def test_extract_token_ids_empty_on_no_data():
    from deepspeed.runtime.rollout.vllm_rollout import VLLMRollout

    assert VLLMRollout._extract_token_ids({}) == []


# -- _start_server command construction --------------------------------


def test_start_server_command_http_backend(monkeypatch):
    from deepspeed.runtime.rollout.vllm_rollout import VLLMRollout

    monkeypatch.delenv("ROLLOUT_VISIBLE_DEVICE", raising=False)
    cfg = _make_cfg(
        weight_transfer_backend="http",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.5,
        vllm_port=12345,
        vllm_enforce_eager=True,
        gpus=[0, 1],
    )
    rollout = VLLMRollout(cfg=cfg, tokenizer=MagicMock(), student_model_path="test-model")

    with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        rollout._start_server()

    args, kwargs = mock_popen.call_args
    cmd = args[0]
    assert cmd[0].endswith("python") or "python" in cmd[0]
    assert "-m" in cmd
    assert "vllm.entrypoints.openai.api_server" in cmd
    assert "--model" in cmd
    assert "test-model" in cmd
    assert "--tensor-parallel-size" in cmd
    assert "2" in cmd
    assert "--gpu-memory-utilization" in cmd
    assert "0.5" in cmd
    assert "--port" in cmd
    assert "12345" in cmd
    assert "--enforce-eager" in cmd
    assert '{"backend": "http"}' in cmd

    env = kwargs["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["VLLM_SERVER_DEV_MODE"] == "1"


def test_start_server_uses_vllm_python():
    from deepspeed.runtime.rollout.vllm_rollout import VLLMRollout

    cfg = _make_cfg(vllm_python="/custom/bin/python")
    rollout = VLLMRollout(cfg=cfg, tokenizer=MagicMock(), student_model_path="test-model")

    with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        rollout._start_server()

    cmd = mock_popen.call_args[0][0]
    assert cmd[0] == "/custom/bin/python"


# -- _wait_for_health detects early exit --------------------------------


def test_wait_for_health_raises_on_crash():
    from deepspeed.runtime.rollout.vllm_rollout import VLLMRollout

    cfg = _make_cfg(vllm_start_timeout=5)
    rollout = VLLMRollout(cfg=cfg, tokenizer=MagicMock(), student_model_path="test-model")

    proc = MagicMock()
    proc.poll.return_value = 1
    proc.returncode = 1
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = b"some error detail"
    rollout._server_proc = proc

    with pytest.raises(RuntimeError, match="exited prematurely"):
        rollout._wait_for_health()


def test_wait_for_health_raises_timeout():
    from deepspeed.runtime.rollout.vllm_rollout import VLLMRollout

    cfg = _make_cfg(vllm_start_timeout=0)
    rollout = VLLMRollout(cfg=cfg, tokenizer=MagicMock(), student_model_path="test-model")
    rollout._server_proc = MagicMock()
    rollout._server_proc.poll.return_value = None

    with pytest.raises(TimeoutError, match="did not become healthy"):
        rollout._wait_for_health()


# -- utility helpers ----------------------------------------------------


def test_get_own_ip():
    from deepspeed.runtime.rollout.vllm_rollout import VLLMRollout

    assert isinstance(VLLMRollout._get_own_ip(), str)


def test_find_free_port():
    from deepspeed.runtime.rollout.vllm_rollout import _find_free_port

    port = _find_free_port()
    assert isinstance(port, int)
    assert 1 <= port <= 65535
