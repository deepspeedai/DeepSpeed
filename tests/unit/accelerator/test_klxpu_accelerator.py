# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

try:
    import torch_xmlir  # noqa: F401 # type: ignore
except ImportError:
    pytest.skip("KLXPU accelerator requires torch_xmlir (KUNLUNXIN XPU stack)", allow_module_level=True)

from deepspeed.accelerator.klxpu_accelerator import KLXPU_Accelerator


def test_identity_is_klxpu_but_device_is_cuda():
    # KLXPU identity is 'klxpu' but torch device strings must stay 'cuda'.
    accelerator = KLXPU_Accelerator()
    assert accelerator._name == "klxpu"
    assert accelerator.device_name() == "cuda"
    assert accelerator.device_name(0) == "cuda:0"
    torch.device(accelerator.device_name(0))


def test_xpu_specific_runtime_overrides():
    accelerator = KLXPU_Accelerator()
    assert accelerator.is_triton_supported() is False
    assert accelerator.prefer_triton_grouped_mm() is False
    assert accelerator.supports_nvtx_domain is False
    assert accelerator._get_nvtx_domain("anything") is None


def test_host_timers_default_on_and_event_timer_opt_in(monkeypatch):
    # KLXPU event timers are stubs by default, so host timers are used unless
    # XPU_EVENT_KL3_ENABLE=1 opts into functional device-event timing.
    accelerator = KLXPU_Accelerator()

    monkeypatch.delenv("XPU_EVENT_KL3_ENABLE", raising=False)
    assert accelerator.use_host_timers() is True

    monkeypatch.setenv("XPU_EVENT_KL3_ENABLE", "0")
    assert accelerator.use_host_timers() is True

    monkeypatch.setenv("XPU_EVENT_KL3_ENABLE", "1")
    assert accelerator.use_host_timers() is False


def test_op_builder_resolves_from_klxpu_dir():
    accelerator = KLXPU_Accelerator()
    assert accelerator.op_builder_dir().endswith("op_builder.klxpu")
    fused_adam_builder = accelerator.get_op_builder("FusedAdamBuilder")
    assert fused_adam_builder is not None
    assert fused_adam_builder.__module__.endswith("klxpu.fused_adam")
    assert accelerator.get_op_builder("KLXPUOpBuilder") is None


def test_fused_adam_fallback_matches_torch_adamw(monkeypatch):
    # Check the pure-PyTorch fallback against torch.optim.AdamW as an oracle.
    from op_builder.klxpu import fused_adam as klxpu_fused_adam

    monkeypatch.setattr(klxpu_fused_adam, "_has_custom_ops", lambda: False)
    KLXPUFusedAdam = klxpu_fused_adam.KLXPUFusedAdam

    torch.manual_seed(0)
    lr, beta1, beta2, eps, weight_decay = 1e-2, 0.9, 0.999, 1e-8, 0.1

    param_ref = torch.randn(64, dtype=torch.float32)
    param_test = param_ref.clone()
    grads = [torch.randn(64, dtype=torch.float32) for _ in range(5)]

    optimizer = torch.optim.AdamW([param_ref], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)

    exp_avg = torch.zeros_like(param_test)
    exp_avg_sq = torch.zeros_like(param_test)
    noop_flag = torch.zeros(1, dtype=torch.int)

    for step, grad in enumerate(grads, start=1):
        param_ref.grad = grad.clone()
        optimizer.step()

        KLXPUFusedAdam.multi_tensor_adam(
            2048,
            noop_flag,
            [[grad.clone()], [param_test], [exp_avg], [exp_avg_sq]],
            lr,
            beta1,
            beta2,
            eps,
            step,
            True,  # adam_w_mode
            True,  # bias_correction
            weight_decay,
        )

    torch.testing.assert_close(param_test, param_ref, rtol=1e-5, atol=1e-6)
