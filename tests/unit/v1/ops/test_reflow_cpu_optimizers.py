# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Drop-in equivalence tests for the Reflow CPU-Adam and CPU-Lion kernels (the async CPU-offload path).

Reflow splits one optimizer step into two passes over the same tensors:
  * Phase-1 (``reflow_*_update_params_halfgrad``): from the FP32 master, the half-precision grad and the
    optimizer state, produce ONLY the new BF16/FP16 weights into a separate buffer. The FP32 master and
    the state are left untouched (deferred).
  * Phase-2 (``reflow_*_update_state_halfgrad``): commit the FP32 master and the state.

The reference is the base ``DeepSpeedCPUAdam`` / ``DeepSpeedCPULion`` step, so these prove the Reflow
kernels are a faithful drop-in. They run purely on CPU; no GPU/distributed env is required.
"""

import pytest
import torch
from cpuinfo import get_cpu_info

import deepspeed
from deepspeed.ops.op_builder import CPUAdamBuilder, CPULionBuilder

pytest.cpu_vendor = get_cpu_info()["vendor_id_raw"].lower() if "vendor_id_raw" in get_cpu_info() else "unknown"

# One shared opt_id counter so each parametrized case uses an isolated native reflow optimizer.
_opt_id = 0


def _next_opt_id():
    global _opt_id
    _opt_id += 1
    return _opt_id


def _skip_amd_fp16(grad_dtype):
    if ("amd" in pytest.cpu_vendor) and (grad_dtype == torch.half):
        pytest.skip("cpu optimizers with half precision not supported on AMD CPUs")


def _skip_without_avx(builder):
    # The Reflow kernels exist only for AVX2/AVX-512; on other builds creating the optimizer raises.
    if builder().simd_width() not in ("-D__AVX512__", "-D__AVX256__"):
        pytest.skip("the CPU optimizer extension is not built with AVX2 or AVX-512")


def test_every_builder_of_the_shared_bindings_compiles_the_reflow_kernels():
    """The Reflow entry points are bound in the shared cpu_adam.cpp / cpu_lion.cpp, so a builder that compiles those
    without the Reflow implementation produces an extension that fails to import with an undefined reflow_* symbol.
    The accelerator-specific builders import accelerator packages, so read their source lists instead of loading them.
    """
    import ast
    import pathlib

    import op_builder

    bindings = {
        "csrc/adam/cpu_adam.cpp": "csrc/adam/reflow_cpu_adam_impl.cpp",
        "csrc/lion/cpu_lion.cpp": "csrc/lion/reflow_cpu_lion_impl.cpp",
    }
    missing = []
    checked = 0
    for path in sorted(pathlib.Path(op_builder.__file__).parent.rglob("*.py")):
        sources = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name == "sources":
                sources += [
                    sub.value for sub in ast.walk(node)
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
                ]
        for binding, reflow_impl in bindings.items():
            if binding in sources:
                checked += 1
                reflow_bindings = reflow_impl.replace("_impl.cpp", "_bindings.cpp")
                for required in (reflow_impl, reflow_bindings):
                    if required not in sources:
                        missing.append(f"{path.name} compiles {binding} without {required}")
    assert checked > 0, "no builder compiles the shared bindings; the paths above are stale"
    assert not missing, missing


@pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME],
                    reason="CPUAdamBuilder is not compatible on this system.")
@pytest.mark.parametrize('model_size', [22, 128, 1000, 1024, 4096])
@pytest.mark.parametrize('grad_dtype', [torch.bfloat16, torch.half], ids=["bf16grad", "fp16grad"])
@pytest.mark.parametrize('adamw_mode, weight_decay', [(True, 0.0), (True, 0.01), (False, 0.01)],
                         ids=["adamw_wd0", "adamw_wd0.01", "adam_wd0.01"])
class TestReflowCPUAdamDropIn:

    LR, BETA1, BETA2, EPS = 1e-3, 0.9, 0.999, 1e-8
    STEPS = 3

    def test_two_phase_step_matches_base(self, model_size, grad_dtype, adamw_mode, weight_decay):
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        _skip_amd_fp16(grad_dtype)
        _skip_without_avx(CPUAdamBuilder)
        lr, b1, b2, eps = self.LR, self.BETA1, self.BETA2, self.EPS
        torch.manual_seed(1234 + model_size)
        param_fp32 = torch.randn(model_size, dtype=torch.float)
        exp_avg = torch.zeros(model_size, dtype=torch.float)
        exp_avg_sq = torch.zeros(model_size, dtype=torch.float)
        half_params = torch.zeros(model_size, dtype=grad_dtype)

        base_param = torch.nn.Parameter(param_fp32.clone())
        base_opt = DeepSpeedCPUAdam([base_param],
                                    lr=lr,
                                    betas=(b1, b2),
                                    eps=eps,
                                    weight_decay=weight_decay,
                                    adamw_mode=adamw_mode)

        module = CPUAdamBuilder().load()
        opt_id = _next_opt_id()
        module.reflow_create_adam(opt_id, lr, b1, b2, eps, weight_decay, adamw_mode, False, -1)
        # Several steps, so bias correction and non-zero moments are exercised.
        for step in range(1, self.STEPS + 1):
            grad_half = torch.randn(model_size, dtype=torch.float).to(grad_dtype)
            # The reflow kernel promotes the half grad to FP32 exactly, so the base step gets the promoted grad.
            base_param.grad = grad_half.float()
            base_opt.step()
            base_state = base_opt.state[base_param]

            param_before = param_fp32.clone()
            module.reflow_adam_update_params_halfgrad(opt_id, step, lr, b1, b2, eps, weight_decay, True, param_fp32,
                                                      grad_half, exp_avg, exp_avg_sq, half_params, 1.0, False)
            module.reflow_adam_memory_fence()
            # Phase-1 emits the base optimizer's new weights in half precision and defers the FP32 master.
            assert torch.equal(half_params, base_param.data.to(grad_dtype)), f"half params differ at step {step}"
            assert torch.equal(param_fp32, param_before), f"Phase-1 changed the FP32 master at step {step}"

            module.reflow_adam_update_state_halfgrad(opt_id, step, lr, b1, b2, eps, weight_decay, True, param_fp32,
                                                     grad_half, exp_avg, exp_avg_sq, 1.0, False)
            module.reflow_adam_memory_fence()
            assert torch.equal(param_fp32, base_param.data), f"FP32 master differs at step {step}"
            assert torch.equal(exp_avg, base_state['exp_avg']), f"exp_avg differs at step {step}"
            assert torch.equal(exp_avg_sq, base_state['exp_avg_sq']), f"exp_avg_sq differs at step {step}"
        module.reflow_destroy_adam(opt_id)


def _base_lion_step(param_fp32, exp_avg, grad_fp32, lr, beta1, beta2, weight_decay):
    """One step of the base DeepSpeedCPULion, seeded with the given momentum, as the reference."""
    from deepspeed.ops.lion import DeepSpeedCPULion
    p = torch.nn.Parameter(param_fp32.clone())
    opt = DeepSpeedCPULion([p], lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    # Seed a non-zero starting momentum (skips the lazy zero-init in step()) so the update direction
    # c_t is exercised, not just the gradient.
    state = opt.state[p]
    state['step'] = 0
    state['exp_avg'] = exp_avg.clone()
    p.grad = grad_fp32.clone()
    opt.step()
    return p.data.detach().clone(), opt.state[p]['exp_avg'].detach().clone()


@pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[CPULionBuilder.NAME],
                    reason="CPULionBuilder is not compatible on this system.")
@pytest.mark.parametrize('model_size', [22, 128, 1000, 1024, 4096])
@pytest.mark.parametrize('grad_dtype', [torch.bfloat16, torch.half], ids=["bf16grad", "fp16grad"])
@pytest.mark.parametrize('weight_decay', [0.0, 0.01], ids=["wd0", "wd0.01"])
class TestReflowCPULionDropIn:

    LR, BETA1, BETA2 = 1e-3, 0.9, 0.99

    def _inputs(self, model_size, grad_dtype):
        torch.manual_seed(1234 + model_size)
        param_fp32 = torch.randn(model_size, dtype=torch.float)
        exp_avg = torch.randn(model_size, dtype=torch.float)
        grad_half = torch.randn(model_size, dtype=torch.float).to(grad_dtype)
        return param_fp32, exp_avg, grad_half

    def test_phase2_halfgrad_matches_base(self, model_size, grad_dtype, weight_decay):
        _skip_amd_fp16(grad_dtype)
        _skip_without_avx(CPULionBuilder)
        lr, b1, b2 = self.LR, self.BETA1, self.BETA2
        param_fp32, exp_avg, grad_half = self._inputs(model_size, grad_dtype)
        # The reflow kernel promotes the half grad to FP32 exactly, so the base reference gets the same
        # promoted gradient.
        param_ref, exp_avg_ref = _base_lion_step(param_fp32, exp_avg, grad_half.float(), lr, b1, b2, weight_decay)

        module = CPULionBuilder().load()
        opt_id = _next_opt_id()
        module.reflow_create_lion(opt_id, lr, b1, b2, weight_decay, False, -1)
        module.reflow_lion_update_state_halfgrad(opt_id, 1, lr, b1, b2, weight_decay, param_fp32, grad_half, exp_avg,
                                                 1.0, False)
        module.reflow_lion_memory_fence()
        module.reflow_destroy_lion(opt_id)

        torch.testing.assert_close(param_fp32, param_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(exp_avg, exp_avg_ref, atol=1e-4, rtol=1e-4)

    def test_phase1_produces_bf16_and_defers(self, model_size, grad_dtype, weight_decay):
        _skip_amd_fp16(grad_dtype)
        _skip_without_avx(CPULionBuilder)
        lr, b1, b2 = self.LR, self.BETA1, self.BETA2
        param_fp32, exp_avg, grad_half = self._inputs(model_size, grad_dtype)
        param_before = param_fp32.clone()
        exp_avg_before = exp_avg.clone()

        param_ref, _ = _base_lion_step(param_fp32, exp_avg, grad_half.float(), lr, b1, b2, weight_decay)
        half_params = torch.zeros(model_size, dtype=grad_dtype)

        module = CPULionBuilder().load()
        opt_id = _next_opt_id()
        module.reflow_create_lion(opt_id, lr, b1, b2, weight_decay, False, -1)
        module.reflow_lion_update_params_halfgrad(opt_id, 1, lr, b1, b2, weight_decay, param_fp32, grad_half, exp_avg,
                                                  half_params, 1.0, False)
        module.reflow_lion_memory_fence()
        module.reflow_destroy_lion(opt_id)

        # Phase-1 emits the base optimizer's new weights into the half buffer (BF16/FP16 rounding)...
        torch.testing.assert_close(half_params.float(), param_ref, atol=2e-2, rtol=2e-2)
        # ...and leaves the FP32 master and momentum untouched (the deferred writeback).
        torch.testing.assert_close(param_fp32, param_before, atol=0, rtol=0)
        torch.testing.assert_close(exp_avg, exp_avg_before, atol=0, rtol=0)
