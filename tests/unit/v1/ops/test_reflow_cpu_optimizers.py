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
from deepspeed.runtime.reflow.reflow_cpu_adam import ReflowCPUAdam
from deepspeed.runtime.reflow.reflow_cpu_lion import ReflowCPULion

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


@pytest.mark.parametrize("optimizer_type", [ReflowCPUAdam, ReflowCPULion])
def test_reflow_optimizer_rejects_low_precision_states(optimizer_type):
    param = torch.nn.Parameter(torch.ones(22, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="requires FP32 optimizer states"):
        optimizer_type([param], fp32_optimizer_states=False)


@pytest.mark.parametrize("optimizer_type, builder", [(ReflowCPUAdam, CPUAdamBuilder), (ReflowCPULion, CPULionBuilder)])
def test_reflow_optimizer_rejects_direct_step(optimizer_type, builder):
    if not deepspeed.ops.__compatible_ops__.get(builder.NAME, False):
        pytest.skip("the CPU optimizer builder is not supported on this accelerator")
    _skip_without_avx(builder)
    param = torch.nn.Parameter(torch.ones(22, dtype=torch.bfloat16))
    param.grad = torch.ones_like(param)
    optimizer = optimizer_type([param], num_threads=1)
    original = param.detach().clone()
    with pytest.raises(NotImplementedError, match="use engine.step"):
        optimizer.step()
    assert torch.equal(param, original)
    assert not optimizer.state


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


@pytest.mark.parametrize("optimizer_kind", ["adam", "lion"])
def test_reused_kernel_threads_follow_the_worker_cpu_mask(optimizer_kind):
    import os
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    if not hasattr(os, "sched_getaffinity") or not Path("/proc/self/task").exists():
        pytest.skip("requires Linux thread affinity")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 4:
        pytest.skip("requires four available CPUs")
    builder = CPUAdamBuilder if optimizer_kind == "adam" else CPULionBuilder
    if not deepspeed.ops.__compatible_ops__[builder.NAME]:
        pytest.skip("CPU optimizer builder is not compatible")
    _skip_without_avx(builder)
    module = builder().load()
    opt_id = _next_opt_id()
    param = torch.zeros(4096)
    grad = torch.ones(4096, dtype=torch.bfloat16)
    momentum = torch.zeros_like(param)
    variance = torch.zeros_like(param)
    half_param = torch.zeros_like(grad)
    common = [opt_id, 1, 1e-3, 0.9, 0.99]
    if optimizer_kind == "adam":
        module.reflow_create_adam(opt_id, 1e-3, 0.9, 0.99, 1e-8, 0.0, True, False, -1)
    else:
        module.reflow_create_lion(opt_id, 1e-3, 0.9, 0.99, 0.0, False, -1)
    helpers = set()

    def run(mask, phase):
        os.sched_setaffinity(0, mask)
        before = {int(path.name) for path in Path("/proc/self/task").iterdir()}
        if optimizer_kind == "adam":
            args = common + [1e-8, 0.0, True, param, grad, momentum, variance]
            update = getattr(module, f"reflow_adam_update_{phase}_halfgrad")
        else:
            args = common + [0.0, param, grad, momentum]
            update = getattr(module, f"reflow_lion_update_{phase}_halfgrad")
        if phase == "params":
            args.append(half_param)
        update(*args, 1.0, False)
        after = {int(path.name) for path in Path("/proc/self/task").iterdir()}
        helpers.update(after - before)
        # Pin the fixed regression: retained OpenMP helpers used to keep the previous task's
        # CPU mask when this same worker moved. Observe their OS affinity after kernel completion.
        for tid in helpers & after:
            assert set(os.sched_getaffinity(tid)) <= set(mask)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            for phase, mask in [("params", cpus[:2]), ("state", cpus[2:4]), ("params", cpus[:2]),
                                ("state", cpus[2:4])]:
                executor.submit(run, mask, phase).result()
    finally:
        destroy = getattr(module, f"reflow_destroy_{optimizer_kind}")
        destroy(opt_id)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half], ids=["bf16", "fp16"])
@pytest.mark.parametrize("size", [22, 4099])
def test_accumulation_matches_addition_on_restricted_worker_cpus(dtype, size):
    import os
    from concurrent.futures import ThreadPoolExecutor

    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("requires Linux thread affinity")
    if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
        pytest.skip("CPUAdamBuilder is not compatible")
    _skip_without_avx(CPUAdamBuilder)
    _skip_amd_fp16(dtype)
    module = CPUAdamBuilder().load()
    cpus = sorted(os.sched_getaffinity(0))
    generator = torch.Generator().manual_seed(1234)
    dst = torch.randn(size, generator=generator).to(dtype)
    src = torch.randn(size, generator=generator).to(dtype)
    expected = (dst.float() + src.float()).to(dtype)

    def accumulate():
        os.sched_setaffinity(0, cpus[:1])
        module.reflow_bf16_accumulate(dst, src)

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(accumulate).result()
    # A restricted OpenMP team must still finish every SIMD chunk and the non-vectorized tail.
    assert torch.equal(dst, expected)


@pytest.mark.parametrize('adamw_mode', [False, True], ids=['adam', 'adamw'])
@pytest.mark.parametrize('model_size', [7, 129, 4099])
@pytest.mark.parametrize('combined_scale', [1.0, 8.0])
def test_adam_maximize_groups_match_pytorch(adamw_mode, model_size, combined_scale):
    # Catch dropped per-group directions, incorrect decay, scalar-tail omissions, and a
    # deferred commit that reads changed live flags instead of the submitted step's snapshot.
    if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
        pytest.skip('CPUAdamBuilder is not compatible')
    _skip_without_avx(CPUAdamBuilder)
    generator = torch.Generator().manual_seed(4321)
    params = [torch.nn.Parameter(torch.randn(model_size, generator=generator)) for _ in range(2)]
    references = [torch.nn.Parameter(p.detach().clone()) for p in params]
    groups = [{'params': [params[0]]}, {'params': [params[1]], 'maximize': False}]
    reference_groups = [{'params': [references[0]]}, {'params': [references[1]], 'maximize': False}]
    kwargs = dict(lr=0.05, betas=(0.8, 0.95), eps=1e-5, weight_decay=0.1, maximize=True)
    optimizer = ReflowCPUAdam(groups, adamw_mode=adamw_mode, num_threads=1, **kwargs)
    reference_type = torch.optim.AdamW if adamw_mode else torch.optim.Adam
    reference = reference_type(reference_groups, foreach=False, **kwargs)
    module = optimizer.ds_opt_adam
    for p in params:
        optimizer.state[p].update(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))

    for step in range(1, 6):
        grads = [torch.randn(model_size, generator=generator).bfloat16() for _ in params]
        for p, grad in zip(references, grads):
            p.grad = grad.float() / combined_scale
        reference.step()
        snapshots = {
            i: {
                k: v
                for k, v in group.items() if k != 'params'
            }
            for i, group in enumerate(optimizer.param_groups)
        }
        for i, p in enumerate(params):
            state = optimizer.state[p]
            half_params = torch.empty(model_size, dtype=torch.bfloat16)
            before = p.detach().clone()
            before_momentum = state['exp_avg'].clone()
            before_variance = state['exp_avg_sq'].clone()
            group = snapshots[i]
            module.reflow_adam_update_params_halfgrad(optimizer.opt_id,
                                                      step,
                                                      group['lr'],
                                                      *group['betas'],
                                                      group['eps'],
                                                      group['weight_decay'],
                                                      group['bias_correction'],
                                                      p,
                                                      grads[i],
                                                      state['exp_avg'],
                                                      state['exp_avg_sq'],
                                                      half_params,
                                                      combined_scale,
                                                      maximize=group['maximize'])
            optimizer.memory_fence()
            torch.testing.assert_close(half_params.float(),
                                       references[i].detach().bfloat16().float(),
                                       rtol=0.008,
                                       atol=1e-5)
            assert torch.equal(p, before)
            assert torch.equal(state['exp_avg'], before_momentum)
            assert torch.equal(state['exp_avg_sq'], before_variance)
            state['step'] = step
            optimizer.param_groups[i]['maximize'] = not group['maximize']
        optimizer.step_state_halfgrad(grads, combined_scale=combined_scale, group_hyperparams=snapshots)
        optimizer.memory_fence()
        for i, p in enumerate(params):
            optimizer.param_groups[i]['maximize'] = snapshots[i]['maximize']
            state = optimizer.state[p]
            expected = reference.state[references[i]]
            torch.testing.assert_close(p, references[i], rtol=2e-6, atol=2e-7)
            torch.testing.assert_close(state['exp_avg'], expected['exp_avg'], rtol=2e-6, atol=2e-7)
            torch.testing.assert_close(state['exp_avg_sq'], expected['exp_avg_sq'], rtol=2e-6, atol=2e-7)
