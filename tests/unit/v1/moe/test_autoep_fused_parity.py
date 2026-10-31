# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""End-to-end parity for the eager and fused combine implementations."""

import functools

import deepspeed
import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.ops.triton_ops import autoep_fused_token_ops as fused_ops
from deepspeed.utils import safe_get_full_grad
from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import (
    MockMoETransformer,
    engine_input_dtype,
    mixed_precision_config,
    seed_everything,
)

HIDDEN_SIZE = 64
SEQ_LEN = 16
NUM_EXPERTS = 4

# The top-k accumulation order differs, so parity allows last-bit noise only.
PARITY_TOLERANCE = {"rtol": 1e-2, "atol": 1e-3}
# How much further than eager the fused input gradient may sit from the fp32 reference. Pure
# reassociation moves the ratio by at most 0.3%; a 0.1% error in one expert's rows moves it by 6%.
INPUT_GRAD_ERROR_ALLOWANCE = 0.02
# The same, per parameter gradient. Reassociation moves the worst parameter by up to 2.5%; a 0.1%
# error in one top-k slot's score gradient moves it by at least 23%.
GRADIENT_ERROR_ALLOWANCE = 0.1
# Updates are compared only where the fp32 reference gradient exceeds this fraction of its
# parameter's largest reference gradient. The noisiest parameter's eager-vs-reference error is
# 0.0044 of that maximum, so 0.02 keeps every compared element at least 4.5 times that error.
# Genuine runs then agree exactly on the kept elements. A lower cut (0.01) leaves a 2.3x margin.
UPDATE_MASK_RELATIVE_GRADIENT = 0.02
# The masked comparison must still cover at least this fraction of all update elements. Across
# the seed sweep the thinnest case kept 57% at the threshold above.
UPDATE_MASK_MIN_FRACTION = 0.5


def _fused_engine_available():
    accelerator = get_accelerator()
    return (accelerator.is_available() and accelerator.device_name().startswith("cuda") and fused_ops.is_available())


pytestmark = pytest.mark.skipif(not _fused_engine_available(),
                                reason="the fused weighted restore needs CUDA and Triton")


def _config(combine_impl, ep_size, mixed_precision=True):
    return {
        **(mixed_precision_config() if mixed_precision else {}),
        "train_micro_batch_size_per_gpu": 1,
        "gradient_clipping": 0.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-3,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        },
        "expert_parallel": {
            "enabled": True,
            "autoep_size": ep_size,
            "preset_model": "mixtral",
            "load_balance_coeff": None,
            "combine_impl": combine_impl,
        },
    }


def _build_engine(combine_impl, ep_size, reference_state, seed, mixed_precision=True):
    seed_everything(seed)
    model = MockMoETransformer(num_layers=2,
                               num_experts=NUM_EXPERTS,
                               hidden_size=HIDDEN_SIZE,
                               intermediate_size=2 * HIDDEN_SIZE)
    model.load_state_dict(reference_state)
    engine, _, _, _ = deepspeed.initialize(model=model,
                                           config=_config(combine_impl, ep_size, mixed_precision=mixed_precision))
    return engine


def _fp32_reference_step(ep_size, reference_state, seed, *, checkpoint_activations):
    """An fp32 eager step, independent of the kernel under test."""
    matmul_precision = torch.get_float32_matmul_precision()
    # TF32 matmuls would make the reference less precise than the paths it judges.
    torch.set_float32_matmul_precision("highest")
    try:
        engine = _build_engine("weighted_sum", ep_size, reference_state, seed, mixed_precision=False)
        return _take_one_step(engine, seed, checkpoint_activations=checkpoint_activations)
    finally:
        torch.set_float32_matmul_precision(matmul_precision)


def _checkpoint_moe_layers(engine):
    """Recompute each MoE block in backward, as the benchmarked runs do."""
    for module in engine.module.modules():
        if isinstance(module, AutoEPMoELayer):
            module.forward = functools.partial(torch.utils.checkpoint.checkpoint, module.forward, use_reentrant=False)


def _named_gradients(engine):
    gradients = {}
    for name, param in engine.module.named_parameters():
        if not param.requires_grad:
            continue
        grad = safe_get_full_grad(param)
        if grad is not None:
            gradients[name] = grad.detach().float().cpu().clone()
    return gradients


def _parameters(engine):
    return {
        name: param.detach().float().cpu().clone()
        for name, param in engine.module.named_parameters() if param.requires_grad
    }


def _take_one_step(engine, seed, *, checkpoint_activations):
    if checkpoint_activations:
        _checkpoint_moe_layers(engine)

    generator = torch.Generator().manual_seed(seed)
    batch = torch.randn((1, SEQ_LEN, HIDDEN_SIZE), generator=generator, dtype=torch.float32)
    batch = batch.to(engine.device, dtype=engine_input_dtype(engine)).requires_grad_(True)

    before = _parameters(engine)
    output = engine(batch)
    loss = output.float().pow(2).mean()
    engine.backward(loss)

    gradients = _named_gradients(engine)
    input_grad = batch.grad.detach().float().cpu().clone()
    engine.step()

    delta = {name: _parameters(engine)[name] - value for name, value in before.items()}
    return {
        "loss": loss.detach().float().cpu().clone(),
        "output": output.detach().float().cpu().clone(),
        "input_grad": input_grad,
        "gradients": gradients,
        "delta": delta,
    }


def _assert_no_further_from_reference(fused, eager, reference, allowance, what):
    assert reference.abs().max() > 0, f"the fp32 reference for {what} is zero, so the comparison would be vacuous"
    fused_error = torch.linalg.vector_norm(fused - reference).item()
    eager_error = torch.linalg.vector_norm(eager - reference).item()
    assert fused_error <= eager_error * (1 + allowance), (
        f"the fused {what} is further from the fp32 reference than eager: "
        f"{fused_error:.6g} vs {eager_error:.6g} (allowance {allowance})")


def _assert_input_grad_no_worse_than_eager(fused, eager, reference):
    # The input gradient passes back through every fp16 layer, so a last-bit difference in the
    # combine grows to one fp16 ulp of its largest entries (2 at magnitude 2000). A fixed atol of
    # 1e-3 is below that resolution, so judge each path by its distance from an fp32 reference.
    _assert_no_further_from_reference(fused["input_grad"], eager["input_grad"], reference["input_grad"],
                                      INPUT_GRAD_ERROR_ALLOWANCE, "input gradient")


def _assert_gradients_no_worse_than_eager(fused, eager, reference):
    # Parameter gradients carry the same amplified reassociation noise as the input gradient. Each
    # parameter is judged on its own so that one bad parameter cannot hide in a global norm.
    for name in sorted(eager["gradients"]):
        _assert_no_further_from_reference(fused["gradients"][name], eager["gradients"][name],
                                          reference["gradients"][name], GRADIENT_ERROR_ALLOWANCE,
                                          f"gradient of {name}")


def _update_mask(reference_gradient):
    magnitude = reference_gradient.abs()
    return magnitude > UPDATE_MASK_RELATIVE_GRADIENT * magnitude.max()


def _assert_resolvable_updates_match(fused, eager, reference):
    # The first Adam step moves each weight by about lr * sign(grad), since m / sqrt(v) with bias
    # correction reduces to the unit sign. The update carries no magnitude information, and its
    # sign is decided by noise wherever the gradient is near zero, so those elements cannot be
    # compared at any tolerance. Compare only where the fp32 reference gradient, which the paths
    # under test cannot perturb, is large enough for the sign to be well determined.
    # This check does not catch faults the per-parameter gradient check misses: a 0.1% score-slot
    # error and a 0.1% expert-row error move gradients by 23% or more and leave these signs
    # unchanged. It only confirms the step's sign where that sign is resolvable.
    compared = total = 0
    for name in sorted(eager["delta"]):
        mask = _update_mask(reference["gradients"][name])
        compared += int(mask.sum())
        total += mask.numel()
        torch.testing.assert_close(fused["delta"][name][mask],
                                   eager["delta"][name][mask],
                                   msg=lambda formatted, name=name: f"parameter update mismatch for {name}\n"
                                   f"{formatted}",
                                   **PARITY_TOLERANCE)
    assert compared >= UPDATE_MASK_MIN_FRACTION * total, (
        f"only {compared} of {total} update elements have a resolvable gradient, so the comparison is too thin")


def _assert_step_matches(fused, eager, reference):
    torch.testing.assert_close(fused["loss"], eager["loss"], **PARITY_TOLERANCE)
    torch.testing.assert_close(fused["output"], eager["output"], **PARITY_TOLERANCE)
    _assert_input_grad_no_worse_than_eager(fused, eager, reference)

    assert fused["gradients"], "no gradients were captured, so the comparison would be vacuous"
    assert set(fused["gradients"]) == set(eager["gradients"]) == set(reference["gradients"])
    # Ensure both sides of the fused restore reached the comparison.
    assert any(".router." in name for name in fused["gradients"]), "no router gradient was captured"
    assert any(".experts.w" in name for name in fused["gradients"]), "no expert gradient was captured"

    _assert_gradients_no_worse_than_eager(fused, eager, reference)
    _assert_resolvable_updates_match(fused, eager, reference)

    assert any(value.abs().sum() > 0 for value in eager["delta"].values()), "the optimizer step changed nothing"


class TestAutoEPFusedParityExpertParallel(DistributedTest):
    world_size = 2

    # Several seeds, since a single one decides by luck whether reassociation noise trips a check.
    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 4321])
    @pytest.mark.parametrize("checkpoint_activations", [True, False])
    def test_fused_matches_eager_through_a_full_step(self, checkpoint_activations, seed):
        seed_everything(seed)
        reference_state = MockMoETransformer(num_layers=2,
                                             num_experts=NUM_EXPERTS,
                                             hidden_size=HIDDEN_SIZE,
                                             intermediate_size=2 * HIDDEN_SIZE).state_dict()

        eager_engine = _build_engine("weighted_sum", 2, reference_state, seed)
        eager = _take_one_step(eager_engine, seed, checkpoint_activations=checkpoint_activations)

        fused_engine = _build_engine("fused_weighted_sum", 2, reference_state, seed)
        assert all(module.combine_impl == "fused_weighted_sum" for module in fused_engine.module.modules()
                   if isinstance(module, AutoEPMoELayer)), "the fused reduction was not actually selected"
        fused = _take_one_step(fused_engine, seed, checkpoint_activations=checkpoint_activations)

        reference = _fp32_reference_step(2, reference_state, seed, checkpoint_activations=checkpoint_activations)
        _assert_step_matches(fused, eager, reference)


class TestAutoEPFusedParityLocalExperts(DistributedTest):
    world_size = 1

    def test_fused_matches_eager_without_expert_parallelism(self):
        seed = 8765
        seed_everything(seed)
        reference_state = MockMoETransformer(num_layers=2,
                                             num_experts=NUM_EXPERTS,
                                             hidden_size=HIDDEN_SIZE,
                                             intermediate_size=2 * HIDDEN_SIZE).state_dict()

        eager = _take_one_step(_build_engine("weighted_sum", 1, reference_state, seed),
                               seed,
                               checkpoint_activations=False)
        fused = _take_one_step(_build_engine("fused_weighted_sum", 1, reference_state, seed),
                               seed,
                               checkpoint_activations=False)

        reference = _fp32_reference_step(1, reference_state, seed, checkpoint_activations=False)
        _assert_step_matches(fused, eager, reference)
