# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Newton-Schulz has to run whether or not a ZeRO optimizer is there to do it.

`MuonWithAuxAdam.step` applied an update it assumed had been orthogonalized already, which holds
under ZeRO - the parameters it sees there are flat partitions and `get_flat_partition` or the
ZeRO-3 sub-group loop did the work. At stage 0, the default, no ZeRO optimizer exists to have
done it, and applying a raw gradient is SGD. Training runs and the loss falls, which is why
counting the Newton-Schulz calls is the assertion that means something here.
"""

import contextlib

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.runtime.zero.muon.original_muon as original_muon
from unit.common import DistributedTest

NS_KERNELS = ("zeropower_via_gram_newtonschulz", "zeropower_via_newtonschulz5")


@contextlib.contextmanager
def counting_newton_schulz():
    """Counts every Newton-Schulz call, whichever kernel the config selects.

    Patched inside the test body rather than in a fixture: `DistributedTest` runs the body in a
    worker process that a fixture in the parent would not reach.
    """
    calls = []
    originals = {name: getattr(original_muon, name) for name in NS_KERNELS}

    def counted(kernel):

        def wrapper(*args, **kwargs):
            calls.append(1)
            return kernel(*args, **kwargs)

        return wrapper

    for name, kernel in originals.items():
        setattr(original_muon, name, counted(kernel))
    try:
        yield calls
    finally:
        for name, kernel in originals.items():
            setattr(original_muon, name, kernel)


def _model():
    return torch.nn.Sequential(torch.nn.Linear(32, 32, bias=False), torch.nn.Linear(32, 32, bias=False))


def _config(stage, dtype="fp32"):
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 0.0,
        "optimizer": {
            "type": "Muon",
            "params": {
                "lr": 0.02
            }
        },
    }
    if stage is not None:
        config["zero_optimization"] = {"stage": stage, "reduce_scatter": stage != 3}
    if dtype != "fp32":
        config[dtype] = {"enabled": True}
        if dtype == "fp16":
            config[dtype]["initial_scale_power"] = 4
    return config


def _skip_if_unsupported(dtype):
    """Mirror the check the engine itself makes.

    `_do_sanity_check` raises `Type fp16 is not supported on your device.` on
    `not get_accelerator().is_fp16_supported()`, which is a different predicate from
    `supported_dtypes()` -- the cpu-torch-latest runner reports fp16 in the latter and
    False from the former, so guarding on the wrong one still fails there.
    """
    supported = {
        "fp16": get_accelerator().is_fp16_supported,
        "bf16": get_accelerator().is_bf16_supported,
    }.get(dtype)
    if supported is not None and not supported():
        pytest.skip(f"{dtype} not supported on this accelerator")


class TestMuonRunsWithoutAZeroOptimizer(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("dtype", ["fp32", "bf16", "fp16"])
    def test_newton_schulz_runs_at_stage_zero(self, dtype):
        """Every stage-0 wrapper: unwrapped for fp32, FP16_UnfusedOptimizer for bf16 and fp16.

        Each hands `step` the weight itself rather than a flat partition, so nothing upstream has
        orthogonalized it. On master all three do zero orthogonalizations and train as SGD.
        """
        _skip_if_unsupported(dtype)
        model = _model()
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=_config(0, dtype))

        # Counting starts after initialize: FP16_UnfusedOptimizer steps once at construction to
        # allocate state, and that call must not be what the assertion below is satisfied by.
        with counting_newton_schulz() as calls:
            x = torch.ones(2, 32, device=engine.device, dtype=next(engine.module.parameters()).dtype)
            engine.backward(engine(x).square().sum())
            engine.step()

        assert len(calls) == 2, f"Newton-Schulz ran {len(calls)} times for two Muon matrices; expected one each"

    def test_the_default_config_runs_muon(self):
        """`zero_optimization.stage` defaults to 0, so this is the plainest Muon config there is."""
        model = _model()
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=_config(None))

        with counting_newton_schulz() as calls:
            x = torch.ones(2, 32, device=engine.device)
            engine.backward(engine(x).square().sum())
            engine.step()

        assert len(calls) == 2, f"Newton-Schulz ran {len(calls)} times; the default config trained as SGD"

    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_newton_schulz_runs_on_the_supported_stages(self, stage):
        """The positive control, and the assertion the existing tests were missing.

        They check that training progresses, which SGD does too. Counting the orthogonalizations
        is what distinguishes Muon from the update it degenerates to.
        """
        model = _model()
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=_config(stage))

        with counting_newton_schulz() as calls:
            x = torch.ones(2, 32, device=engine.device, dtype=next(engine.module.parameters()).dtype)
            engine.backward(engine(x).square().sum())
            engine.step()

        assert len(calls) == 2, \
            f"Newton-Schulz ran {len(calls)} times for two Muon matrices; the step was not Muon"


class TestMuonBF16Optimizer(DistributedTest):
    """BF16 accumulation boundaries must run NS before consuming flat updates."""
    world_size = [1, 2]

    @pytest.mark.parametrize("ns_method", ["standard", "gram"])
    def test_bf16_optimizer_runs_muon_after_accumulation(self, ns_method):
        _skip_if_unsupported("bf16")
        model = _model()
        config = _config(1, "bf16")
        config["data_types"] = {"grad_accum_dtype": "fp32"}
        config["gradient_accumulation_steps"] = 2
        config["optimizer"]["params"]["ns_method"] = ns_method
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        original = [p.detach().clone() for p in engine.module.parameters()]
        with counting_newton_schulz() as calls:
            x = torch.ones(2, 32, device=engine.device, dtype=torch.bfloat16)
            engine.backward(engine(x).square().sum())
            engine.step()
            assert not calls
            for param, before in zip(engine.module.parameters(), original):
                torch.testing.assert_close(param, before, rtol=0, atol=0)
            engine.backward(engine(x).square().sum())
            engine.step()
            # These equal matrices land on exact DP partition boundaries.
            assert len(calls) == 2 // deepspeed.comm.get_world_size()
        assert any(not torch.equal(p, old) for p, old in zip(engine.module.parameters(), original))
        for group in engine.optimizer.optimizer.param_groups:
            if group['use_muon']:
                partition = group['params'][0]
                momentum = engine.optimizer.optimizer.state[partition]['momentum_buffer']
                assert momentum.shape == partition.shape
                assert momentum.dtype == torch.float32
                assert torch.isfinite(momentum).all() and torch.count_nonzero(momentum)

    def test_the_same_config_without_grad_accum_dtype_still_runs_muon(self):
        """The existing BF16-gradient wrapper remains supported."""
        _skip_if_unsupported("bf16")
        model = _model()
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=_config(1, "bf16"))

        with counting_newton_schulz() as calls:
            x = torch.ones(2, 32, device=engine.device, dtype=next(engine.module.parameters()).dtype)
            engine.backward(engine(x).square().sum())
            engine.step()

        assert len(calls) == 2 // deepspeed.comm.get_world_size()
