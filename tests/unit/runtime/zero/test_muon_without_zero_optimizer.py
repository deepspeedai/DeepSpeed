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


class TestMuonRunsWithoutAZeroOptimizer(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("dtype", ["fp32", "bf16", "fp16"])
    def test_newton_schulz_runs_at_stage_zero(self, dtype):
        """Every stage-0 wrapper: unwrapped for fp32, FP16_UnfusedOptimizer for bf16 and fp16.

        Each hands `step` the weight itself rather than a flat partition, so nothing upstream has
        orthogonalized it. On master all three do zero orthogonalizations and train as SGD.
        """
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
