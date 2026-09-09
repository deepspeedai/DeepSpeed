# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""ZenFlow's select interval, and what it takes for re-selection to happen.

ZenFlow re-picks the important gradient columns every ``select_interval``
micro-steps. The 'epoch' strategy -- which 'auto', the default, resolves to --
expresses that interval in epochs, so it needs to know how many steps an epoch
is. DeepSpeed can only read that from a dataloader it owns, so a caller driving
its own dataloader has to say. These pin what each combination produces.
"""

import types

import pytest

from deepspeed.runtime.zenflow.engine import configure_zenflow
from deepspeed.runtime.zenflow.zenflow_config import ZenFlowConfig


class _StubEngine:
    """The attributes ``configure_zenflow`` reads, and nothing else."""

    def __init__(self, config, training_dataloader=None):
        self._zenflow_config = config
        self.training_dataloader = training_dataloader
        self._config = types.SimpleNamespace(gradient_accumulation_steps=1)

    def zenflow_config(self):
        return self._zenflow_config


class _Loader:

    def __init__(self, steps):
        self.steps = steps

    def __len__(self):
        return self.steps


def _select_boundaries(select_interval, steps, full_warm_up_rounds=0):
    """Count the boundaries ``is_zenflow_select_boundary`` would report."""
    return sum(1 for micro_step in range(steps) if (micro_step - full_warm_up_rounds) >= 0 and (
        (micro_step - full_warm_up_rounds) == 0 or (select_interval != 0 and micro_step % select_interval == 0)))


@pytest.mark.parametrize("select_strategy,select_interval", [("auto", "auto"), ("epoch", 1)])
def test_epoch_length_from_a_dataloader_deepspeed_owns(select_strategy, select_interval):
    config = ZenFlowConfig(select_strategy=select_strategy, select_interval=select_interval, update_interval="auto")
    engine = _StubEngine(config, training_dataloader=_Loader(10))

    configure_zenflow(engine)

    assert config.steps_per_epoch == 10
    assert engine.select_interval == 10
    assert _select_boundaries(engine.select_interval, steps=40) == 4


@pytest.mark.parametrize("select_strategy,select_interval", [("auto", "auto"), ("epoch", 1)])
def test_epoch_length_given_by_the_user_needs_no_dataloader(select_strategy, select_interval):
    # The path for a caller that drives its own dataloader.
    config = ZenFlowConfig(select_strategy=select_strategy,
                           select_interval=select_interval,
                           update_interval="auto",
                           steps_per_epoch=10)
    engine = _StubEngine(config, training_dataloader=None)

    configure_zenflow(engine)

    assert engine.select_interval == 10
    assert _select_boundaries(engine.select_interval, steps=40) == 4


@pytest.mark.parametrize("select_strategy,select_interval", [("auto", "auto"), ("epoch", 1)])
def test_no_epoch_length_selects_once_and_says_so(select_strategy, select_interval, caplog):
    # select_interval 0 makes is_zenflow_select_boundary true exactly once, so
    # the columns chosen at the first step are used for the whole run. That is
    # the shape of the bug; the point of the test is that it is now announced.
    config = ZenFlowConfig(select_strategy=select_strategy, select_interval=select_interval, update_interval="auto")
    engine = _StubEngine(config, training_dataloader=None)

    with caplog.at_level("WARNING"):
        configure_zenflow(engine)

    assert engine.select_interval == 0
    assert _select_boundaries(engine.select_interval, steps=40) == 1
    assert any("never re-selected" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("steps_per_epoch", [0, -5])
def test_a_non_positive_epoch_length_is_refused(steps_per_epoch):
    """Reported by @ebarkhordar on #8456.

    0 multiplies select_interval straight to 0, which is_zenflow_select_boundary
    reads as "never re-select" -- the state the warning exists to announce, reached
    silently through the knob that documents it, because the warning only fires in
    the None branch. -5 is worse: select_interval stays negative and
    `micro_step % -5` fires on a schedule nobody asked for, 8 times in 40 steps.

    validate_fields already rejects an update_interval below 1 one field away.
    """
    with pytest.raises(ValueError, match="steps_per_epoch"):
        ZenFlowConfig(select_strategy="auto",
                      select_interval="auto",
                      update_interval="auto",
                      steps_per_epoch=steps_per_epoch)


def test_an_empty_dataloader_warns_rather_than_selecting_once_in_silence(caplog):
    """The programmatic path to the same 0: len(dataloader) == 0.

    configure_zenflow fills steps_per_epoch from the dataloader it owns, and the
    config validator never sees that assignment.
    """
    config = ZenFlowConfig(select_strategy="auto", select_interval="auto", update_interval="auto")
    engine = _StubEngine(config, training_dataloader=_Loader(0))

    with caplog.at_level("WARNING"):
        configure_zenflow(engine)

    assert engine.select_interval == 0
    assert any("never re-selected" in record.getMessage() for record in caplog.records)


def test_step_strategy_needs_no_epoch_length():
    config = ZenFlowConfig(select_strategy="step", select_interval=10, update_interval="auto")
    engine = _StubEngine(config, training_dataloader=None)

    configure_zenflow(engine)

    assert engine.select_interval == 10
    assert _select_boundaries(engine.select_interval, steps=40) == 4


def test_auto_strategy_warns_rather_than_raising_on_an_ignored_interval():
    # `raise Warning(...)` raises: Warning is an Exception. This combination
    # could not run at all, and the `select_interval = 1` line after it was dead.
    config = ZenFlowConfig(select_strategy="auto", select_interval=5, update_interval="auto")
    engine = _StubEngine(config, training_dataloader=_Loader(10))

    configure_zenflow(engine)

    assert engine.select_interval == 10  # one epoch, not the ignored 5
