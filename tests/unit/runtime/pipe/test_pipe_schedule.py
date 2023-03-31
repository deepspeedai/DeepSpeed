# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import deepspeed.runtime.pipe.schedule as schedule


def _count_type(cmds, classtype):
    return len(list(filter(lambda c: type(c) == classtype, cmds)))


def test_pipe_inference_schedule_singlestage():
    sched = schedule.InferenceSchedule(micro_batches=4, stages=1, stage_id=0)
    assert sched.num_micro_batches == 4
    full = list(iter(sched))
    for idx, cmds in enumerate(full):
        assert len(cmds) == 2
        assert type(cmds[0]) == schedule.LoadMicroBatch
        assert type(cmds[1]) == schedule.ForwardPass
        assert cmds[0].buffer_id == cmds[1].buffer_id
    assert len(full) == sched.num_micro_batches


def test_pipe_train_schedule_singlestage():
    sched = schedule.TrainSchedule(micro_batches=4, stages=1, stage_id=0)
    assert sched.num_micro_batches == 4
    full = list(iter(sched))
    for idx, cmds in enumerate(full):
        if (idx % 2) != 0:
            assert (len(cmds) == 1) or (len(cmds) == 4)
            assert type(cmds[0]) == schedule.BackwardPass
        else:
            assert len(cmds) == 2
            assert type(cmds[0]) == schedule.LoadMicroBatch
            assert type(cmds[1]) == schedule.ForwardPass
            assert cmds[0].buffer_id == cmds[1].buffer_id
    assert len(full) == sched.num_micro_batches * 2


@pytest.mark.parametrize('micro_batches', [1, 3, 8, 10])
def test_pipe_inference_schedule_firststage(micro_batches, stages=3):
    sched = schedule.InferenceSchedule(micro_batches=micro_batches, stages=stages, stage_id=0)
    assert sched.num_micro_batches == micro_batches
    full = list(iter(sched))
    for idx, cmds in enumerate(full):
        # Ensure we don't send an activation the first step
        if idx == 0:
            assert len(cmds) == 2
            assert type(cmds[0]) == schedule.LoadMicroBatch
            assert type(cmds[1]) == schedule.ForwardPass
            assert cmds[0].buffer_id == cmds[1].buffer_id
            continue

        # the last active step is only a send
        if idx == sched.num_micro_batches:
            assert len(cmds) == 1
            assert type(cmds[0]) == schedule.SendActivation
            continue

        # no work later on
        if idx > sched.num_micro_batches:
            assert len(cmds) == 0
            continue

        # Normally we need to load/forward/send
        assert len(cmds) == 3
        assert _count_type(cmds, schedule.LoadMicroBatch) == 1
        assert _count_type(cmds, schedule.ForwardPass) == 1
        assert _count_type(cmds, schedule.SendActivation) == 1
    assert len(full) == micro_batches + stages - 1


@pytest.mark.parametrize('micro_batches', [1, 3, 8, 10])
def test_pipe_inference_schedule_midstage(micro_batches, stages=3):
    sched = schedule.InferenceSchedule(micro_batches=micro_batches, stages=stages, stage_id=1)

    full = list(iter(sched))
    for idx, cmds in enumerate(full):
        if idx < sched.stage:
            assert len(cmds) == 0
            continue
        if idx == sched.stage + sched.num_micro_batches:
            assert len(cmds) == 1
            assert type(cmds[0]) == schedule.SendActivation
            continue
        if idx > sched.stage + sched.num_micro_batches:
            assert len(cmds) == 0
            continue
        assert _count_type(cmds, schedule.LoadMicroBatch) == 0
        assert _count_type(cmds, schedule.ForwardPass) == 1
        assert _count_type(cmds, schedule.RecvActivation) == 1
        if idx > sched.stage:
            assert _count_type(cmds, schedule.SendActivation) == 1
    assert len(full) == micro_batches + stages - 1


@pytest.mark.parametrize('micro_batches', [1, 3, 8, 10])
def test_pipe_inference_schedule_laststage(micro_batches, stages=3):
    sched = schedule.InferenceSchedule(micro_batches=micro_batches, stages=stages, stage_id=2)
    full = list(iter(sched))
    for idx, cmds in enumerate(full):
        if idx < sched.stage or idx > sched.stage + sched.num_micro_batches:
            assert len(cmds) == 0
            continue
        assert _count_type(cmds, schedule.LoadMicroBatch) == 1
        assert _count_type(cmds, schedule.ForwardPass) == 1
        assert _count_type(cmds, schedule.RecvActivation) == 1
        assert _count_type(cmds, schedule.SendActivation) == 0
    assert len(full) == micro_batches + stages - 1


def test_pipe_schedule_firststage():
    sched = schedule.TrainSchedule(micro_batches=8, stages=3, stage_id=0)
    for cmds in sched:
        assert all(instr.__class__ != schedule.SendGrad for instr in cmds)
        assert all(instr.__class__ != schedule.RecvActivation for instr in cmds)
        for instr in cmds:
            if isinstance(instr, schedule.BufferOpInstruction):
                assert 0 <= instr.buffer_id < sched.num_pipe_buffers()


def test_pipe_schedule_laststage():
    sched = schedule.TrainSchedule(stages=3, micro_batches=4, stage_id=2)
    assert len(list(iter(sched))) == 2 * (sched.micro_batches + sched.stages - 1)
    for cmds in sched:
        assert all(instr.__class__ != schedule.SendActivation for instr in cmds)
        assert all(instr.__class__ != schedule.RecvGrad for instr in cmds)


def test_pipe_stagequery():
    sched = schedule.TrainSchedule(stages=3, micro_batches=4, stage_id=0)
    assert sched.is_first_stage
    assert not sched.is_last_stage

    sched = schedule.TrainSchedule(stages=3, micro_batches=4, stage_id=1)
    assert not sched.is_first_stage
    assert not sched.is_last_stage

    sched = schedule.TrainSchedule(stages=3, micro_batches=4, stage_id=2)
    assert not sched.is_first_stage
    assert sched.is_last_stage
