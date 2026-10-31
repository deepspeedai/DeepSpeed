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


def _dualpipev_channel(cmd, rank, stages):
    """(sender rank, receiver rank, kind) of a DualPipeV communication instruction."""
    if not hasattr(cmd, 'phase'):
        return None
    downstream = rank + 1 if cmd.phase == 0 else rank - 1
    upstream = rank - 1 if cmd.phase == 0 else rank + 1
    if type(cmd) == schedule.SendActivation:
        return (rank, downstream, 'act')
    if type(cmd) == schedule.RecvActivation:
        return (upstream, rank, 'act')
    if type(cmd) == schedule.SendGrad:
        return (rank, upstream, 'grad')
    if type(cmd) == schedule.RecvGrad:
        return (downstream, rank, 'grad')
    return None


@pytest.mark.parametrize('forward_only', [False, True])
@pytest.mark.parametrize('stages, micro_batches', [(1, 2), (1, 5), (2, 4), (2, 7), (3, 6), (4, 8), (4, 13)])
def test_dualpipev_schedule_runs_to_completion(stages, micro_batches, forward_only):
    """Replay all ranks' instruction streams against the batched p2p protocol.

    Catches a schedule that deadlocks (a rank waits on a batch its peer never matches, or the
    one-time shape exchange is reached by the two sides in different situations), computes on a
    micro-batch before its data arrived, or sends and receives micro-batches in different orders.
    """
    streams = [[
        cmd for step in schedule.DualPipeVSchedule(micro_batches, stages, rank, forward_only=forward_only)
        for cmd in step
    ] for rank in range(stages)]
    pos = [0] * stages
    pending = [[] for _ in range(stages)]  # (channel, index, micro-batch) posted since the last CommitP2P
    posted = {}  # channel -> [sent micro-batches, received micro-batches]
    completed = [set() for _ in range(stages)]  # ('recv', buffer) once its batch finished, plus compute done
    meta_done = [set() for _ in range(stages)]

    def chunk(cmd):
        return cmd.buffer_id % micro_batches

    def phase1_buffer(buffer_id):
        return buffer_id + micro_batches

    def can_run(rank, cmd):
        done = completed[rank]
        if type(cmd) == schedule.ForwardPass:
            if rank == 0 and cmd.phase == 0:
                return ('load', cmd.buffer_id) in done
            if rank == stages - 1 and cmd.phase == 1:
                return ('fwd', cmd.buffer_id - micro_batches) in done
            return ('recv', cmd.buffer_id) in done
        if type(cmd) == schedule.BackwardPass:
            if rank == 0 and cmd.phase == 1:
                return ('fwd', cmd.buffer_id) in done
            if rank == stages - 1 and cmd.phase == 0:
                return ('bwd', phase1_buffer(cmd.buffer_id)) in done
            return ('recv', cmd.buffer_id) in done and ('fwd', cmd.buffer_id) in done
        if type(cmd) == schedule.SendActivation:
            return ('fwd', cmd.buffer_id) in done
        if type(cmd) == schedule.SendGrad:
            return ('bwd', cmd.buffer_id) in done
        if type(cmd) == schedule.CommitP2P:
            return all(len(posted[ch][1 - role]) > idx for ch, role, idx, _ in pending[rank])
        return True

    def peer_reached_meta(rank, channel):
        peer = channel[1] if channel[0] == rank else channel[0]
        for cmd in streams[peer][pos[peer]:]:
            if _dualpipev_channel(cmd, peer, stages) == channel and channel not in meta_done[peer]:
                return cmd is streams[peer][pos[peer]]
        return True

    def run(rank):
        cmd = streams[rank][pos[rank]]
        if not can_run(rank, cmd):
            return False
        channel = _dualpipev_channel(cmd, rank, stages)
        if channel is not None:
            if channel[2] == 'act' and channel not in meta_done[rank]:
                if not peer_reached_meta(rank, channel):
                    return False
                meta_done[rank].add(channel)
            role = 0 if channel[0] == rank else 1
            sides = posted.setdefault(channel, [[], []])
            pending[rank].append((channel, role, len(sides[role]), cmd.buffer_id))
            sides[role].append(chunk(cmd))
        elif type(cmd) == schedule.CommitP2P:
            for ch, role, idx, buffer_id in pending[rank]:
                assert posted[ch][0][idx] == posted[ch][1][idx], f'{ch}: transfer {idx} pairs different micro-batches'
                if role == 1:
                    completed[rank].add(('recv', buffer_id))
            pending[rank] = []
        elif type(cmd) == schedule.LoadMicroBatch:
            completed[rank].add(('load', cmd.buffer_id))
        elif type(cmd) == schedule.ForwardPass:
            completed[rank].add(('fwd', cmd.buffer_id))
        elif type(cmd) == schedule.BackwardPass:
            completed[rank].add(('bwd', cmd.buffer_id))
        pos[rank] += 1
        return True

    while any(pos[rank] < len(streams[rank]) for rank in range(stages)):
        progressed = [pos[rank] < len(streams[rank]) and run(rank) for rank in range(stages)]
        assert any(
            progressed
        ), f'deadlock at {[streams[r][pos[r]] if pos[r] < len(streams[r]) else None for r in range(stages)]}'

    assert all(not pending[rank] for rank in range(stages))
    for rank in range(stages):
        for phase in (0, 1):
            expected = 0 if forward_only else micro_batches
            assert sum(1 for c in streams[rank]
                       if type(c) == schedule.ForwardPass and c.phase == phase) == micro_batches
            assert sum(1 for c in streams[rank] if type(c) == schedule.BackwardPass and c.phase == phase) == expected
    for channel, (sent, received) in posted.items():
        assert sent == received, f'{channel}: sends {sent} vs receives {received}'
