# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from ..utils import call_to_str

from abc import ABC, abstractmethod


class PipeSchedule(ABC):
    """Directs the execution of a pipeline engine by generating sequences of
    :class:`PipeInstruction`.

    Schedules are generators that yield sequences of
    :class:`PipeInstruction` to process the micro-batches in one batch.
    Each yielded step is atomic in the sense that a barrier
    synchronization can be placed between successive steps without
    deadlock.

    Below is an example schedule that implements data parallelism with gradient accumulation:

    .. code-block:: python

        class DataParallelSchedule(PipeSchedule):
            def steps(self):
                for step_id in range(self.micro_batches):
                    cmds = [
                        LoadMicroBatch(buffer_id=0),
                        ForwardPass(buffer_id=0),
                        BackwardPass(buffer_id=0),
                    ]
                    if step_id == self.micro_batches - 1:
                        cmds.extend([
                            ReduceGrads(),
                            OptimizerStep(),
                        ])
                    yield cmds

            def num_pipe_buffers(self):
                return 1

    Args:
        micro_batches (int): The number of micro-batches that comprise a batch.
        stages (int): The number of pipeline stages.
        stage_id (int): The pipe stage that will execute the generated schedule.
    """

    def __init__(self, micro_batches, stages, stage_id):
        super().__init__()
        self.micro_batches = micro_batches
        self.stages = stages
        self.stage_id = stage_id
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

    @abstractmethod
    def steps(self):
        """Yield a list of :class:`PipeInstruction` for each step in the schedule.

        .. note::
            Schedules must implement ``steps()`` to define the schedule.

        Returns:
            Instructions to be executed as one step of the pipeline
        """
        pass

    def num_pipe_buffers(self):
        """The number of pipeline buffers that will be used by this stage.

        .. note::
            Schedules should specialize ``num_pipe_buffers()`` for memory savings at scale.

        Returns:
            The number of buffers for the engine to allocate.
        """
        return self.micro_batches

    def _valid_micro_batch(self, micro_batch_id):
        return 0 <= micro_batch_id < self.micro_batches

    def _valid_stage(self, stage_id):
        return 0 <= stage_id < self.stages

    @property
    def stage(self):
        """Stage index used to configure this schedule."""
        return self.stage_id

    @property
    def num_stages(self):
        """The number of total pipeline stages used to configure this schedule."""
        return self.stages

    @property
    def num_micro_batches(self):
        """The number of total micro_batches used to configure this schedule."""
        return self.micro_batches

    @property
    def is_first_stage(self):
        """True if the configured ``stage_id`` is the first stage in the pipeline."""
        return self.stage_id == 0

    @property
    def is_last_stage(self):
        """True if the configured ``stage_id`` is the last stage in the pipeline."""
        return self.stage_id == self.stages - 1

    def _buffer_idx(self, micro_batch_id):
        """Map a micro-batch index to a pipeline buffer index.

        This method uses a cyclic allocation strategy.

        Args:
            micro_batch_id (int): The micro-batch index relative to the beginning of the schedule.

        Returns:
            int: The index of the buffer that should store data.
        """
        assert self._valid_micro_batch(micro_batch_id)
        return micro_batch_id % self.num_pipe_buffers()

    def __iter__(self):
        self.it = None
        return self

    def __next__(self):
        if self.it is None:
            self.it = self.steps()
        return next(self.it)


class InferenceSchedule(PipeSchedule):
    """A schedule for inferencing batches using pipeline parallelism.
    """

    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = self.micro_batches + self.stages - 1
        for step_id in range(total_steps):
            cmds = []
            micro_batch_id = step_id - self.stage_id

            # Alternate send/recv buffers
            if _is_even(self.stage_id):
                recv_buf = step_id % 2
                send_buf = (step_id + 1) % 2
            else:
                recv_buf = (step_id + 1) % 2
                send_buf = step_id % 2

            if self.is_first_stage or self.is_last_stage:
                if self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(recv_buf))

            if _is_even(self.stage_id):
                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id - 1):
                        cmds.append(SendActivation(send_buf))
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        cmds.append(RecvActivation(recv_buf))
            else:
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        cmds.append(RecvActivation(recv_buf))

                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id - 1):
                        cmds.append(SendActivation(send_buf))

            if self._valid_micro_batch(micro_batch_id):
                cmds.append(ForwardPass(recv_buf))

            yield cmds

    def num_pipe_buffers(self):
        """Only two pipeline buffers are required for inferencing.

        Returns:
            ``2``
        """
        return 2


class TrainSchedule(PipeSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """

    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.prev_stage):
                    cmds.append(SendGrad(prev_buffer))
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(self.prev_stage):
                    cmds.append(RecvActivation(curr_buffer))
            else:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(self.next_stage):
                    cmds.append(RecvGrad(curr_buffer))
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.next_stage):
                    cmds.append(SendActivation(prev_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append(ForwardPass(curr_buffer))
                else:
                    cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

    def num_pipe_buffers(self):
        """Return the number of pipeline buffers required for this stage.

        This is equivalent to the maximum number of in-flight forward passes,
        since we need to remember the activations of forward passes in order
        to run backpropagation. For synchronous 1F1B, this is equivalent to
        the index difference between this stage and the last stage.
        """
        buffers = min(self.stages - self.stage_id, self.micro_batches)
        return max(2, buffers)

    def _step_to_micro_batch(self, step_id):
        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False

        else:
            assert False

        return micro_batch_id, is_forward

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id


class DataParallelSchedule(PipeSchedule):
    """An example schedule that trains using traditional data parallelism with gradient
    accumulation.
    """

    def steps(self):
        """"""
        for step_id in range(self.micro_batches):
            cmds = [
                LoadMicroBatch(buffer_id=0),
                ForwardPass(buffer_id=0),
                BackwardPass(buffer_id=0),
            ]
            if step_id == self.micro_batches - 1:
                cmds.extend([
                    ReduceGrads(),
                    OptimizerStep(),
                ])
            yield cmds

    def num_pipe_buffers(self):
        """Only one pipeline buffer needed.
        """
        return 1


class DualPipeVSchedule(PipeSchedule):
    """The DualPipeV schedule of DeepSeek-V3.

    GitHub Repo: https://github.com/deepseek-ai/DualPipe

    Every rank holds two pipeline stages: stage ``stage_id`` (phase 0, activations flow
    towards the last rank) and stage ``2 * stages - 1 - stage_id`` (phase 1, activations
    flow back towards rank 0). Micro-batches enter phase 0 on rank 0, turn around on the
    last rank and finish in phase 1 on rank 0, which is therefore both the first and the
    loss stage. Interleaving the two phases fills the 1F1B bubble.

    Sends and receives are queued and launched as one batch by :class:`CommitP2P`, so an
    instruction's communication is only complete once the next ``CommitP2P`` has run.
    The reference implementation also defers weight gradients ("zero bubble"); this
    schedule keeps them inside :class:`BackwardPass`.

    Args:
        micro_batches (int): Must be at least ``2 * stages``.
        stages (int): The number of pipeline ranks (each holds two stages).
        stage_id (int): The pipeline rank that will execute the generated schedule.
        forward_only (bool): Skip backward passes and gradient exchanges.
    """

    def __init__(self, micro_batches, stages, stage_id, forward_only=False):
        super().__init__(micro_batches, stages, stage_id)
        if micro_batches < 2 * stages:
            raise ValueError(f'DualPipeV needs at least 2 * {stages} micro-batches, got {micro_batches}')
        self.forward_only = forward_only

    def num_pipe_buffers(self):
        """One slot per (phase, micro-batch): ``phase * micro_batches + micro_batch``."""
        return 2 * self.micro_batches

    def steps(self):
        rank = self.stage_id
        num_ranks = self.stages
        is_first = rank == 0
        is_last = rank == num_ranks - 1
        counters = {name: [0, 0] for name in ('fwd', 'bwd', 'send_fwd', 'send_bwd', 'recv_fwd', 'recv_bwd')}

        def buffer(phase, counter):
            micro_batch = counters[counter][phase]
            counters[counter][phase] += 1
            return phase * self.micro_batches + micro_batch

        def recv_forward(phase):
            if (is_first and phase == 0) or (is_last and phase == 1):
                return []
            return [RecvActivation(buffer(phase, 'recv_fwd'), phase=phase)]

        def send_forward(phase):
            if (is_first and phase == 1) or (is_last and phase == 0):
                return []
            return [SendActivation(buffer(phase, 'send_fwd'), phase=phase)]

        def recv_backward(phase):
            if self.forward_only or (is_first and phase == 1) or (is_last and phase == 0):
                return []
            return [RecvGrad(buffer(phase, 'recv_bwd'), phase=phase)]

        def send_backward(phase):
            if self.forward_only or (is_first and phase == 0) or (is_last and phase == 1):
                return []
            return [SendGrad(buffer(phase, 'send_bwd'), phase=phase)]

        def forward(phase):
            buffer_id = buffer(phase, 'fwd')
            load = [LoadMicroBatch(buffer_id)] if is_first and phase == 0 else []
            return load + [ForwardPass(buffer_id, phase=phase)]

        def backward(phase):
            if self.forward_only:
                return []
            return [BackwardPass(buffer(phase, 'bwd'), phase=phase)]

        def forward_chunk(phase, recv=True, send=True):
            cmds = recv_forward(phase) if recv else []
            cmds += [CommitP2P()] + forward(phase)
            return cmds + (send_forward(phase) if send else [])

        def backward_chunk(phase, send=True):
            cmds = recv_backward(phase) + [CommitP2P()] + backward(phase)
            return cmds + (send_backward(phase) if send else [])

        def forward_backward_chunk(phase0, phase1, recv0=True):
            cmds = recv_forward(phase0) if recv0 else []
            cmds += recv_backward(phase1) + [CommitP2P()] + forward(phase0) + backward(phase1)
            return cmds + send_forward(phase0) + send_backward(phase1)

        def weight_chunk():
            return [] if self.forward_only else [CommitP2P()]

        # Step 1: nF0
        for _ in range((num_ranks - rank - 1) * 2):
            yield forward_chunk(0)

        # Step 2: nF0F1
        step_2 = rank + 1
        yield recv_forward(0)
        for i in range(step_2):
            cmds = forward_chunk(0, recv=False, send=False) + recv_forward(0)
            cmds += forward_chunk(1, send=(not is_last) or (i < step_2 - 1))
            yield cmds + send_forward(0)

        # Step 3: nB1W1F1
        for _ in range(num_ranks - rank - 1):
            yield backward_chunk(1) + recv_forward(1) + weight_chunk() + forward_chunk(1, recv=False)

        # Step 4 (main step): nF0B1F1B0
        for i in range(self.micro_batches - num_ranks * 2 + rank + 1):
            if i == 0:
                if is_last:
                    cmds = forward_chunk(0, recv=False, send=False) + send_forward(1)
                    cmds += backward_chunk(1, send=False) + send_forward(0) + send_backward(1)
                else:
                    cmds = forward_backward_chunk(0, 1, recv0=False)
            else:
                cmds = forward_backward_chunk(0, 1)
            yield cmds + forward_backward_chunk(1, 0)

        # Step 5: nB1F1B0
        for _ in range(num_ranks - rank - 1):
            yield backward_chunk(1) + forward_backward_chunk(1, 0)

        # Step 6: nB1B0
        for _ in range(rank + 1):
            yield backward_chunk(1) + backward_chunk(0)

        # Step 7: nWB0
        for _ in range(num_ranks - rank - 1):
            yield weight_chunk() + backward_chunk(0)

        # Step 8: nW
        for _ in range(rank + 1):
            yield weight_chunk()

        cmds = [CommitP2P()]
        if not self.forward_only:
            cmds += [ReduceTiedGrads(), ReduceGrads(), OptimizerStep()]
        yield cmds


class PipeInstruction:
    """Base class for all instructions to be executed by the pipeline engine.

    All keyword arguments are stored as members similar to a ``namedtuple``. These are
    then accessible to the :class:`PipeEngine` during execution.

    Args:
        kwargs (optional): keyword arguments to store as members
    """

    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        self.kwargs = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        return call_to_str(self.name, **self.kwargs)


class OptimizerStep(PipeInstruction):
    """Performs one step with the optimizer and zeros gradients.

    .. note:: Should be issued after :class:`ReduceGrads` and :class:`ReduceTiedGrads`.

    .. note:: Can be a synchronization point among data-parallel ranks.
    """
    pass


class ReduceGrads(PipeInstruction):
    """Reduce the computed gradients among data-parallel processes within the stage.
    """
    pass


class ReduceTiedGrads(PipeInstruction):
    """Reduce the computed gradients of tied modules within a pipeline-parallel group.

    .. warning::
        The stages included in this synchronization point are not known until
        the model is partitioned among pipeline stages. In the worst case, it
        includes all pipeline stages. This instruction should be scheduled
        carefully to avoid deadlocks.
    """
    pass


class BufferOpInstruction(PipeInstruction):
    """A pipeline instruction that operates on pipeline buffer(s).

    Args:
        buffer_id (int): the index of the pipeline buffer() to modify.
    """

    def __init__(self, buffer_id, **kwargs):
        super().__init__(buffer_id=buffer_id, **kwargs)


# IO
class LoadMicroBatch(BufferOpInstruction):
    """Load a micro-batch into a buffer.

    Roughly:

    .. code-block:: python

        buffers['inputs'][buffer_id] = next(data_iter)
    """
    pass


# Compute
class ForwardPass(BufferOpInstruction):
    """Compute a forward pass.

    Roughly:

    .. code-block:: python

        buffers['outputs'][buffer_id] = forward(buffers['inputs'][buffer_id])
    """
    pass


class BackwardPass(BufferOpInstruction):
    """Compute a backward pass and accumulate gradients.

    Roughly:

    .. code-block:: python

        outputs = buffers['outputs'][buffer_id]
        gradients = buffers['gradients'][buffer_id]
        torch.autograd.backward(tensors=outputs,
                                grad_tensors=gradients)
    """
    pass


# Communication
class SendActivation(BufferOpInstruction):
    """Send activations to the next stage in the pipeline.

    Roughly:

    .. code-block:: python

        send(buffers['outputs'][buffer_id])

    .. note::
        The communication is blocking and must be paired with a :class:`RecvActivation`
        on the next pipeline stage to avoid deadlock.
    """
    pass


class RecvActivation(BufferOpInstruction):
    """Receive activations from the previous stage in the pipeline.

    Roughly:

    .. code-block:: python

        buffers['inputs'][buffer_id] = recv()

    .. note::
        The communication is blocking and must be paired with a :class:`SendActivation`
        on the previous pipeline stage to avoid deadlock.
    """
    pass


class SendGrad(BufferOpInstruction):
    """Send computed gradients to the previous pipeline stage.
    with respect to the received activations

    .. note::
        Only received tensors with ``requires_grad==True`` will produce gradients.
        Missing gradients will be replaced with ``None`` on the receiving stage.

    .. note::
        The communication is blocking and must be paired with a :class:`RecvGrad`
        on the previous pipeline stage to avoid deadlock.
    """
    pass


class RecvGrad(BufferOpInstruction):
    """Receive computed gradients the next pipeline stage.

    .. note::
        Only activations with ``requires_grad==True`` will produce gradients.
        Missing gradients will be replaced with ``None``.

    .. note::
        The communication is blocking and must be paired with a :class:`SendGrad`
        on the next pipeline stage to avoid deadlock.
    """
    pass


class CommitP2P(PipeInstruction):
    """Launch the queued asynchronous sends and receives as one batch and wait for them.

    Used by schedules whose communication instructions only queue work, such as
    :class:`DualPipeVSchedule`.
    """
    pass


def _is_even(x):
    return x % 2 == 0


def _is_odd(x):
    return x % 2 != 0
