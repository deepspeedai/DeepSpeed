# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Use to partition the activations stored for backward propagation
Therefore reduces the memory consumption
Also implements CPU checkpointing and contiguous memory checkpointing
Reduces memory consumption and memory fragmentation

Code for rng checkpointing taken from NVIDIA Megatron-LM mpu/random.py
b886b7bb972afe72bac0f5de4f42a4a7bae8ebef
"""

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch
import copy
import torch
import contextlib
from deepspeed import comm as dist
import weakref

import mmap
from torch import _C

from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.utils import logger
from deepspeed.runtime.utils import copy_to_device, move_to_device, see_memory_usage, bwc_tensor_model_parallel_rank
from deepspeed.utils.timer import SynchronizedWallClockTimer as Timers, FORWARD_GLOBAL_TIMER
from deepspeed.accelerator import get_accelerator

# DeepSpeed Checkpointing Enabled or Disabled
deepspeed_checkpointing_enabled = False

# MP parameters
mpu = None
mp_rank = None
mp_size = None
mp_group = None

# Model Parameters
num_layers = None

# Checkpointing buffers
contiguous_data_buffers = []
data_offsets = []

contiguous_size_buffers = []
size_offsets = []

timers = None

# optimization flags
PARTITION_ACTIVATIONS = False
CPU_CHECKPOINT = False
CONTIGUOUS_CHECKPOINTING = False
SYNCHRONIZE = False
PROFILE_TIME = False

# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'
transport_stream = None
cuda_device = None


def detach_variable(inputs, device=None):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            requires_grad = inp.requires_grad

            if device is not None:
                x = inp.to(device=device)
            else:
                x = inp

            x = x.detach()
            x.requires_grad = requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError("Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state) #ignore-cuda
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with get_accelerator().device(device):
                _C._cuda_setRNGState(new_state)
    else:
        # newer PyTorch
        if device == -1:
            device = torch.device(get_accelerator().device_name())
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(get_accelerator().device_name(), device)

        def cb():
            idx = device.index
            if idx is None:
                idx = get_accelerator().current_device()
            default_generator = get_accelerator().default_generator(idx)
            default_generator.set_state(new_state)

    get_accelerator().lazy_call(cb)


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        return copy.copy(self.states_)

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception('cuda rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = get_accelerator().get_rng_state()
        # Set the new state and store it.
        get_accelerator().manual_seed(seed)
        self.states_[name] = get_accelerator().get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state.
        orig_cuda_rng_state = get_accelerator().get_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = get_accelerator().get_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER


def model_parallel_cuda_manual_seed(seed):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no get_accelerator().manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel GPUs but different across
                       different model parallel groups. This is used for
                       example for dropout in the non-model-parallel regions.
        model-parallel state: This state is different among a set of model
                              parallel GPUs, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    global mpu

    tp_rank = bwc_tensor_model_parallel_rank(mpu)

    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    model_parallel_seed = offset + tp_rank
    # Data parallel gets the original seed.
    data_parallel_seed = seed

    if dist.get_rank() == 0:
        logger.info(
            '> initializing model parallel cuda seeds on global rank {}, '
            'model parallel rank {}, and data parallel rank {} with '
            'model parallel seed: {} and data parallel seed: {}'.format(dist.get_rank(), tp_rank,
                                                                        mpu.get_data_parallel_rank(),
                                                                        model_parallel_seed, data_parallel_seed), )
    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    get_accelerator().manual_seed(data_parallel_seed)
    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, model_parallel_seed)


def get_partition_start(item):
    global mp_rank, mp_size, mp_group
    size = item.numel()
    partition_size = size / mp_size
    start = partition_size * mp_rank
    return int(start)


def get_partition_size(item):
    global mp_rank, mp_size, mp_group
    size = item.numel()
    assert size % mp_size == 0, "Doesn't handle if partition activation if item is not divisible by mp size"
    partition_size = size / mp_size
    return int(partition_size)


def gather_partitioned_activations(tensors, device=None):
    global mp_rank, mp_size, mp_group
    assert len(tensors) % 2 == 0, f'Expected even count of tensors, instead got {len(tensors)}'
    inputs = []
    num_args = int(len(tensors) / 2)
    for i in range(num_args):

        item = tensors[2 * i]
        size = tensors[2 * i + 1]

        if not is_activation_to_checkpoint(item):
            inputs.append(item)
            continue

        # don't need to do all_gather if model parallel is not enabled
        if mp_group is None or mp_size == 1:
            item = item.view(list(size.numpy()))
            if device is not None:
                item = item.to(device)
            inputs.append(item)
            continue

        partition_size = item.numel()
        tensor_size = partition_size * mp_size
        if device is not None:
            flat_tensor = torch.zeros([tensor_size], dtype=item.dtype, device=device)
        else:
            flat_tensor = torch.zeros([tensor_size], dtype=item.dtype, device=item.device)
        partitions = []
        for i in range(mp_size):
            part_i = flat_tensor.narrow(0, partition_size * i, partition_size)
            if i == mp_rank:
                part_i.copy_(item)
            partitions.append(part_i)
        dist.all_gather(partitions, partitions[mp_rank], group=mp_group)
        input_tensor = flat_tensor.view(list(size.numpy()))
        item.data = input_tensor.data

        inputs.append(item)

    return tuple(inputs)


def extract_tensors(all_objects):
    """
    Separate objects in list/tuple into tensors and non-tensors and create a mapping to enable re-aggregation.
    The order of tensors and non-tensors is preserved in their respective output groups.

    Parameters:
        all_objects (list/tuple): Objects containing tensors and non-tensors to be split.

    Returns:
        tuple: Containing tensors, non-tensors, and bools of whether each position in original list/tuple was a tensor.

    """
    tensor_objects = [v for v in all_objects if torch.is_tensor(v)]
    non_tensor_objects = [v for v in all_objects if not torch.is_tensor(v)]
    tensor_flags = [torch.is_tensor(v) for v in all_objects]
    if type(all_objects) is tuple:
        return tuple(tensor_objects), tuple(non_tensor_objects), tuple(tensor_flags)
    return tensor_objects, non_tensor_objects, tensor_flags


def merge_tensors(tensor_objects, non_tensor_objects, tensor_flags):
    """
    Merge two lists (or tuples) of tensors and non-tensors using a mapping of positions in merged list (or tuple).

    Parameters:
        tensor_objects (list/tuple): Tensors to merge.
        non_tensor_objects (list/tuple): Non-tensors to merge.
        tensor_flags (list/tuple): Indicates whether each position in output is a tensor.

    Returns:
        tuple: Merge of tensors and non-tensors
    """
    merged_objects = []
    tensor_idx = 0
    non_tensor_idx = 0

    real_tensor_flags = None

    # remove the flags that are assigned to the size of the flattened tensors
    if PARTITION_ACTIVATIONS:
        real_tensor_flags = []
        previous_flag = False
        for flag in tensor_flags:
            if previous_flag:
                previous_flag = False
                continue
            previous_flag = flag
            real_tensor_flags.append(flag)
    else:
        real_tensor_flags = tensor_flags

    for is_tensor in real_tensor_flags:
        if is_tensor:
            merged_objects.append(tensor_objects[tensor_idx])
            tensor_idx += 1
        else:
            merged_objects.append(non_tensor_objects[non_tensor_idx])
            non_tensor_idx += 1

    return tuple(merged_objects)


def is_activation_to_checkpoint(item):
    """
        Is an activation to be checkpointed
    """
    global mp_size
    return torch.is_tensor(item) and item.is_floating_point() and item.numel() >= mp_size


def partition_activations(args, cpu_checkpoint, contiguous_checkpoint):
    global contiguous_data_buffers, data_offsets

    inputs = []
    num_non_fp_tensors = 0

    for arg_index, item in enumerate(args):
        if not is_activation_to_checkpoint(item):
            inputs.append(item)
            num_non_fp_tensors += 1
            continue

        i = arg_index - num_non_fp_tensors
        partition_size = get_partition_size(item)
        partition = item.detach().contiguous().view(-1).narrow(0, get_partition_start(item), partition_size).clone()

        buffer_device = torch.device('cpu') if cpu_checkpoint else partition.device

        if contiguous_checkpoint:
            if i >= len(contiguous_data_buffers):
                tensor_list = [
                    torch.tensor(()).new_empty([partition_size], dtype=partition.dtype, device=buffer_device)
                    for _ in range(num_layers)
                ]
                contiguous_data_buffers.append(tensor_list)
                data_offsets.append(0)
            elif contiguous_data_buffers[i] is None:
                tensor_list = [
                    torch.tensor(()).new_empty([partition_size], dtype=partition.dtype, device=buffer_device)
                    for _ in range(num_layers)
                ]
                contiguous_data_buffers[i] = tensor_list
                data_offsets[i] = 0

            # Because the 'new_empty' returns uninitialized pages,
            # the pages need to be populated during the cudaMemcpy time
            # which increases the data copy time. To avoid this, we
            # pre-populate these pages by simply writing 0 ahead of
            # the actual cudaMemcpy operation time. Due to the
            # previously launched GPU kernels, there is a small
            # window of time here for CPUs to populate pages asynchronously.
            contiguous_data_buffers[i][data_offsets[i]].data[range(
                0, contiguous_data_buffers[i][data_offsets[i]].data.shape[0],
                int(mmap.PAGESIZE / contiguous_data_buffers[i][data_offsets[i]].data.element_size()))] = 0

            contiguous_partition = contiguous_data_buffers[i][data_offsets[i]].data.copy_(partition.data)
            data_offsets[i] = data_offsets[i] + 1
            inputs.append(contiguous_partition)
        else:
            partition = partition.cpu() if CPU_CHECKPOINT else partition
            inputs.append(partition)

    return inputs


def get_partitioned_activations_for_backward(args, inputs, contiguous_checkpoint):
    global contiguous_size_buffers, size_offsets

    new_args = []
    num_non_fp_tensors = 0

    for arg_index, (arg, inp) in enumerate(zip(args, inputs)):
        size = torch.tensor(arg.size()) if torch.is_tensor(arg) else None
        if not is_activation_to_checkpoint(arg):
            new_args.append(arg)
            new_args.append(size)
            num_non_fp_tensors += 1
            continue

        arg.data = inp.data
        new_args.append(arg)
        i = arg_index - num_non_fp_tensors

        if contiguous_checkpoint:
            numel = size.numel()
            if i >= len(contiguous_size_buffers):
                tmp = torch.tensor(())
                contiguous_size_buffers.append(
                    tmp.new_empty([numel * num_layers], dtype=size.dtype, device=size.device))
                size_offsets.append(0)
            elif contiguous_size_buffers[i] is None:
                tmp = torch.tensor(())
                contiguous_size_buffers[i] = tmp.new_empty([numel * num_layers], dtype=size.dtype, device=size.device)
                size_offsets[i] = 0

            contiguous_size = contiguous_size_buffers[i].narrow(0, size_offsets[i], numel).data.copy_(size.data)
            contiguous_size = contiguous_size.view_as(size)
            size_offsets[i] = size_offsets[i] + numel
            new_args.append(contiguous_size)
        else:
            new_args.append(size)

    return new_args


def get_cpu_activations_for_backward(args, inputs):
    new_args = []
    for i, (arg, inp) in enumerate(zip(args, inputs)):
        if not is_activation_to_checkpoint(arg):
            new_args.append(arg)
            continue

        arg.data = inp.data
        new_args.append(arg)

    return new_args


class CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
       two main changes:
           1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`  #ignore-cuda
           2) the states in the model parallel tracker are also properly
              tracked/set/reset.
           3) Performance activation partitioning, contiguous memory optimization
           4) CPU Checkpointing
           5) Profile forward and backward functions
    """

    @staticmethod
    def forward(ctx, run_function, all_outputs, *args):
        global mpu, timers, SYNCHRONIZE, PROFILE_TIME

        def save_args_for_backward(*all_args):
            tensor_args, non_tensor_args, tensor_flags = extract_tensors(all_objects=all_args)
            ctx.deepspeed_saved_tensors = tensor_args
            ctx.non_tensor_args = non_tensor_args
            ctx.tensor_flags = tensor_flags

        if SYNCHRONIZE:
            get_accelerator().synchronize()

        if timers is None and PROFILE_TIME:
            timers = Timers()

        if PROFILE_TIME:
            timers(FORWARD_GLOBAL_TIMER).start()

        ctx.run_function = run_function
        global num_layers
        global mp_rank, mp_size, mp_group
        global contiguous_data_buffers, contiguous_size_buffers
        global data_offsets, size_offsets
        if mp_rank is None:
            if mpu is not None:
                if hasattr(mpu, 'get_tensor_model_parallel_rank'):
                    mp_rank = mpu.get_tensor_model_parallel_rank()
                    mp_size = mpu.get_tensor_model_parallel_world_size()
                    mp_group = mpu.get_tensor_model_parallel_group()
                else:
                    mp_rank = mpu.get_model_parallel_rank()
                    mp_size = mpu.get_model_parallel_world_size()
                    mp_group = mpu.get_model_parallel_group()
            else:
                mp_rank = 0
                mp_size = 1
                mp_group = None

        global cuda_device, transport_stream, PARTITION_ACTIVATIONS, buffer_0, buffer_1, buffer_0_offset, buffer_1_offset

        if cuda_device is None:
            see_memory_usage("First Forward Beginning", force=False)
            if dist.get_rank() == 0:
                logger.info(f"Activation Checkpointing Information")
                logger.info(f"----Partition Activations {PARTITION_ACTIVATIONS}, CPU CHECKPOINTING {CPU_CHECKPOINT}")
                logger.info(
                    f"----contiguous Memory Checkpointing {CONTIGUOUS_CHECKPOINTING} with {num_layers} total layers")
                logger.info(f"----Synchronization {SYNCHRONIZE}")
                logger.info(f"----Profiling time in checkpointing {PROFILE_TIME}")

            cuda_device = get_accelerator().current_device_name()
            transport_stream = get_accelerator().Stream(device=cuda_device)

        if PARTITION_ACTIVATIONS:
            inputs = partition_activations(args, CPU_CHECKPOINT, CONTIGUOUS_CHECKPOINTING)
        elif CPU_CHECKPOINT:
            inputs = copy_to_device(args, device=torch.device('cpu'), criterion_func=is_activation_to_checkpoint)

        # just in case something funky is happening such as reuse of inputs
        inputs_cuda = copy_to_device(args, device=cuda_device, criterion_func=is_activation_to_checkpoint)

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = get_accelerator().get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        see_memory_usage("Before running forward on the layer", force=False)
        # ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*inputs_cuda)

        see_memory_usage("After running forward on the layer", force=False)
        del inputs_cuda

        if PARTITION_ACTIVATIONS:
            new_args = get_partitioned_activations_for_backward(args, inputs, CONTIGUOUS_CHECKPOINTING)
            assert len(new_args) % 2 == 0, f'save_for_backward called with odd number of args, {len(new_args)}'
            save_args_for_backward(*new_args)
        elif CPU_CHECKPOINT:
            new_args = get_cpu_activations_for_backward(args, inputs)
            save_args_for_backward(*new_args)
        else:
            save_args_for_backward(*args)

        if PROFILE_TIME:
            timers(FORWARD_GLOBAL_TIMER).stop()
            timers.log([FORWARD_GLOBAL_TIMER])
        if SYNCHRONIZE:
            get_accelerator().synchronize()

        # Tensors returned from forward() may not be differentiable.
        if torch.is_tensor(outputs):
            non_grad_outputs = [outputs] if not outputs.is_floating_point() else []
        else:
            non_grad_outputs = [o for o in outputs if torch.is_tensor(o) and not o.is_floating_point()]
        ctx.mark_non_differentiable(*non_grad_outputs)

        if torch.is_tensor(outputs):
            all_outputs += [outputs]
            return outputs
        else:
            all_outputs += outputs
            outputs, _, _ = extract_tensors(all_objects=outputs)
            return tuple(outputs)

    @staticmethod
    def backward(ctx, *grads):
        global timers
        see_memory_usage("In backward", force=False)
        # removing pointers to the contiguous buffer memory
        # so that they can be garbage collected once the checkpoints
        # have been used
        if SYNCHRONIZE:
            get_accelerator().synchronize()
        if PROFILE_TIME:
            timers('backward').start()

        if CONTIGUOUS_CHECKPOINTING:
            global data_offsets, size_offsets
            global contiguous_data_buffers, contiguous_size_buffers

            for buffers in contiguous_data_buffers:
                buffers = []

            # frees up all the pointers to the checkpoints except for the ones
            # stored by save for backward
            contiguous_data_buffers = []
            contiguous_size_buffers = []
            data_offsets = []
            size_offsets = []

        see_memory_usage("In backward checkpointing code", force=False)
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), "
                               "please use .backward() if possible")

        global cuda_device, transport_stream, PARTITION_ACTIVATIONS

        if PARTITION_ACTIVATIONS:
            # with get_accelerator().stream(transport_stream):
            inputs = gather_partitioned_activations(ctx.deepspeed_saved_tensors,
                                                    device=cuda_device if CPU_CHECKPOINT else None)
            detached_inputs = detach_variable(inputs)
        elif CPU_CHECKPOINT:
            inputs = move_to_device(ctx.deepspeed_saved_tensors, cuda_device, is_activation_to_checkpoint)
            detached_inputs = detach_variable(inputs)
        else:
            inputs = ctx.deepspeed_saved_tensors
            detached_inputs = detach_variable(inputs)

        # Add non tensor input args
        detached_inputs = merge_tensors(tensor_objects=detached_inputs,
                                        non_tensor_objects=ctx.non_tensor_args,
                                        tensor_flags=ctx.tensor_flags)

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = get_accelerator().get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # if PARTITION_ACTIVATIONS:
        #     current_stream=get_accelerator().current_stream()
        #     current_stream.wait_stream(transport_stream)

        see_memory_usage("In backward checkpointing code before forward", force=False)

        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        see_memory_usage("In backward checkpointing code after forward", force=False)
        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs, )

        # Filter out non tensor outputs
        outputs, _, _ = extract_tensors(all_objects=outputs)

        # Construct arguments to autograd.backward().
        # This is usually just outputs and grads, but forward() can return tensors that
        # are not differentiable.
        output_tensors = []
        grad_tensors = []
        for out, grad in zip(outputs, grads):
            if out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)

        see_memory_usage("In backward checkpointing code before backward", force=False)

        torch.autograd.backward(output_tensors, grad_tensors)

        # Force clear our stashed tensors to prevent a memory leak in certain scenarios
        ctx.deepspeed_saved_tensors = None
        ctx.non_tensor_args = None
        ctx.tensor_flags = None

        see_memory_usage("After backward checkpointing code after backward", force=False)

        if PROFILE_TIME:
            timers('backward').stop()
            timers.log(['backward'])
        if SYNCHRONIZE:
            get_accelerator().synchronize()
        ret_list = [None, None]  # first None for ctx
        for inp in detached_inputs:
            if torch.is_tensor(inp):
                ret_list.append(inp.grad)
            else:
                ret_list.append(None)

        return tuple(ret_list)


def non_reentrant_checkpoint(function, *args):
    """This function is union of `torch.utils.checkpoint._checkpoint_without_reentrant` and `CheckpointFunction` in this module

    This function is aim to solve the back probagation error raised from all input requires no grad.
    * has already been implemented in pytorch for a while, the solution is stable at most time except for jit module mode.
    * can help to solve the issue which is hacked by `deepspeed.runtime.pipe.module.PipelineModule._is_checkpointable`

    Main modifications compared to the implementation of torch:
    1. adapt to the signature of `checkpoint` function in this module
    2. solve the non-deterministic by random state management consistent with deepspeed `CheckpointFunction`
    3. when there is partition or cpu checkpointing, gather them in the unpack_hook during back probagation
    4. make all after backward blocks in the hook which will executed after all leaf nodes backward execution.
    5. above 4. is inspired by `torch.autograd.graph.register_multi_grad_hook`, which is only implemented after 2.0.0
    """
    global mpu, timers, SYNCHRONIZE, PROFILE_TIME

    deepspeed_saved_tensors = None
    non_tensor_args = None
    tensor_flags = None

    def save_args_for_backward(*all_args):
        """keep this function to reduce the modification from original implementation"""
        nonlocal deepspeed_saved_tensors, non_tensor_args, tensor_flags
        tensor_args, non_tensor_args, tensor_flags = extract_tensors(all_objects=all_args)
        deepspeed_saved_tensors = tensor_args
        non_tensor_args = non_tensor_args
        tensor_flags = tensor_flags

    if SYNCHRONIZE:
        get_accelerator().synchronize()

    if timers is None and PROFILE_TIME:
        timers = Timers()

    if PROFILE_TIME:
        timers(FORWARD_GLOBAL_TIMER).start()

    global num_layers
    global mp_rank, mp_size, mp_group
    global contiguous_data_buffers, contiguous_size_buffers
    global data_offsets, size_offsets
    if mp_rank is None:
        if mpu is not None:
            if hasattr(mpu, 'get_tensor_model_parallel_rank'):
                mp_rank = mpu.get_tensor_model_parallel_rank()
                mp_size = mpu.get_tensor_model_parallel_world_size()
                mp_group = mpu.get_tensor_model_parallel_group()
            else:
                mp_rank = mpu.get_model_parallel_rank()
                mp_size = mpu.get_model_parallel_world_size()
                mp_group = mpu.get_model_parallel_group()
        else:
            mp_rank = 0
            mp_size = 1
            mp_group = None

    global cuda_device, transport_stream, PARTITION_ACTIVATIONS, buffer_0, buffer_1, buffer_0_offset, buffer_1_offset

    if cuda_device is None:
        see_memory_usage("First Forward Beginning", force=False)
        if dist.get_rank() == 0:
            logger.info(f"Activation Checkpointing Information")
            logger.info(f"----Partition Activations {PARTITION_ACTIVATIONS}, CPU CHECKPOINTING {CPU_CHECKPOINT}")
            logger.info(
                f"----contiguous Memory Checkpointing {CONTIGUOUS_CHECKPOINTING} with {num_layers} total layers")
            logger.info(f"----Synchronization {SYNCHRONIZE}")
            logger.info(f"----Profiling time in checkpointing {PROFILE_TIME}")

        cuda_device = get_accelerator().current_device_name()
        transport_stream = get_accelerator().Stream(device=cuda_device)

    if PARTITION_ACTIVATIONS:
        inputs = partition_activations(args, CPU_CHECKPOINT, CONTIGUOUS_CHECKPOINTING)
    elif CPU_CHECKPOINT:
        inputs = copy_to_device(args, device=torch.device('cpu'), criterion_func=is_activation_to_checkpoint)

    # just in case something funky is happening such as reuse of inputs
    inputs_cuda = copy_to_device(args, device=cuda_device, criterion_func=is_activation_to_checkpoint)

    # Copy the rng states.
    fwd_cpu_rng_state = torch.get_rng_state()
    fwd_cuda_rng_state = get_accelerator().get_rng_state()
    fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

    if PARTITION_ACTIVATIONS:
        new_args = get_partitioned_activations_for_backward(args, inputs, CONTIGUOUS_CHECKPOINTING)
        assert len(new_args) % 2 == 0, f'save_for_backward called with odd number of args, {len(new_args)}'
        save_args_for_backward(*new_args)
    elif CPU_CHECKPOINT:
        new_args = get_cpu_activations_for_backward(args, inputs)
        save_args_for_backward(*new_args)
    else:
        save_args_for_backward(*args)

    class Holder():
        """the place holder object used as activations to save memory"""
        pass

    # weakref seems utilized to discover the tensor deletion before a whole
    # forward backward pair loop finished
    storage: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
    weak_holder_list = []
    leaf_tensors = []
    backward_visited_leaf_nodes = 0

    def checkpoint_pack(tensor_from_forward):
        """used to record the activation order in the `weak_holder_list`

        the activation order in holder list is consistent between the first forward and recomputing forward.
        * the jit compiled forward will break the order consistency *
        """
        res = Holder()
        weak_holder_list.append(weakref.ref(res))

        # if this is a leaf tensor, save it for backward progression trace
        # leaf tensor used to be input or parameters, which is not activations and
        # has no memory overhead
        if tensor_from_forward.requires_grad and tensor_from_forward.is_leaf:
            leaf_tensors.append(tensor_from_forward)
        return res

    def checkpoint_unpack(holder_from_backward):
        """retrieve the activations from recompute"""
        nonlocal deepspeed_saved_tensors, non_tensor_args, tensor_flags

        # if this is the first step of backward probagation, recompute the graph and save
        # all the activations with the same order as `checkpoint_pack` does
        if len(storage) == 0:
            unpack_counter = 0

            def replay_pack(tensor_from_replay):
                """save recompute activations"""
                nonlocal unpack_counter
                unpack_counter += 1

                if weak_holder_list[unpack_counter - 1]() is None:
                    return

                detached_activations = tensor_from_replay.detach()
                storage[weak_holder_list[unpack_counter - 1]()] = detached_activations

                return

            def replay_unpack(none_value):
                """recompute graph need not to backward"""
                raise RuntimeError("You are calling backwards on a tensor that is never exposed.")

            global timers
            see_memory_usage("In backward", force=False)
            # removing pointers to the contiguous buffer memory
            # so that they can be garbage collected once the checkpoints
            # have been used
            if SYNCHRONIZE:
                get_accelerator().synchronize()
            if PROFILE_TIME:
                timers('backward').start()

            if CONTIGUOUS_CHECKPOINTING:
                global data_offsets, size_offsets
                global contiguous_data_buffers, contiguous_size_buffers

                for buffers in contiguous_data_buffers:
                    buffers = []

                # frees up all the pointers to the checkpoints except for the ones
                # stored by save for backward
                contiguous_data_buffers = []
                contiguous_size_buffers = []
                data_offsets = []
                size_offsets = []

            see_memory_usage("In backward checkpointing code", force=False)
            if not torch.autograd._is_checkpoint_valid():
                raise RuntimeError("Checkpointing is not compatible with .grad(), "
                                   "please use .backward() if possible")

            global cuda_device, transport_stream, PARTITION_ACTIVATIONS

            # gather inputs which is partitioned or checkpointed before first forward
            if PARTITION_ACTIVATIONS:
                # with get_accelerator().stream(transport_stream):
                inputs = gather_partitioned_activations(deepspeed_saved_tensors,
                                                        device=cuda_device if CPU_CHECKPOINT else None)
                detached_inputs = detach_variable(inputs)
            elif CPU_CHECKPOINT:
                inputs = move_to_device(deepspeed_saved_tensors, cuda_device, is_activation_to_checkpoint)
                detached_inputs = detach_variable(inputs)
            else:
                inputs = deepspeed_saved_tensors
                detached_inputs = detach_variable(inputs)

            # Add non tensor input args
            detached_inputs = merge_tensors(tensor_objects=detached_inputs,
                                            non_tensor_objects=non_tensor_args,
                                            tensor_flags=tensor_flags)

            # Store the current states.
            bwd_cpu_rng_state = torch.get_rng_state()
            bwd_cuda_rng_state = get_accelerator().get_rng_state()
            bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

            # Set the states to what it used to be before the forward pass.
            torch.set_rng_state(fwd_cpu_rng_state)
            _set_cuda_rng_state(fwd_cuda_rng_state)
            get_cuda_rng_tracker().set_states(fwd_cuda_rng_state_tracker)

            see_memory_usage("In backward checkpointing code before forward", force=False)
            with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(replay_pack, replay_unpack):
                _unused = function(*detached_inputs)

            see_memory_usage("In backward checkpointing code after forward", force=False)
            # Set the states back to what it was at the start of this function.
            torch.set_rng_state(bwd_cpu_rng_state)
            _set_cuda_rng_state(bwd_cuda_rng_state)
            get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

            deepspeed_saved_tensors = None
            non_tensor_args = None
            tensor_flags = None

        if holder_from_backward not in storage:
            raise RuntimeError("Attempt to retrieve a tensor saved by autograd multiple times without checkpoint"
                               " recomputation being triggered in between, this is not currently supported.")

        return storage[holder_from_backward]

    def after_backward_hook(_nonuse_grads):
        """the hook registered to all leaf tensors"""
        nonlocal leaf_tensors, backward_visited_leaf_nodes
        backward_visited_leaf_nodes += 1

        if backward_visited_leaf_nodes == len(leaf_tensors):
            see_memory_usage("After backward checkpointing code after backward", force=False)

            if PROFILE_TIME:
                timers('backward').stop()
                timers.log(['backward'])
            if SYNCHRONIZE:
                get_accelerator().synchronize()

    with torch.autograd.graph.saved_tensors_hooks(checkpoint_pack, checkpoint_unpack):
        outputs = function(*inputs_cuda)
    for leaf_tensor in leaf_tensors:
        leaf_tensor.register_hook(after_backward_hook)

    see_memory_usage("After running forward on the layer", force=False)

    if PROFILE_TIME:
        timers(FORWARD_GLOBAL_TIMER).stop()
        timers.log([FORWARD_GLOBAL_TIMER])
    if SYNCHRONIZE:
        get_accelerator().synchronize()

    all_outputs = []
    if torch.is_tensor(outputs):
        all_outputs += [outputs]
    else:
        all_outputs += outputs

    if len(all_outputs) == 1:
        return all_outputs[0]
    else:
        return tuple(all_outputs)


def checkpoint(function, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint. """

    all_outputs = []
    CheckpointFunction.apply(function, all_outputs, *args)
    if len(all_outputs) == 1:
        return all_outputs[0]
    else:
        return tuple(all_outputs)


def partition_activations_in_checkpoint(partition_activation):
    global PARTITION_ACTIVATIONS
    PARTITION_ACTIVATIONS = partition_activation
    if dist.get_rank() == 0:
        logger.info(f"**************Partition Activations {PARTITION_ACTIVATIONS}************")


def set_num_layers(nlayers):
    global num_layers
    num_layers = nlayers


def reset():
    """Resets memory buffers related to contiguous memory optimizations.
    Should be called during eval when multiple forward propagations are
    computed without any backward propagation that usually clears these
    buffers.
    Arguments:
        None

    Return:
        None
    """
    if CONTIGUOUS_CHECKPOINTING:
        global data_offsets, size_offsets
        global contiguous_data_buffers, contiguous_size_buffers

        for buffers in contiguous_data_buffers:
            buffers = []

        # frees up all the pointers to the checkpoints except for the ones
        # stored by save for backward
        contiguous_data_buffers = []
        contiguous_size_buffers = []
        data_offsets = []
        size_offsets = []


def _configure_using_config_file(config, mpu=None):
    global num_layers, PARTITION_ACTIVATIONS, CONTIGUOUS_CHECKPOINTING, \
        CPU_CHECKPOINT, SYNCHRONIZE, PROFILE_TIME

    config = DeepSpeedConfig(config, mpu=mpu).activation_checkpointing_config
    if dist.get_rank() == 0:
        logger.info(config.repr())
    PARTITION_ACTIVATIONS = config.partition_activations
    CONTIGUOUS_CHECKPOINTING = config.contiguous_memory_optimization
    num_layers = config.number_checkpoints
    CPU_CHECKPOINT = config.cpu_checkpointing
    SYNCHRONIZE = config.synchronize_checkpoint_boundary
    PROFILE_TIME = config.profile


def _configure_defaults():

    global mpu, num_layers, deepspeed_checkpointing_enabled

    global PARTITION_ACTIVATIONS, CONTIGUOUS_CHECKPOINTING, \
        CPU_CHECKPOINT, SYNCHRONIZE, PROFILE_TIME

    PARTITION_ACTIVATIONS = False
    CONTIGUOUS_CHECKPOINTING = False
    num_layers = False
    CPU_CHECKPOINT = False
    SYNCHRONIZE = False
    PROFILE_TIME = False
    deepspeed_checkpointing_enabled = True


def configure(
    mpu_,
    deepspeed_config=None,
    partition_activations=None,
    contiguous_checkpointing=None,
    num_checkpoints=None,
    checkpoint_in_cpu=None,
    synchronize=None,
    profile=None,
):
    """Configure DeepSpeed Activation Checkpointing.

    Arguments:
        mpu_: Optional: An object that implements the following methods
            get_model_parallel_rank/group/world_size, and get_data_parallel_rank/group/world_size

        deepspeed_config: Optional: DeepSpeed Config json file when provided will be used to
            configure DeepSpeed Activation Checkpointing

        partition_activations: Optional: Partitions activation checkpoint across model parallel
            GPUs when enabled. By default False. Will overwrite deepspeed_config if provided

        contiguous_checkpointing: Optional: Copies activation checkpoints to a contiguous memory
            buffer. Works only with homogeneous checkpoints when partition_activations is enabled.
            Must provide num_checkpoints. By default False. Will overwrite deepspeed_config if
            provided

        num_checkpoints: Optional: Number of activation checkpoints stored during the forward
            propagation of the model. Used to calculate the buffer size for contiguous_checkpointing
            Will overwrite deepspeed_config if provided

        checkpoint_in_cpu: Optional: Moves the activation checkpoint to CPU. Only works with
            partition_activation. Default is false. Will overwrite deepspeed_config if provided

        synchronize: Optional: Performs get_accelerator().synchronize() at the beginning and end of
            each call to deepspeed.checkpointing.checkpoint for both forward and backward pass.
            By default false. Will overwrite deepspeed_config if provided

        profile: Optional: Logs the forward and backward time for each
            deepspeed.checkpointing.checkpoint invocation. Will overwrite deepspeed_config
            if provided

    Returns:
        None
    """
    global mpu, num_layers, deepspeed_checkpointing_enabled

    global PARTITION_ACTIVATIONS, CONTIGUOUS_CHECKPOINTING, \
        CPU_CHECKPOINT, SYNCHRONIZE, PROFILE_TIME

    _configure_defaults()

    if mpu_ is not None:
        mpu = mpu_

    if deepspeed_config is not None:
        _configure_using_config_file(deepspeed_config, mpu=mpu)

    if partition_activations is not None:
        PARTITION_ACTIVATIONS = partition_activations

    if contiguous_checkpointing is not None:
        CONTIGUOUS_CHECKPOINTING = contiguous_checkpointing

    if num_checkpoints is not None:
        num_layers = num_checkpoints

    if checkpoint_in_cpu is not None:
        CPU_CHECKPOINT = checkpoint_in_cpu

    if synchronize is not None:
        SYNCHRONIZE = synchronize

    if profile is not None:
        PROFILE_TIME = profile

    if CONTIGUOUS_CHECKPOINTING:
        assert PARTITION_ACTIVATIONS, "Contiguous Checkpointing is only available with partitioned activations. Set partitioned activations to true in deepspeed config"
    if CONTIGUOUS_CHECKPOINTING:
        assert num_layers is not None, "Must specify the number of layers with contiguous memory checkpointing"


def is_configured():
    """True if deepspeed activation checkpointing has been configured
        by calling deepspeed.checkpointing.configure, else returns false

    Arguments:
        None

    Return:
        True of configured, else False
    """
    return deepspeed_checkpointing_enabled
