# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from functools import partial
from threading import Lock

import torch

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import InsertPostInitMethodToModuleSubClasses
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload

from .passes import zero3_compile, prefetch, selective_gather, offload_parameters, offload_activation
from .backend import make_backend, launch_compile_passes, init_schedule
from .patch_fake_tensor import patch_fake_tensor
from .util import get_deepcompile_handle, add_pre_backward_hook, add_post_backward_hook
from .z3_eager_fallback import DeepCompileZ3EagerFallback

WARMUP = 5

_MISSING = object()
_DYNAMO_CONFIG_NAMES = ("force_parameter_static_shapes", "force_nn_module_property_static_shapes")
_DYNAMO_CONFIG_OWNERS = {}
_DYNAMO_CONFIG_LOCK = Lock()


def _allow_dynamo_dynamic_parameter_shapes_for_z3(compile_kwargs):
    """Acquire process-wide ZeRO-3 Dynamo config ownership and return its release callback."""
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is None:
        try:
            import torch._dynamo as dynamo
        except ImportError:
            return None

    dynamo_config = getattr(dynamo, "config", None)
    if dynamo_config is None:
        return None

    owner_token = object()
    config_key = id(dynamo_config)
    with _DYNAMO_CONFIG_LOCK:
        state = _DYNAMO_CONFIG_OWNERS.get(config_key)
        if state is None or state["config"] is not dynamo_config:
            previous_values = {
                config_name: getattr(dynamo_config, config_name)
                for config_name in _DYNAMO_CONFIG_NAMES if hasattr(dynamo_config, config_name)
            }
            if not previous_values:
                return None
            state = {"config": dynamo_config, "previous_values": previous_values, "owner_tokens": set()}
            _DYNAMO_CONFIG_OWNERS[config_key] = state
        state["owner_tokens"].add(owner_token)
        for config_name in state["previous_values"]:
            setattr(dynamo_config, config_name, False)

    def restore():
        with _DYNAMO_CONFIG_LOCK:
            state = _DYNAMO_CONFIG_OWNERS.get(config_key)
            if state is None or state["config"] is not dynamo_config or owner_token not in state["owner_tokens"]:
                return
            state["owner_tokens"].remove(owner_token)
            if state["owner_tokens"]:
                return
            for config_name, previous_value in state["previous_values"].items():
                setattr(dynamo_config, config_name, previous_value)
            del _DYNAMO_CONFIG_OWNERS[config_key]

    return restore


def _resolve_expected_grad_dtype(param):
    # Match PyTorch's leaf grad accumulation contract. grad_dtype can be a
    # dtype, or None to allow any incoming gradient dtype:
    # https://docs.pytorch.org/docs/main/generated/torch.sparse.semi_structured.SparseSemiStructuredTensorCUSPARSELT.html#torch.sparse.semi_structured.SparseSemiStructuredTensorCUSPARSELT.grad_dtype
    grad_dtype = getattr(param, "grad_dtype", _MISSING)
    if grad_dtype is None:
        return None
    if grad_dtype is not _MISSING:
        return grad_dtype
    return param.dtype


def init_z3(engine, backend, compile_config, compile_kwargs, schedule=None):

    # Validate before touching the engine: everything below removes hooks and unpatches modules,
    # so raising later would leave a half-converted engine behind.
    # zero_use_cpu_optimizer(), not zero_offload_optimizer(): the latter returns the config
    # object, which is present but inert for `offload_optimizer: {}` or `device: none`.
    if compile_config.offload_opt_states and engine.zero_use_cpu_optimizer():
        raise ValueError("compile.offload_opt_states cannot be combined with ZeRO's "
                         "zero_optimization.offload_optimizer set to cpu or nvme: both manage the "
                         "same optimizer state. ZeRO keeps it off the accelerator for the whole step "
                         "and runs the optimizer there, while this pass keeps it resident when memory "
                         "allows and moves it around the compiled graph. Enable one of them.")

    if compile_config.offload_activation and (compile_config.offload_parameters or compile_config.offload_opt_states):
        raise ValueError("compile.offload_activation cannot be combined with offload_parameters or "
                         "offload_opt_states; choose one offloading target per run. Each of them plans "
                         "against the whole memory budget on its own, so two of them together move far "
                         "more data than the run needs.")

    optimizer = engine.optimizer
    use_opt = not isinstance(optimizer, DeepSpeedZeRoOffload)

    if use_opt and hasattr(optimizer, "ipg_buckets"):
        optimizer.ipg_buckets.clear()
        get_accelerator().empty_cache()

    dc = get_deepcompile_handle()
    dc.init(engine.data_parallel_group, compile_config, engine.zero_reduce_bucket_size())

    engine._deepcompile_z3_eager_fallback = DeepCompileZ3EagerFallback(engine)
    add_post_backward_hook(engine._deepcompile_z3_eager_fallback.complete_backward)

    if use_opt:
        optimizer.parameter_offload._remove_module_hooks()

        for hook in optimizer._grad_acc_hooks:
            hook.remove()
        optimizer._grad_acc_hooks.clear()

    # Unpatch linear
    if hasattr(InsertPostInitMethodToModuleSubClasses, "linear_bk"):
        torch.nn.functional.linear = InsertPostInitMethodToModuleSubClasses.linear_bk

    if compile_config.symmetric_memory:
        group_name = engine.data_parallel_group.group_name
        dist.enable_symm_mem_for_group(group_name)

    for p in engine.module.parameters():
        grad_buffer = torch.Tensor()
        if use_opt:
            grad_buffer = optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[p.ds_id]

        # Disable persistent param
        p.ds_persist = False
        dc.register_z3_param(p.ds_id, p.ds_shape, p.ds_tensor, grad_buffer, p.ds_persist,
                             _resolve_expected_grad_dtype(p))

    if schedule is None:
        if compile_config.offload_parameters and compile_config.offload_opt_states:
            raise ValueError("offload_parameters and offload_opt_states cannot be enabled together; "
                             "choose one offloading target per run. Note that offload_parameters may have "
                             "been enabled implicitly: the engine turns it on when the ZeRO config "
                             "offloads both optimizer and parameters to CPU.")
        schedule = []
        if (compile_config.offload_parameters):
            schedule.append((0, [zero3_compile.add_z3_gather_release, offload_parameters.offload_parameter_fwd]))
        elif compile_config.offload_opt_states:
            from .passes.offload_adam_states import move_opt_states, offload_adam_states_for_init
            schedule.append((0, [zero3_compile.add_z3_gather_release]))
            # States exist from step 0's optimizer step, so offloading engages at step 1.
            # for_init empties them before profiling, so the plan is made against the floor and a
            # job that only fits with offloading never runs a step with everything resident.
            schedule.append((1, [offload_adam_states_for_init, zero3_compile.add_z3_gather_release, move_opt_states]))
        elif compile_config.offload_activation:
            offload_activation.register_activation_offload_ops()
            # The two halves run at different steps, and the split matters.
            #
            # The floor pass has to engage at step 0: a job that only fits with its activations
            # moved out would run out of memory during the first steps, before it ever ran. It
            # moves every eligible activation out in the forward and copies each back in the
            # backward, so what it produces runs correctly on its own.
            #
            # The planner runs at WARMUP because it only gives memory back, so deferring it is
            # safe -- the steps before it execute the fully-offloaded configuration, which is the
            # one that survives everywhere. Planning at step 0 was the bug: the floor it read came
            # from profiling the compiled forward graph, and work outside that graph is invisible
            # to it. DeepSpeed's tiled loss drives a nested autograd.backward from inside its own
            # forward, so none of that memory is counted. Measured on Qwen3-14B, the floor
            # overshoots the real peak with an untiled loss (seq2048 +4.4GB, seq3072 +3.1GB,
            # seq3584 +1.7GB) and undershoots as soon as tiling is on -- seq3072 goes from +3.1GB
            # to -1.9GB with nothing else changed, and at seq4096 it undershot by 3.1GB against
            # 2.3GB of real headroom, so every plan it made there died. By WARMUP the profile is
            # taken with the run's real memory in place.
            # Both entries rebuild from the captured graph, so the floor pass has to appear in each
            # of them. Listing only the planner at WARMUP rebuilds a forward with no offloads at
            # all, the planner finds nothing to trim, and every activation goes resident -- measured
            # as a 14.2GB jump in step peak the moment WARMUP was reached. This is the same reason
            # the default schedule repeats add_z3_gather_release below.
            schedule.append((0, [zero3_compile.add_z3_gather_release, offload_activation.offload_activation_floor]))
            schedule.append((WARMUP, [
                zero3_compile.add_z3_gather_release, offload_activation.offload_activation_floor,
                offload_activation.offload_activation
            ]))
        else:
            schedule.append((0, [zero3_compile.add_z3_gather_release]))
            schedule.append(
                (WARMUP,
                 [zero3_compile.add_z3_gather_release, prefetch.schedule_prefetch, selective_gather.selective_gather]))

    init_schedule(schedule)

    if use_opt:

        def set_grad_buffer(_is_gradient_accumulation_boundary):
            for i, sub_group in enumerate(optimizer.fp16_groups):
                optimizer.averaged_gradients[i] = [
                    optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[param.ds_id]
                    if param.requires_grad else torch.zeros_like(param.ds_tensor) for param in sub_group
                ]

        add_pre_backward_hook(set_grad_buffer)

        # offloading opt states need additional setup
        from .passes.offload_adam_states import (move_opt_states, move_opt_states_sync, offload_adam_states_for_init,
                                                 init_offload_opt_states)
        for _, passes in schedule:
            if move_opt_states in passes or move_opt_states_sync in passes or offload_adam_states_for_init in passes:
                init_offload_opt_states(optimizer, dc)

    engine._deepcompile_owned_frames = set()
    engine.launch_compile_passes = partial(launch_compile_passes, owned_frames=engine._deepcompile_owned_frames)

    patch_fake_tensor()
    torch._inductor.config.size_asserts = False

    previous_restore = getattr(engine, "_deepcompile_dynamo_config_restore", None)
    if previous_restore is not None:
        previous_restore()
        del engine._deepcompile_dynamo_config_restore
    restore_dynamo_config = _allow_dynamo_dynamic_parameter_shapes_for_z3(compile_kwargs)
    if restore_dynamo_config is not None:
        engine._deepcompile_dynamo_config_restore = restore_dynamo_config

    return make_backend(backend,
                        compile_config,
                        compile_kwargs=compile_kwargs,
                        owned_frames=engine._deepcompile_owned_frames)
