# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import deque

import torch

from deepspeed.compile import backend as backend_mod
from deepspeed.compile.backend import _get_fw_real_inputs
from deepspeed.compile.inductor import patch_create_aot_dispatcher_function
from deepspeed.compile.input_storage import InputStorage
from deepspeed.compile.patch_compiled_func import (clear_backward_inputs, get_backward_inputs, patch_compiled_func,
                                                   unpatch_compiled_func)


def test_forward_real_inputs_are_graph_local():
    local_inputs = (torch.nn.Parameter(torch.ones(2, dtype=torch.float32)), )
    storage = InputStorage()
    storage.put((torch.ones(1, dtype=torch.float32), ))

    selected = _get_fw_real_inputs(deque([local_inputs]), storage, graph_id=7)

    assert selected is local_inputs


def test_forward_real_inputs_fall_back_to_storage_when_local_queue_is_empty():
    storage = InputStorage()
    storage.put((torch.ones(3, dtype=torch.float32), ))

    selected = _get_fw_real_inputs(deque(), storage, graph_id=7)

    assert len(selected) == 1
    assert selected[0].shape == torch.Size([3])
    assert selected[0].dtype is torch.float32


def test_launch_compile_passes_clears_owned_compiled_backward_state(monkeypatch):

    class DummyDeepCompileHandle:

        def reset(self):
            pass

    clear_backward_inputs()
    backend_mod.frames_needing_bwd.clear()
    unpatch_compiled_func()
    original_autograd_function = torch.autograd.Function
    owner = object()
    owned_frames = {(owner, 17)}
    backend_mod.frames_needing_bwd.update(owned_frames)
    patch_compiled_func()
    get_backward_inputs()[next(iter(owned_frames))] = [(torch.ones(1), )]
    monkeypatch.setattr(backend_mod, "log_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_mod, "get_deepcompile_handle", lambda: DummyDeepCompileHandle())

    backend_mod.init_schedule([(0, [])])
    try:
        backend_mod.launch_compile_passes(0, owned_frames=owned_frames)

        assert owned_frames == set()
        assert backend_mod.frames_needing_bwd == set()
        assert get_backward_inputs() == {}
        assert torch.autograd.Function is original_autograd_function
    finally:
        backend_mod.frames_needing_bwd.clear()
        unpatch_compiled_func()


def test_unpatch_compiled_func_clears_backward_inputs():
    clear_backward_inputs()
    patch_compiled_func()
    try:
        get_backward_inputs()[(object(), 17)] = [(torch.ones(1), )]
        unpatch_compiled_func()
        assert get_backward_inputs() == {}
    finally:
        unpatch_compiled_func()


def _patch_aot_constructor():
    return patch_create_aot_dispatcher_function(graph_id=7,
                                                z3_partition=False,
                                                make_fw_graph=lambda gm, sample_inputs: gm.graph,
                                                make_bw_graph=lambda gm, sample_inputs: gm.graph,
                                                real_inputs=(torch.ones(1), ),
                                                param_indices=[],
                                                param_manager={},
                                                frame_id=0,
                                                frames_partitioned=set())


def test_inductor_aot_constructor_patch_is_restorable():
    from torch._dynamo.backends.common import AotAutograd

    original_init = AotAutograd.__init__
    restore = _patch_aot_constructor()
    try:
        assert AotAutograd.__init__ is not original_init
    finally:
        restore()

    assert AotAutograd.__init__ is original_init
    assert not hasattr(AotAutograd, "__original_init")


def test_older_aot_restore_does_not_clobber_newer_patch():
    from torch._dynamo.backends.common import AotAutograd

    original_init = AotAutograd.__init__
    restore_first = _patch_aot_constructor()
    restore_second = _patch_aot_constructor()
    newer_init = AotAutograd.__init__
    try:
        restore_first()
        assert AotAutograd.__init__ is newer_init
        assert hasattr(AotAutograd, "__original_init")
    finally:
        restore_second()

    assert AotAutograd.__init__ is original_init
    assert not hasattr(AotAutograd, "__original_init")
