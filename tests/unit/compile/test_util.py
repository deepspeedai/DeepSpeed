# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed.compile.util as compile_util


def test_all_reduce_preserves_default_and_explicit_group_forwarding(monkeypatch):
    calls = []
    sentinel = object()

    def unexpected_is_initialized():
        raise AssertionError("all_reduce must not consult distributed initialization state")

    def record_all_reduce(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(compile_util.dist, "is_initialized", unexpected_is_initialized)
    monkeypatch.setattr(compile_util.dist, "all_reduce", record_all_reduce)

    tensor = object()
    op = object()
    process_group = object()

    assert compile_util.all_reduce(tensor, op) is sentinel
    assert compile_util.all_reduce(tensor, op, process_group) is sentinel
    assert calls == [((tensor, op), {}), ((tensor, op), {"group": process_group})]
