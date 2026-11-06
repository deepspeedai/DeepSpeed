# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""``zero_optimization.optimizer_state_swap``: the ZeRO-1/2 FP32 state in pinned host memory outside the step."""

import contextlib
import types

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero import optimizer_state_swap as swap_mod
from deepspeed.runtime.zero.config import DeepSpeedZeroOptimizerStateSwapConfig
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from unit.common import DistributedTest

# ---------------------------------------------------------------------------------------------------------------------
# Fakes


def _resident(tensor) -> bool:
    return tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


class _InnerAdam:
    """Adam on the param groups it is handed, its state created at the first step; it reads only resident tensors."""

    def __init__(self, masters, owner):
        self.param_groups = [{"params": [master]} for master in masters]
        self.state = {}
        self.owner = owner

    def step(self):
        masters = self.owner.single_partition_of_fp32_groups
        self.owner.resident_during.append({i for i, m in enumerate(masters) if _resident(m)})
        for group in self.param_groups:
            master = group["params"][0]
            assert _resident(master)
            state = self.state.setdefault(master, {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros_like(master),
                "exp_avg_sq": torch.zeros_like(master)
            })
            assert _resident(state["exp_avg"]) and _resident(state["exp_avg_sq"])
            state["step"] += 1
            grad = master.grad
            state["exp_avg"].mul_(0.9).add_(grad, alpha=0.1)
            state["exp_avg_sq"].mul_(0.99).add_(grad * grad, alpha=0.01)
            master.sub_(state["exp_avg"] / (state["exp_avg_sq"].sqrt() + 1e-8), alpha=1e-2)


class _FakeZero:
    """The ZeRO-1/2 optimizer surface the swap touches, with DeepSpeedZeroOptimizer's own hook sites."""

    overlapping_partition_gradients_reduce_epilogue = (
        DeepSpeedZeroOptimizer.overlapping_partition_gradients_reduce_epilogue)
    step = DeepSpeedZeroOptimizer.step
    _optimizer_step = DeepSpeedZeroOptimizer._optimizer_step
    optimizer_state_resident = DeepSpeedZeroOptimizer.optimizer_state_resident
    _check_optimizer_state_resident = DeepSpeedZeroOptimizer._check_optimizer_state_resident
    _optimizer_state_hold = DeepSpeedZeroOptimizer._optimizer_state_hold
    _relink_optimizer_state = DeepSpeedZeroOptimizer._relink_optimizer_state
    _optimizer_state_swap = None
    _optimizer_state_hold_depth = 0
    _optimizer_state_hold_reselect = False
    _optimizer_state_masters_on_host = False

    def __init__(self, sizes, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.single_partition_of_fp32_groups = [torch.randn(n, generator=g) for n in sizes]
        self.bit16 = [torch.zeros(n) for n in sizes]
        self.bit16_groups = [[torch.zeros(n // 2), torch.zeros(n - n // 2)] for n in sizes]
        self.optimizer = _InnerAdam(self.single_partition_of_fp32_groups, self)
        self.torch_autocast_gradscaler = None
        self.zenflow = False
        self.partition_gradients = True
        self.cpu_offload = False
        self.boundary = True
        self.steps = 0
        self.resident_during = []
        self._hp_optimizer_states_linked = True
        self._hp_optimizer_states_linked_groups = set()

    def is_gradient_accumulation_boundary(self):
        return self.boundary

    def independent_gradient_partition_epilogue(self):
        pass

    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
        pass

    def _lazy_init_hp_params_optimizer_state(self):
        pass

    def _step(self, closure=None):
        """Per group, as DeepSpeed's step: the FP32 gradient, the group's optimizer step, then the BF16 copy."""
        self.steps += 1
        for i, master in enumerate(self.single_partition_of_fp32_groups):
            master.grad = torch.full_like(master, 0.1 * self.steps) + 0.01 * i
            self._optimizer_step(i)
            assert _resident(master)  # the BF16 copy reads the master after its step
            self.bit16[i].copy_(master)
            master.grad = None
        return "stepped"


def _install(opt, fraction=1.0, **config):
    streams = _HostStreams()
    swap = swap_mod.build(opt, DeepSpeedZeroOptimizerStateSwapConfig(fraction=fraction, **config), streams)
    opt._optimizer_state_swap = swap
    swap.install()
    return swap, streams


def _cycle(opt):
    """One training step: the backward epilogue at the accumulation boundary, then the step."""
    opt.boundary = True
    opt.overlapping_partition_gradients_reduce_epilogue()
    return opt.step()


def _snapshot(opt):
    tensors = list(opt.single_partition_of_fp32_groups)
    for master in opt.single_partition_of_fp32_groups:
        tensors += [v for k, v in opt.optimizer.state.get(master, {}).items() if k != "step"]
    return [t.clone() for t in tensors]


# ---------------------------------------------------------------------------------------------------------------------
# The whole-state swap


def test_the_step_evicts_and_frees_the_storage_then_the_epilogue_fetches_it_back():
    opt = _FakeZero([6, 10])
    swap, _ = _install(opt)
    assert _cycle(opt) == "stepped"  # the first step runs on fetched tensors and creates the state
    assert not swap.resident
    tensors = swap_mod.state_tensors(opt)
    assert len(tensors) == 6 and all(t.untyped_storage().nbytes() == 0 for t in tensors)
    assert swap.bytes == sum(t.numel() * t.element_size() for t in tensors)
    saved = [swap.host[id(t)].clone() for t in tensors]

    opt.boundary = True
    opt.overlapping_partition_gradients_reduce_epilogue()
    assert swap.resident and all(_resident(t) for t in tensors)
    assert all(torch.equal(t, s) for t, s in zip(tensors, saved))


# ---------------------------------------------------------------------------------------------------------------------
# The streamed step


def test_the_streamed_step_holds_one_group_and_the_next_and_matches_the_resident_step():
    reference, streamed = _FakeZero([4, 6, 4, 6]), _FakeZero([4, 6, 4, 6])
    stream, _ = _install(streamed, stream=True)
    kept = stream.kept
    assert all(_resident(m) == (i in kept) for i, m in enumerate(streamed.single_partition_of_fp32_groups))
    for _ in range(3):
        reference.step()
        streamed.step()
        swapped = [i for i in range(4) if i not in kept]
        assert all(not _resident(t) for i in swapped for t in stream.tensors(i))
        assert all(_resident(t) for i in kept for t in stream.tensors(i))
    for during in streamed.resident_during:  # at most the stepping group, the prefetched next, and the kept ones
        assert len(during - kept) <= 2
    assert all(torch.equal(a, b) for a, b in zip(reference.bit16, streamed.bit16))
    with streamed.optimizer_state_resident():
        assert all(torch.equal(a, b) for a, b in zip(_snapshot(reference), _snapshot(streamed)))


def test_the_streamed_step_skips_the_initial_fp32_zero_gradients_where_they_do_nothing():
    stream = DeepSpeedZeroOptimizerStateSwapConfig(fraction=1, stream=True)
    whole = DeepSpeedZeroOptimizerStateSwapConfig(fraction=1)
    adam = torch.optim.AdamW([torch.zeros(1, requires_grad=True)])
    adagrad = torch.optim.Adagrad([torch.zeros(1, requires_grad=True)])
    assert swap_mod.skips_initial_zero_grads(stream, adam, cpu_offload=False)
    assert not swap_mod.skips_initial_zero_grads(whole, adam, cpu_offload=False)
    assert not swap_mod.skips_initial_zero_grads(None, adam, cpu_offload=False)
    assert not swap_mod.skips_initial_zero_grads(stream, adagrad, cpu_offload=False)  # Adagrad's state is built there


# ---------------------------------------------------------------------------------------------------------------------
# Through deepspeed.initialize, on one rank


class _HostStreams:
    """The CPU accelerator has no streams or events: the copies run synchronously, so nothing needs ordering."""

    pin_memory = False

    def new_stream(self, device):
        return types.SimpleNamespace(wait_stream=lambda other: None, wait_event=lambda event: None)

    def current_stream(self):
        return self.new_stream(None)

    def use(self, stream):
        return contextlib.nullcontext()

    def new_event(self):
        return types.SimpleNamespace(record=lambda stream: None, synchronize=lambda: None)

    def record_stream(self, tensor, stream):
        pass

    def synchronize(self, event):
        pass

    def synchronize_stream(self, stream):
        pass


@contextlib.contextmanager
def _device_streams():
    on_device = get_accelerator().device_name() != "cpu"
    original = swap_mod.AcceleratorStreams
    if not on_device:
        swap_mod.AcceleratorStreams = _HostStreams
    try:
        yield
    finally:
        swap_mod.AcceleratorStreams = original


SPLIT_GIB = 64 * 256 * 4 / (1 << 30)  # one slice per weight matrix of the test model


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 64))


class _HostAdamW(torch.optim.AdamW):
    """AdamW for the CPU accelerator (no FusedAdam) that copies its loaded state, as a GPU load would."""

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for state in self.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.clone()


def _engine(model, swap=None, gas=2):
    zero = {"stage": 2, "overlap_comm": True}
    if swap is not None:
        zero["optimizer_state_swap"] = swap
    config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": gas,
        "gradient_clipping": 1.0,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": zero,
    }
    params = list(model.parameters())
    groups = [{"params": params}]
    optimizer = None
    if get_accelerator().device_name() == "cpu":  # no FusedAdam there
        optimizer = _HostAdamW(groups, lr=1e-3, weight_decay=0.0)
        config["zero_allow_untested_optimizer"] = True
    else:
        config["optimizer"] = {"type": "AdamW", "params": {"lr": 1e-3, "weight_decay": 0.0}}
    engine, *_ = deepspeed.initialize(model=model, model_parameters=groups, optimizer=optimizer, config=config)
    return engine


def _train(engine, steps, gas=2, seed=1):
    gen = torch.Generator().manual_seed(seed)
    off_device = []
    for _ in range(steps):
        for _ in range(gas):
            x = torch.randn(4, 64, generator=gen).to(engine.device, torch.bfloat16)
            engine.backward(engine(x).float().pow(2).mean())
            engine.step()
        get_accelerator().synchronize()
        tensors = swap_mod.state_tensors(engine.optimizer)
        off_device.append(sum(t.untyped_storage().nbytes() == 0 for t in tensors))
    return off_device


def _weights(engine):
    return [p.detach().float().cpu().clone() for p in engine.module.parameters()]


class TestOptimizerStateSwapEngine(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("swap", [{"fraction": 0.5}])
    def test_a_checkpoint_saves_and_restores_the_swapped_state(self, swap, tmpdir):
        with _device_streams():
            reference = _engine(_model(), swap=dict(swap))
            _train(reference, 2)
            reference.save_checkpoint(str(tmpdir), tag="t")
            _train(reference, 2, seed=2)
            restored = _engine(_model(), swap=dict(swap))
            restored.load_checkpoint(str(tmpdir), tag="t")
            off_device = _train(restored, 2, seed=2)
        assert all(n > 0 for n in off_device), off_device
        for a, b in zip(_weights(reference), _weights(restored)):
            assert torch.equal(a, b), (swap, (a - b).abs().max())

    def test_every_slice_links_its_optimizer_state_fragments(self):
        from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_optimizer_state

        with _device_streams():
            engines = []
            for swap in (None, STREAM):
                model = _model()
                engines.append(_client_engine(model, [{"params": list(model.parameters())}], swap=swap))
                _train(engines[-1], 2)
            reference, streamed = engines
            with streamed.optimizer.optimizer_state_resident():
                for p, q in zip(reference.module.parameters(), streamed.module.parameters()):
                    for key in ("exp_avg", "exp_avg_sq"):
                        assert torch.equal(safe_get_full_optimizer_state(p, key),
                                           safe_get_full_optimizer_state(q, key))
                    assert torch.equal(safe_get_full_fp32_param(p), safe_get_full_fp32_param(q))


def _client_engine(model, groups, swap=None, gas=2):
    """A client optimizer over ``groups``, without gradient clipping."""
    zero = {"stage": 2, "overlap_comm": True}
    if swap is not None:
        zero["optimizer_state_swap"] = swap
    config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": gas,
        "gradient_clipping": 0.0,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": zero,
        "zero_allow_untested_optimizer": True,
    }
    optimizer_class = _HostAdamW if get_accelerator().device_name() == "cpu" else torch.optim.AdamW
    optimizer = optimizer_class(groups, lr=1e-3, weight_decay=0.0)
    engine, *_ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
    return engine


STREAM = {"fraction": 1.0, "stream": True, "group_gib": SPLIT_GIB}
