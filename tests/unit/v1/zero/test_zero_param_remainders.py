# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""bf16.store_param_remainders: ZeRO-1/2 fp32 master weights stored as the bf16 weights plus int16 remainders."""

import os

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.op_builder import FusedAdamBuilder
from deepspeed.utils import safe_get_full_fp32_param, safe_set_full_fp32_param
from deepspeed.utils.param_remainders import rebuild_fp32_master, split_fp32_master

from unit.common import DistributedTest
from unit.simple_model import SimpleModel
from unit.util import bf16_required_version_check

HIDDEN_DIM = 64


def _bits(t):
    return t.contiguous().view(torch.int32)


def _is_tie(master):
    # The lower 16 bits are exactly half a bf16 step: the only values whose rounding differs from round-to-even.
    return (_bits(master) & 0xFFFF) == 0x8000


class TestParamRemainderLayout:

    def test_split_and_rebuild_are_exact_and_round_to_nearest(self):
        # Catches a lost or misplaced bit, a wrong sign in the rounding correction, and a NaN master whose bf16
        # weight turns into a zero or an infinity.
        generator = torch.Generator().manual_seed(0)
        random_bits = torch.randint(-2**31, 2**31 - 1, (1 << 18, ), generator=generator, dtype=torch.int64)
        edge_bits = torch.tensor([
            0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7F7FFFFF, 0x7F7F8000, 0x00008000, 0x80008000, 0x3F808000,
            0x3F818000, 0xBF808000, 0x00000001, 0x007FFFFF, 0x7FC00000, 0x7FFF8000, 0xFFFF8001
        ],
                                 dtype=torch.int64)
        master = torch.cat([random_bits, edge_bits]).to(torch.int32).view(torch.float32)
        bf16 = torch.empty(master.shape, dtype=torch.bfloat16)
        remainder = torch.empty(master.shape, dtype=torch.int16)

        split_fp32_master(master, bf16, remainder)
        rebuilt = rebuild_fp32_master(bf16, remainder)

        nan = torch.isnan(master)
        assert torch.equal(_bits(rebuilt)[~nan], _bits(master)[~nan])
        assert torch.isnan(rebuilt[nan]).all() and torch.isnan(bf16[nan].float()).all()
        # Away from ties the bf16 half is PyTorch's round-to-nearest-even cast; at ties it rounds away from zero.
        tie = _is_tie(master) & ~nan
        assert tie.any()
        rounded = master.to(torch.bfloat16)
        assert torch.equal(bf16[~tie & ~nan].view(torch.int16), rounded[~tie & ~nan].view(torch.int16))
        assert (bf16[tie].float().abs() >= master[tie].abs()).all()


@pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME], reason="FusedAdam is not compatible")
class TestFusedAdamParamRemainders(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("adam_w_mode", [True, False])
    @pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
    def test_matches_the_fp32_master_kernel(self, grad_dtype, adam_w_mode):
        # Catches a remainder update whose Adam math, rounding or tail handling differs from FusedAdam's fp32 path:
        # the masters and both moments must stay equal bit for bit over several steps (in L2 mode, within a few fp32
        # rounding steps; see below).
        device = get_accelerator().current_device_name()
        torch.manual_seed(0)
        numel = (1 << 17) + 13
        master = torch.randn(numel, device=device) * 0.1
        hyper = dict(lr=1e-2, weight_decay=0.1, adam_w_mode=adam_w_mode)

        reference = torch.nn.Parameter(master.clone())
        reference_optimizer = FusedAdam([reference], **hyper)

        bf16 = torch.nn.Parameter(torch.empty(numel, dtype=torch.bfloat16, device=device))
        remainder = torch.empty(numel, dtype=torch.int16, device=device)
        split_fp32_master(master, bf16.data, remainder)
        optimizer = FusedAdam([bf16], **hyper)

        for _ in range(3):
            grad = torch.randn(numel, device=device).to(grad_dtype)
            reference.grad = grad.float()
            reference_optimizer.step()
            optimizer.step_param_remainders(optimizer.param_groups[0], bf16, grad, remainder)

        rebuilt = rebuild_fp32_master(bf16.data, remainder)
        pairs = [(rebuilt, reference.data)]
        pairs += [(optimizer.state[bf16][key], reference_optimizer.state[reference][key])
                  for key in ("exp_avg", "exp_avg_sq")]
        for actual, expected in pairs:
            if adam_w_mode:
                # AdamW, FusedAdam's default: both kernels compile the shared update the same way, bit for bit.
                assert torch.equal(_bits(actual), _bits(expected))
            else:
                # L2 mode first adds the decay to the gradient; nvcc fuses that sum into the moment updates
                # differently in the two kernels, so values can differ by a few fp32 rounding steps.
                atol = 4 * torch.finfo(torch.float32).eps * expected.abs().max().item()
                torch.testing.assert_close(actual, expected, rtol=0, atol=atol)


def _config(zero_stage, grad_accum_dtype, remainders):
    return {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 1000,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-2,
                "weight_decay": 0.01
            }
        },
        "bf16": {
            "enabled": True,
            "store_param_remainders": remainders
        },
        "data_types": {
            "grad_accum_dtype": grad_accum_dtype
        },
        "zero_optimization": {
            "stage": zero_stage
        },
    }


def _engine(config, hidden_dim=HIDDEN_DIM, seed=123):
    torch.manual_seed(seed)
    model = SimpleModel(hidden_dim, nlayers=2)
    engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
    return engine


def _batches(engine, count, seed=7):
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(count):
        x = torch.randn(2, HIDDEN_DIM, generator=generator).to(engine.device, torch.bfloat16)
        y = torch.randint(0, HIDDEN_DIM, (2, ), generator=generator).to(engine.device)
        batches.append((x, y))
    return batches


def _train(engine, batches):
    losses = []
    for x, y in batches:
        loss = engine(x, y)
        engine.backward(loss)
        engine.step()
        losses.append(loss.float().item())
    return losses


def _masters(engine):
    return {name: safe_get_full_fp32_param(param).clone() for name, param in engine.module.named_parameters()}


def _bf16_weights(engine):
    return {name: param.detach().clone() for name, param in engine.module.named_parameters()}


@pytest.mark.parametrize("zero_stage,grad_accum_dtype", [(1, "bf16"), (2, "fp32"), (2, "bf16")])
class TestZeroParamRemainders(DistributedTest):
    world_size = 2

    def _skip_without_bf16(self):
        if not bf16_required_version_check():
            pytest.skip("bf16 needs torch >= 1.10, NCCL >= 2.10.3, CUDA >= 11.0 and hardware support")

    def test_first_step_matches_fp32_master(self, zero_stage, grad_accum_dtype):
        # Catches a remainder step that updates the master differently from an fp32 copy, and bf16 weights that are
        # not the updated masters rounded to nearest.
        self._skip_without_bf16()
        baseline = _engine(_config(zero_stage, grad_accum_dtype, remainders=False))
        _train(baseline, _batches(baseline, 1))
        remainders = _engine(_config(zero_stage, grad_accum_dtype, remainders=True))
        _train(remainders, _batches(remainders, 1))

        expected_masters = _masters(baseline)
        masters = _masters(remainders)
        weights = _bf16_weights(remainders)
        for name, expected in expected_masters.items():
            assert torch.equal(_bits(masters[name]), _bits(expected)), name
            tie = _is_tie(expected)
            rounded = expected.to(torch.bfloat16)
            assert torch.equal(weights[name][~tie], rounded[~tie]), name
            assert (weights[name][tie].float().abs() >= expected[tie].abs()).all(), name

    def test_training_tracks_fp32_master(self, zero_stage, grad_accum_dtype):
        # Catches drift that a single step hides, such as moments or step counts that are not carried between steps.
        # Only bf16 weights at exact ties differ from the fp32-master run, so the runs stay close but need not match.
        self._skip_without_bf16()
        baseline = _engine(_config(zero_stage, grad_accum_dtype, remainders=False))
        baseline_losses = _train(baseline, _batches(baseline, 6))
        remainders = _engine(_config(zero_stage, grad_accum_dtype, remainders=True))
        losses = _train(remainders, _batches(remainders, 6))

        assert losses == pytest.approx(baseline_losses, rel=1e-3)
        expected_masters = _masters(baseline)
        for name, master in _masters(remainders).items():
            assert torch.allclose(master, expected_masters[name], rtol=1e-3, atol=1e-5), name

    def test_checkpoint_round_trip(self, tmpdir, zero_stage, grad_accum_dtype):
        # Catches masters that a checkpoint does not save or restore exactly, a resumed run that diverges from the
        # uninterrupted one, and a checkpoint format that differs from the fp32-master one in either direction.
        self._skip_without_bf16()
        config = _config(zero_stage, grad_accum_dtype, remainders=True)
        engine = _engine(config)
        batches = _batches(engine, 4)
        _train(engine, batches[:2])
        engine.save_checkpoint(tmpdir)
        saved_masters = _masters(engine)
        continued_losses = _train(engine, batches[2:])
        continued_masters = _masters(engine)

        resumed = _engine(config, seed=999)
        resumed.load_checkpoint(tmpdir)
        for name, master in _masters(resumed).items():
            assert torch.equal(_bits(master), _bits(saved_masters[name])), name
        assert _train(resumed, batches[2:]) == continued_losses
        for name, master in _masters(resumed).items():
            assert torch.equal(_bits(master), _bits(continued_masters[name])), name

        fp32_master = _engine(_config(zero_stage, grad_accum_dtype, remainders=False), seed=999)
        fp32_master.load_checkpoint(tmpdir)
        for name, master in _masters(fp32_master).items():
            assert torch.equal(_bits(master), _bits(saved_masters[name])), name

        fp32_dir = os.path.join(tmpdir, "fp32_master")
        fp32_master.save_checkpoint(fp32_dir)
        from_fp32 = _engine(config, seed=999)
        from_fp32.load_checkpoint(fp32_dir)
        for name, master in _masters(from_fp32).items():
            assert torch.equal(_bits(master), _bits(saved_masters[name])), name

    def test_full_fp32_param_set_and_get(self, zero_stage, grad_accum_dtype):
        # Catches the fp32 parameter API reading only the bf16 half, or writing a value that is then rounded.
        self._skip_without_bf16()
        engine = _engine(_config(zero_stage, grad_accum_dtype, remainders=True))
        _train(engine, _batches(engine, 1))
        for name, param in engine.module.named_parameters():
            value = safe_get_full_fp32_param(param) * (1 + 2**-20) + 2**-30
            safe_set_full_fp32_param(param, value)
            assert torch.equal(_bits(safe_get_full_fp32_param(param)), _bits(value)), name

    def test_master_weights_take_two_bytes_per_parameter(self, zero_stage, grad_accum_dtype):
        # Catches a configuration that still allocates an fp32 master copy next to the remainders.
        self._skip_without_bf16()
        hidden_dim = 512
        get_accelerator().empty_cache()
        allocated = {}
        for remainders in (True, False):
            before = get_accelerator().memory_allocated()
            engine = _engine(_config(zero_stage, grad_accum_dtype, remainders), hidden_dim=hidden_dim)
            allocated[remainders] = get_accelerator().memory_allocated() - before
            local_numel = sum(p.numel() for p in engine.module.parameters()) // self.world_size
        saved = allocated[False] - allocated[True]
        # 2 bytes per parameter of this rank's partition, up to padding and allocator rounding.
        assert abs(saved - 2 * local_numel) <= 64 * 1024, (saved, 2 * local_numel)


class TestParamRemaindersRejected(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("change", [
        "zero_stage_3", "optimizer_offload", "bf16_master_weights_and_grads", "zero_stage_1_fp32_grad_accum",
        "client_torch_adamw"
    ])
    def test_unsupported_configurations_raise(self, change):
        # Catches a configuration that would silently keep (or lose) an fp32 master instead of the remainders.
        if not bf16_required_version_check():
            pytest.skip("bf16 needs torch >= 1.10, NCCL >= 2.10.3, CUDA >= 11.0 and hardware support")
        config = _config(2, "bf16", remainders=True)
        optimizer = None
        torch.manual_seed(0)
        model = SimpleModel(HIDDEN_DIM, nlayers=2)
        if change == "zero_stage_3":
            config["zero_optimization"]["stage"] = 3
        elif change == "optimizer_offload":
            config["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
        elif change == "bf16_master_weights_and_grads":
            config["bf16"]["bf16_master_weights_and_grads"] = True
        elif change == "zero_stage_1_fp32_grad_accum":
            config["zero_optimization"]["stage"] = 1
            config["data_types"]["grad_accum_dtype"] = "fp32"
        elif change == "client_torch_adamw":
            del config["optimizer"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # pytest.raises inside a DistributedTest worker hangs the test when nothing is raised; check by hand.
        error = None
        try:
            deepspeed.initialize(config=config, model=model, model_parameters=model.parameters(), optimizer=optimizer)
        except ValueError as raised:
            error = raised
        assert error is not None and "store_param_remainders" in str(error), error
