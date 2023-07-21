# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
import deepspeed
import pytest
import random
import numpy as np
import deepspeed.comm as dist
from unit.common import DistributedTest, DistributedFixture
from unit.megatron_model import get_megatron_version
from unit.megatron_model import MockGPT2ModelPipe as GPT2ModelPipe
from deepspeed.utils import RepeatingLoader
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.utils import required_torch_version

pytestmark = pytest.mark.skipif(not required_torch_version(min_version=1.5, max_version=1.13),
                                reason='Megatron-LM package requires Pytorch version >=1.5 and <=1.13')


def get_deepspeed_model(model):
    ds_config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Lamb",
            "params": {
                "lr": 0.00015
            }
        },
    }

    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config_dict)
    return model.to(get_accelerator().device_name())


def get_topology(mp, pp, world_size):
    assert world_size % (pp * mp) == 0
    dp = world_size // (pp * mp)

    from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
    topo = PipeModelDataParallelTopology(num_pp=pp, num_mp=mp, num_dp=dp)

    return topo


class ConfigurablePP(DistributedTest):

    @pytest.fixture(autouse=True)
    def reset_random(self, seed=1234):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)

    @pytest.fixture
    def inputs(self, bs=1, seq_len=1, hidden_size=128):
        hidden_states = torch.randn(bs, seq_len, hidden_size)
        attention_mask = torch.randint(low=0, high=2, size=(bs, seq_len), dtype=torch.bool)
        return (hidden_states, attention_mask)


class TestConfigurablePP(ConfigurablePP):
    mp_size = 2
    pp_size = 2
    world_size = 4  # mp_size * pp_size

    @pytest.mark.skip(reason="megatron-lm is currently broken so this test cannot be run.")
    def test_pp_basic(self, inputs, tmpdir):
        # basic test case, mp_size=2, pp_size=2, verify ckpt saving/loading.
        args_defaults = {
            'num_layers': 8,
            'hidden_size': 128,
            'num_attention_heads': 8,
            'max_position_embeddings': 128,
        }
        mp_size = self.mp_size
        pp_size = self.pp_size
        world_size = self.world_size

        topo = get_topology(mp_size, pp_size, world_size)
        gpt2_pipe_model = GPT2ModelPipe(num_layers=8,
                                        num_stages=pp_size,
                                        mp_size=mp_size,
                                        args_others=args_defaults,
                                        topo=topo)
        model = get_deepspeed_model(gpt2_pipe_model)

        tag = 'pp_basic'
        state_dict = {}
        state_dict['checkpoint_version'] = get_megatron_version()
        model.save_checkpoint(tmpdir, tag=tag, client_state=state_dict)

        if model.is_first_stage() or model.is_last_stage():
            loader = RepeatingLoader([(inputs[0], 0)])
            data_iter = iter(loader)
        else:
            data_iter = None

        baseline = model.eval_batch(data_iter=data_iter, compute_loss=False, reduce_output=None)

        dist.barrier()
        model.load_checkpoint(tmpdir, tag=tag, load_optimizer_states=False, load_lr_scheduler_states=False)
        dist.barrier()

        test = model.eval_batch(data_iter=data_iter, compute_loss=False, reduce_output=None)

        if test is not None:
            assert len(baseline) == len(test)
            # Compare outputs of each microbatch
            for mb in range(len(baseline)):
                for b, t in zip(baseline[mb], test[mb]):
                    if b.is_floating_point():  # don't compare masks
                        assert torch.allclose(
                            b, t,
                            atol=1e-07), f"Baseline output {baseline} is not equal to save-then-load output {test}"


# Fixture for defining the checkpoint path since all tests in
# TestConfigurableResizePP will use the same tmpdir
@pytest.fixture
def checkpoint_tag(mp_size, pp_size, mp_resize, pp_resize):
    return f"{mp_size}-{pp_size}-{mp_resize}-{pp_resize}"


# Base class for creating / saving model output for baseline models. This is
# not meant to be used directly as a fixture to any classes
class _baseline(DistributedFixture):
    world_size = None

    def run(self, inputs, class_tmpdir, checkpoint_tag, mp_size, pp_size):
        assert int(os.environ["WORLD_SIZE"]) == (pp_size *
                                                 mp_size), "world size does not match provided pp_size and mp_size"
        args_defaults = {
            'num_layers': 8,
            'hidden_size': 128,
            'num_attention_heads': 8,
            'max_position_embeddings': 128,
        }

        topo = get_topology(mp_size, pp_size, mp_size * pp_size)
        gpt2_pipe_model = GPT2ModelPipe(num_layers=8,
                                        num_stages=pp_size,
                                        mp_size=mp_size,
                                        args_others=args_defaults,
                                        topo=topo)
        model = get_deepspeed_model(gpt2_pipe_model)

        with torch.no_grad():
            inputs = [x.to(get_accelerator().device_name()) for x in inputs]
            if model.is_first_stage() or model.is_last_stage():
                loader = RepeatingLoader([(inputs[0], 0)])
                data_iter = iter(loader)
            else:
                data_iter = None

            baseline = model.eval_batch(data_iter=data_iter, compute_loss=False, reduce_output=None)

            if baseline is not None:
                # baseline should be [[hidden, True]]]
                assert len(baseline) == 1
                assert len(baseline[0]) == 1
                assert torch.is_tensor(baseline[0][0])
                save_path = os.path.join(class_tmpdir, f"output-{checkpoint_tag}.pt")
                torch.save(baseline[0][0].cpu(), save_path)

            state_dict = {}
            state_dict['checkpoint_version'] = get_megatron_version()
            model.save_checkpoint(class_tmpdir, tag=checkpoint_tag, client_state=state_dict)


# This may look odd, but there is a limitation with DistributedFixture that
# doesn't allow us to reuse a fixture with different worldsizes. This could be
# implemented in conftest.py::pytest_fixture_setup and common.py::DistributedFixture
class baseline_ws1(_baseline):
    world_size = 1


class baseline_ws2(_baseline):
    world_size = 2


class baseline_ws4(_baseline):
    world_size = 4


class TestConfigurableResizePP(ConfigurablePP):

    def _test(self, inputs, class_tmpdir, checkpoint_tag, mp_size, pp_size, mp_resize, pp_resize):
        args_defaults = {
            'num_layers': 8,
            'hidden_size': 128,
            'num_attention_heads': 8,
            'max_position_embeddings': 128,
        }

        topo = get_topology(mp_resize, pp_resize, mp_resize * pp_resize)
        gpt2_pipe_model = GPT2ModelPipe(num_layers=8,
                                        num_stages=pp_resize,
                                        mp_size=mp_resize,
                                        args_others=args_defaults,
                                        topo=topo)
        model = get_deepspeed_model(gpt2_pipe_model)

        with torch.no_grad():
            model.load_checkpoint(class_tmpdir,
                                  tag=checkpoint_tag,
                                  load_optimizer_states=False,
                                  load_lr_scheduler_states=False)
            inputs = [x.to(get_accelerator().device_name()) for x in inputs]
            if model.is_first_stage() or model.is_last_stage():
                loader = RepeatingLoader([(inputs[0], 0)])
                data_iter = iter(loader)
            else:
                data_iter = None

            test = model.eval_batch(data_iter=data_iter, compute_loss=False, reduce_output=None)

            if test is not None:
                # test should be [[hidden, True]]]
                assert len(test) == 1
                assert len(test[0]) == 1
                assert torch.is_tensor(test[0][0])
                test = test[0][0].cpu()
                load_path = os.path.join(class_tmpdir, f"output-{checkpoint_tag}.pt")
                baseline = torch.load(load_path)
                assert torch.allclose(
                    baseline, test,
                    atol=1e-03), f"Baseline output {baseline} is not equal to save-then-load output {test}"

    # These tests are divided by baseline model worldsize and test model worldsize
    @pytest.mark.world_size(1)
    @pytest.mark.parametrize("mp_size, pp_size, mp_resize, pp_resize", [(1, 2, 1, 1)])
    @pytest.mark.skip(reason="megatron-lm is currently broken so this test cannot be run.")
    def test_world_size_2to1(self, inputs, class_tmpdir, checkpoint_tag, baseline_ws2, mp_size, pp_size, mp_resize,
                             pp_resize):
        self._test(inputs, class_tmpdir, checkpoint_tag, mp_size, pp_size, mp_resize, pp_resize)

    @pytest.mark.world_size(1)
    @pytest.mark.parametrize("mp_size, pp_size, mp_resize, pp_resize", [(2, 2, 1, 1)])
    @pytest.mark.skip(reason="megatron-lm is currently broken so this test cannot be run.")
    def test_world_size_4to1(self, inputs, class_tmpdir, checkpoint_tag, baseline_ws4, mp_size, pp_size, mp_resize,
                             pp_resize):
        self._test(inputs, class_tmpdir, checkpoint_tag, mp_size, pp_size, mp_resize, pp_resize)

    @pytest.mark.world_size(2)
    @pytest.mark.parametrize("mp_size, pp_size, mp_resize, pp_resize", [(2, 2, 2, 1)])
    @pytest.mark.skip(reason="megatron-lm is currently broken so this test cannot be run.")
    def test_world_size_4to2(self, inputs, class_tmpdir, checkpoint_tag, baseline_ws4, mp_size, pp_size, mp_resize,
                             pp_resize):
        self._test(inputs, class_tmpdir, checkpoint_tag, mp_size, pp_size, mp_resize, pp_resize)

    @pytest.mark.world_size(4)
    @pytest.mark.parametrize("mp_size, pp_size, mp_resize, pp_resize", [(1, 1, 2, 2)])
    @pytest.mark.skip(reason="megatron-lm is currently broken so this test cannot be run.")
    def test_world_size_1to4(self, inputs, class_tmpdir, checkpoint_tag, baseline_ws1, mp_size, pp_size, mp_resize,
                             pp_resize):
        self._test(inputs, class_tmpdir, checkpoint_tag, mp_size, pp_size, mp_resize, pp_resize)

    @pytest.mark.world_size(4)
    @pytest.mark.parametrize("mp_size, pp_size, mp_resize, pp_resize", [(1, 2, 1, 4), (2, 1, 2, 2)])
    @pytest.mark.skip(reason="megatron-lm is currently broken so this test cannot be run.")
    def test_world_size_2to4(self, inputs, class_tmpdir, checkpoint_tag, baseline_ws2, mp_size, pp_size, mp_resize,
                             pp_resize):
        self._test(inputs, class_tmpdir, checkpoint_tag, mp_size, pp_size, mp_resize, pp_resize)
