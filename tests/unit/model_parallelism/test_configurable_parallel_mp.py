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
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest, DistributedFixture
from unit.megatron_model import get_gpt2_model, get_megatron_version
from unit.util import required_minimum_torch_version, required_maximum_torch_version

pytestmark = pytest.mark.skipif(not required_minimum_torch_version(major_version=1, minor_version=5),
                                reason='Megatron-LM package requires Pytorch version 1.5 or above')
pytestmark = pytest.mark.skipif(not required_maximum_torch_version(major_version=1, minor_version=13),
                                reason='Megatron-LM package requires Pytorch version 1.13 or below')


# TODO: integrated testing of TP and ZeRO 1/2/3
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

    from megatron import mpu
    model, _, _, _ = deepspeed.initialize(model=model,
                                          mpu=mpu,
                                          model_parameters=model.parameters(),
                                          config=ds_config_dict)
    return model


class ConfigurableMP(DistributedTest):

    @pytest.fixture(autouse=True)
    def reset_random(self, seed=1234):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)

    @pytest.fixture
    def inputs(self, bs=1, seq_len=20):
        input_ids = torch.randint(low=0, high=1000, size=(bs, seq_len))
        position_ids = torch.randint(low=0, high=2, size=(bs, seq_len))
        attention_mask = torch.randint(low=0, high=2, size=(bs, seq_len), dtype=torch.bool)
        return [input_ids, position_ids, attention_mask]


class TestConfigurableMP(ConfigurableMP):

    @pytest.mark.world_size(1)
    @pytest.mark.skip(reason="megatron-lm is currently broken so this test cannot be run.")
    def test_gpt2_basic(self, tmpdir, inputs):
        args_defaults = {
            'num_layers': 2,
            'hidden_size': 128,
            'num_attention_heads': 8,
            'max_position_embeddings': 128,
        }

        model = get_gpt2_model(args_defaults)
        model = get_deepspeed_model(model)

        model.eval()
        device_name = get_accelerator().device_name()
        baseline = model(inputs[0].to(device_name), inputs[1].to(device_name), inputs[2].to(device_name))

        tag = 'mp_1'
        state_dict = {}
        state_dict['checkpoint_version'] = get_megatron_version()
        model.save_checkpoint(tmpdir, tag=tag, client_state=state_dict)
        dist.barrier()
        model.load_checkpoint(tmpdir, tag=tag, load_optimizer_states=False, load_lr_scheduler_states=False)

        test = model(inputs[0], inputs[1], inputs[2])
        assert torch.allclose(baseline, test,
                              atol=1e-07), f"Baseline output {baseline} is not equal to save-then-load output {test}"

    @pytest.mark.world_size(2)
    @pytest.mark.skip(reason="megatron-lm is currently broken so this test cannot be run.")
    def test_gpt2_mp2_no_resize(self, tmpdir, inputs):
        args_defaults = {
            'num_layers': 2,
            'hidden_size': 128,
            'num_attention_heads': 8,
            'max_position_embeddings': 128,
        }

        model = get_gpt2_model(args_defaults, mp_size=2)
        model = get_deepspeed_model(model)

        model.eval()

        device_name = get_accelerator().device_name()
        baseline = model(inputs[0].to(device_name), inputs[1].to(device_name), inputs[2].to(device_name))

        tag = 'mp_2'
        state_dict = {}
        state_dict['checkpoint_version'] = get_megatron_version()
        model.save_checkpoint(tmpdir, tag=tag, client_state=state_dict)
        dist.barrier()
        model.load_checkpoint(tmpdir, tag=tag, load_optimizer_states=False, load_lr_scheduler_states=False)

        device_name = get_accelerator().device_name()
        test = model(inputs[0].to(device_name), inputs[1].to(device_name), inputs[2].to(device_name))
        assert torch.allclose(baseline, test, rtol=1.0,
                              atol=1e-07), f"Baseline output {baseline} is not equal to save-then-load output {test}"


# This fixture provides the baseline model with mp=2 to TestConfigurableMPResize
class baseline_mp2(DistributedFixture):
    world_size = 2

    def run(self, inputs, class_tmpdir):
        args_defaults = {
            'num_layers': 2,
            'hidden_size': 128,
            'num_attention_heads': 8,
            'max_position_embeddings': 128,
        }

        model = get_gpt2_model(args_defaults, mp_size=self.world_size)
        model = get_deepspeed_model(model)

        model.eval()

        with torch.no_grad():
            device_name = get_accelerator().device_name()
            baseline = model(inputs[0].to(device_name), inputs[1].to(device_name), inputs[2].to(device_name))
            if dist.get_rank() == 0:
                save_path = os.path.join(class_tmpdir, "output.pt")
                torch.save(baseline.cpu(), save_path)

            state_dict = {}
            state_dict['checkpoint_version'] = get_megatron_version()
            model.save_checkpoint(class_tmpdir, client_state=state_dict)


class TestConfigurableResizeMP(ConfigurableMP):
    world_size = [1, 4]

    @pytest.mark.skip(reason="megatron-lm is currently broken so this test cannot be run.")
    def test(self, baseline_mp2, inputs, class_tmpdir):
        args_defaults = {
            'num_layers': 2,
            'hidden_size': 128,
            'num_attention_heads': 8,
            'max_position_embeddings': 128,
        }

        world_size = os.environ["WORLD_SIZE"]
        model = get_gpt2_model(args_defaults, mp_size=world_size)
        model = get_deepspeed_model(model)

        model.eval()

        with torch.no_grad():
            model.load_checkpoint(class_tmpdir, load_optimizer_states=False, load_lr_scheduler_states=False)
            device_name = get_accelerator().device_name()
            test = model(inputs[0].to(device_name), inputs[1].to(device_name), inputs[2].to(device_name))
            if dist.get_rank() == 0:
                load_path = os.path.join(class_tmpdir, "output.pt")
                baseline = torch.load(load_path)
                test = test.cpu()
                assert torch.allclose(
                    baseline, test,
                    atol=1e-03), f"Baseline output {baseline} is not equal to save-then-load output {test}"
