# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# A test on its own
import os
import pytest
import json
import hjson
import argparse

from deepspeed.runtime.zero.config import DeepSpeedZeroConfig
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest, get_test_path
from unit.simple_model import SimpleModel, create_config_from_dict, random_dataloader
import deepspeed.comm as dist

# A test on its own
import deepspeed
from deepspeed.runtime.config import DeepSpeedConfig, get_bfloat16_enabled


class TestBasicConfig(DistributedTest):
    world_size = 1

    def test_accelerator(self):
        assert (get_accelerator().is_available())

    def test_check_version(self):
        assert hasattr(deepspeed, "__git_hash__")
        assert hasattr(deepspeed, "__git_branch__")
        assert hasattr(deepspeed, "__version__")
        assert hasattr(deepspeed, "__version_major__")
        assert hasattr(deepspeed, "__version_minor__")
        assert hasattr(deepspeed, "__version_patch__")


@pytest.fixture
def base_config():
    config_dict = {
        "train_batch_size": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        },
    }
    return config_dict


def _run_batch_config(ds_config, train_batch=None, micro_batch=None, gas=None):
    ds_config.train_batch_size = train_batch
    ds_config.train_micro_batch_size_per_gpu = micro_batch
    ds_config.gradient_accumulation_steps = gas
    success = True
    try:
        ds_config._configure_train_batch_size()
    except AssertionError:
        success = False
    return success


def _batch_assert(status, ds_config, batch, micro_batch, gas, success):

    if not success:
        assert not status
        print("Failed but All is well")
        return

    assert ds_config.train_batch_size == batch
    assert ds_config.train_micro_batch_size_per_gpu == micro_batch
    assert ds_config.gradient_accumulation_steps == gas
    print("All is well")


#Tests different batch config provided in deepspeed json file
@pytest.mark.parametrize('num_ranks,batch,micro_batch,gas,success',
                         [(2,32,16,1,True),
                         (2,32,8,2,True),
                         (2,33,17,2,False),
                         (2,32,18,1,False)]) # yapf: disable
class TestBatchConfig(DistributedTest):
    world_size = 2

    def test(self, num_ranks, batch, micro_batch, gas, success):
        assert dist.get_world_size() == num_ranks, \
        f'The test assumes a world size of {num_ranks}'

        ds_batch_config = get_test_path('ds_batch_config.json')
        ds_config = DeepSpeedConfig(ds_batch_config)

        #test cases when all parameters are provided
        status = _run_batch_config(ds_config, train_batch=batch, micro_batch=micro_batch, gas=gas)
        _batch_assert(status, ds_config, batch, micro_batch, gas, success)

        #test cases when two out of three parameters are provided
        status = _run_batch_config(ds_config, train_batch=batch, micro_batch=micro_batch)
        _batch_assert(status, ds_config, batch, micro_batch, gas, success)

        if success:
            #when gas is provided with one more parameter
            status = _run_batch_config(ds_config, train_batch=batch, gas=gas)
            _batch_assert(status, ds_config, batch, micro_batch, gas, success)

            status = _run_batch_config(ds_config, micro_batch=micro_batch, gas=gas)
            _batch_assert(status, ds_config, batch, micro_batch, gas, success)

            #test the case when only micro_batch or train_batch is provided
            if gas == 1:
                status = _run_batch_config(ds_config, micro_batch=micro_batch)
                _batch_assert(status, ds_config, batch, micro_batch, gas, success)

                status = _run_batch_config(ds_config, train_batch=batch)
                _batch_assert(status, ds_config, batch, micro_batch, gas, success)
        else:
            #when only gas is provided
            status = _run_batch_config(ds_config, gas=gas)
            _batch_assert(status, ds_config, batch, micro_batch, gas, success)

            #when gas is provided with something else and gas does not divide batch
            if gas != 1:
                status = _run_batch_config(ds_config, train_batch=batch, gas=gas)
                _batch_assert(status, ds_config, batch, micro_batch, gas, success)


def test_temp_config_json(tmpdir):
    config_dict = {
        "train_batch_size": 1,
    }
    config_path = create_config_from_dict(tmpdir, config_dict)
    config_json = json.load(open(config_path, 'r'))
    assert 'train_batch_size' in config_json


@pytest.mark.parametrize("gather_weights_key",
                         ["stage3_gather_16bit_weights_on_model_save", "stage3_gather_fp16_weights_on_model_save"])
def test_gather_16bit_params_on_model_save(gather_weights_key):
    config_dict = {
        gather_weights_key: True,
    }
    config = DeepSpeedZeroConfig(**config_dict)

    assert config.gather_16bit_weights_on_model_save == True


@pytest.mark.parametrize("bf16_key", ["bf16", "bfloat16"])
def test_get_bfloat16_enabled(bf16_key):
    cfg = {
        bf16_key: {
            "enabled": True,
        },
    }
    assert get_bfloat16_enabled(cfg) == True


class TestConfigLoad(DistributedTest):
    world_size = 1

    def test_dict(self, base_config):
        if get_accelerator().is_fp16_supported():
            base_config["fp16"] = {"enabled": True}
        elif get_accelerator().is_bf16_supported():
            base_config["bf16"] = {"enabled": True}
        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=base_config, model=model, model_parameters=model.parameters())

    def test_json(self, base_config, tmpdir):
        if get_accelerator().is_fp16_supported():
            base_config["fp16"] = {"enabled": True}
        elif get_accelerator().is_bf16_supported():
            base_config["bf16"] = {"enabled": True}
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, 'w') as fp:
            json.dump(base_config, fp)
        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_path, model=model, model_parameters=model.parameters())

    def test_hjson(self, base_config, tmpdir):
        if get_accelerator().is_fp16_supported():
            base_config["fp16"] = {"enabled": True}
        elif get_accelerator().is_bf16_supported():
            base_config["bf16"] = {"enabled": True}
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, 'w') as fp:
            hjson.dump(base_config, fp)
        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_path, model=model, model_parameters=model.parameters())


class TestDeprecatedDeepScaleConfig(DistributedTest):
    world_size = 1

    def test(self, base_config, tmpdir):
        if get_accelerator().is_fp16_supported():
            base_config["fp16"] = {"enabled": True}
        elif get_accelerator().is_bf16_supported():
            base_config["bf16"] = {"enabled": True}
        config_path = create_config_from_dict(tmpdir, base_config)
        parser = argparse.ArgumentParser()
        args = parser.parse_args(args='')
        args.deepscale_config = config_path
        args.local_rank = 0

        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model, total_samples=5, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestDistInit(DistributedTest):
    world_size = 1

    def test(self, base_config):
        if get_accelerator().is_fp16_supported():
            base_config["fp16"] = {"enabled": True}
        elif get_accelerator().is_bf16_supported():
            base_config["bf16"] = {"enabled": True}
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=base_config,
                                              model=model,
                                              model_parameters=model.parameters(),
                                              dist_init_required=True)
        data_loader = random_dataloader(model=model, total_samples=5, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestInitNoOptimizer(DistributedTest):
    world_size = 1

    def test(self, base_config):
        if get_accelerator().is_fp16_supported():
            base_config["fp16"] = {"enabled": True}
        elif get_accelerator().is_bf16_supported():
            base_config["bf16"] = {"enabled": True}
        if get_accelerator().device_name() == "cpu":
            pytest.skip("This test timeout with CPU accelerator")
        del base_config["optimizer"]
        hidden_dim = 10

        model = SimpleModel(hidden_dim=hidden_dim)

        model, _, _, _ = deepspeed.initialize(config=base_config, model=model)
        data_loader = random_dataloader(model=model, total_samples=5, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            with pytest.raises(AssertionError):
                model.backward(loss)
            with pytest.raises(AssertionError):
                model.step()


class TestArgs(DistributedTest):
    world_size = 1

    def test_none_args(self, base_config):
        if get_accelerator().is_fp16_supported():
            base_config["fp16"] = {"enabled": True}
        elif get_accelerator().is_bf16_supported():
            base_config["bf16"] = {"enabled": True}
        model = SimpleModel(hidden_dim=10)
        model, _, _, _ = deepspeed.initialize(args=None, model=model, config=base_config)
        data_loader = random_dataloader(model=model, total_samples=5, hidden_dim=10, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])

    def test_no_args(self, base_config):
        if get_accelerator().is_fp16_supported():
            base_config["fp16"] = {"enabled": True}
        elif get_accelerator().is_bf16_supported():
            base_config["bf16"] = {"enabled": True}
        model = SimpleModel(hidden_dim=10)
        model, _, _, _ = deepspeed.initialize(model=model, config=base_config)
        data_loader = random_dataloader(model=model, total_samples=5, hidden_dim=10, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])


class TestNoModel(DistributedTest):
    world_size = 1

    def test(self, base_config):
        if get_accelerator().is_fp16_supported():
            base_config["fp16"] = {"enabled": True}
        elif get_accelerator().is_bf16_supported():
            base_config["bf16"] = {"enabled": True}
        model = SimpleModel(hidden_dim=10)
        with pytest.raises(AssertionError):
            model, _, _, _ = deepspeed.initialize(model=None, config=base_config)

        with pytest.raises(AssertionError):
            model, _, _, _ = deepspeed.initialize(model, config=base_config)
