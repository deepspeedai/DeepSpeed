import pytest
import deepspeed
from common import distributed_test
from deepspeed.git_version_info import version as ds_version
from simple_model import SimpleModel, SimpleOptimizer, random_dataloader, args_from_dict

base_ds_config = {
    "elasticity": {
        "enabled": True,
        "max_train_batch_size": 10000,
        "micro_batch_sizes": [8,
                              12,
                              16,
                              17],
        "min_gpus": 32,
        "max_gpus": 1500,
        "min_time": 20,
        "version": 0.1
    }
}


def test_basic_10k():
    ds_config = base_ds_config.copy()
    final_batch_size, valid_gpus = deepspeed.elasticity.compute_elastic_config(
        ds_config=ds_config,
        target_deepspeed_version=ds_version)

    for gpu_num in valid_gpus:
        assert final_batch_size % gpu_num == 0, f"Batch {final_batch_size} is not divisible by GPU count {gpu_num}"
        batch_per_gpu = final_batch_size // gpu_num
        found_valid_mbsize = False

        for mb in ds_config['elasticity']['micro_batch_sizes']:
            if batch_per_gpu % mb == 0:
                found_valid_mb = True
                break
        assert found_valid_mb, "No valid mb found"

    assert len(valid_gpus) == 23
    assert final_batch_size == 9792


def test_old_version():
    ds_config = base_ds_config.copy()
    with pytest.raises(deepspeed.elasticity.config.ElasticityError):
        final_batch_size, valid_gpus = deepspeed.elasticity.compute_elastic_config(
            ds_config=ds_config,
            target_deepspeed_version="0.2")


def test_disabled():
    ds_config = base_ds_config.copy()
    ds_config['elasticity']['enabled'] = False
    with pytest.raises(deepspeed.elasticity.config.ElasticityError):
        final_batch_size, valid_gpus = deepspeed.elasticity.compute_elastic_config(
            ds_config=ds_config,
            target_deepspeed_version=ds_version)


def test_valid_world_size():
    ds_config = base_ds_config.copy()
    final_batch_size, valid_gpus, mbsize = deepspeed.elasticity.compute_elastic_config(
            ds_config=ds_config,
            target_deepspeed_version=ds_version,
            world_size=64)
    assert mbsize == 17


def test_invalid_world_size():
    ds_config = base_ds_config.copy()
    with pytest.raises(deepspeed.elasticity.config.ElasticityIncompatibleWorldSize):
        final_batch_size, valid_gpus, mbsize = deepspeed.elasticity.compute_elastic_config(
            ds_config=ds_config,
            target_deepspeed_version=ds_version,
            world_size=128)


def test_future_elastic_version():
    ds_config = base_ds_config.copy()
    ds_config['elasticity']['version'] = '0.2'
    with pytest.raises(deepspeed.elasticity.config.ElasticityError):
        deepspeed.elasticity.compute_elastic_config(ds_config=ds_config,
                                                    target_deepspeed_version=ds_version)


def test_missing_max_batch():
    ds_config = base_ds_config.copy()
    del ds_config['elasticity']['max_train_batch_size']
    with pytest.raises(deepspeed.elasticity.config.ElasticityError):
        deepspeed.elasticity.compute_elastic_config(ds_config=ds_config,
                                                    target_deepspeed_version=ds_version)


def test_missing_micro_batch():
    ds_config = base_ds_config.copy()
    del ds_config['elasticity']['micro_batch_sizes']
    with pytest.raises(deepspeed.elasticity.config.ElasticityError):
        deepspeed.elasticity.compute_elastic_config(ds_config=ds_config,
                                                    target_deepspeed_version=ds_version)


def test_empty_config():
    ds_config = {"elasticity": {"enabled": True}}
    with pytest.raises(deepspeed.elasticity.config.ElasticityError):
        deepspeed.elasticity.compute_elastic_config(ds_config=ds_config,
                                                    target_deepspeed_version=ds_version)


@pytest.mark.parametrize('key, value',
                         [('micro_batch_sizes',
                           [1,
                            4,
                            -1,
                            2,
                            -10]),
                          ('min_gpus',
                           -1),
                          ('max_gpus',
                           -1),
                          ('micro_batch_sizes',
                           5),
                          ('micro_batch_sizes',
                           ['a',
                            None,
                            0.5]),
                          ('micro_batch_sizes',
                           [2,
                            0.5,
                            4])])
def test_invalid_config_values(key, value):
    ds_config = base_ds_config.copy()
    ds_config['elasticity'][key] = value
    with pytest.raises(deepspeed.elasticity.config.ElasticityError):
        deepspeed.elasticity.compute_elastic_config(ds_config=ds_config,
                                                    target_deepspeed_version=ds_version)


def test_proper_mbsz():
    ds_config = base_ds_config.copy()
    ds_config["elasticity"]["max_train_batch_size"] = 32
    ds_config["elasticity"]["micro_batch_sizes"] = [1, 2, 3, 7]
    ds_config["elasticity"]["min_gpus"] = 1
    final_batch_size, valid_gpus, mbsize = deepspeed.elasticity.compute_elastic_config(
        ds_config=ds_config,
        target_deepspeed_version=ds_version,
        world_size=7)
    assert mbsize == 3


def test_non_elastic_batch_params(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Lamb",
            "params": {
                "lr": 0.00015
            }
        },
        "gradient_clipping": 1.0,
        "elasticity": {
            "enabled": True,
            "max_train_batch_size": 4,
            "micro_batch_sizes": [1,
                                  2,
                                  3,
                                  4],
            "min_gpus": 1,
            "max_gpus": 4,
            "min_time": 20,
            "version": 0.1
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[1, 2])
    def _test_elastic(args, model, hidden_dim):
        with pytest.raises(deepspeed.elasticity.config.ElasticityError):
            model, _, _,_ = deepspeed.initialize(args=args,
                                                 model=model,
                                                 model_parameters=model.parameters())

    _test_elastic(args=args, model=model, hidden_dim=hidden_dim)


def test_non_elastic_batch_params_w_override(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Lamb",
            "params": {
                "lr": 0.00015
            }
        },
        "gradient_clipping": 1.0,
        "elasticity": {
            "enabled": True,
            "max_train_batch_size": 4,
            "micro_batch_sizes": [1,
                                  2,
                                  3,
                                  4],
            "min_gpus": 1,
            "max_gpus": 4,
            "min_time": 20,
            "version": 0.1,
            "ignore_non_elastic_batch_info": True
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[1, 2])
    def _test_elastic(args, model, hidden_dim):
        model, _, _,_ = deepspeed.initialize(args=args,
                                             model=model,
                                             model_parameters=model.parameters())

    _test_elastic(args=args, model=model, hidden_dim=hidden_dim)


def test_elastic_config_changed(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Lamb",
            "params": {
                "lr": 0.00015
            }
        },
        "gradient_clipping": 1.0,
        "elasticity": {
            "enabled": True,
            "max_train_batch_size": 4,
            "micro_batch_sizes": [1,
                                  2,
                                  3,
                                  4],
            "min_gpus": 1,
            "max_gpus": 4,
            "min_time": 20,
            "version": 0.1,
            "ignore_non_elastic_batch_info": True
        }
    }
    import json, os
    scheduler_elastic_config = config_dict.copy()
    scheduler_elastic_config["elasticity"]["max_train_batch_size"] = 27
    os.environ['DEEPSPEED_ELASTICITY_CONFIG'] = json.dumps(scheduler_elastic_config)
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[1, 2])
    def _test_elastic(args, model, hidden_dim):
        with pytest.raises(deepspeed.elasticity.config.ElasticityError):
            model, _, _,_ = deepspeed.initialize(args=args,
                                                 model=model,
                                                 model_parameters=model.parameters())

    _test_elastic(args=args, model=model, hidden_dim=hidden_dim)
