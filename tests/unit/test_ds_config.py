import pytest
import os
import json
from deepspeed.runtime import config as ds_config


def test_only_required_fields(tmpdir):
    '''Ensure that config containing only the required fields is accepted. '''
    cfg_json = tmpdir.mkdir('ds_config_unit_test').join('minimal.json')

    with open(cfg_json, 'w') as f:
        required_fields = {'train_batch_size': 64}
        json.dump(required_fields, f)

    run_cfg = ds_config.DeepSpeedConfig(cfg_json)
    assert run_cfg is not None
    assert run_cfg.train_batch_size == 64
    assert run_cfg.train_micro_batch_size_per_gpu == 64
    assert run_cfg.gradient_accumulation_steps == 1


def test_config_duplicate_key(tmpdir):
    config_dict = '''
    {
        "train_batch_size": 24,
        "train_batch_size": 24,
    }
    '''
    config_path = os.path.join(tmpdir, 'temp_config.json')

    with open(config_path, 'w') as jf:
        jf.write("%s" % config_dict)

    with pytest.raises(ValueError):
        run_cfg = ds_config.DeepSpeedConfig(config_path)
