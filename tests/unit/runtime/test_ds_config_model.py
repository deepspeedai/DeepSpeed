# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import os
import json
from pydantic import Field, ValidationError
from typing import List
from deepspeed.runtime import config as ds_config
from deepspeed.runtime.config_utils import DeepSpeedConfigModel


class SimpleConf(DeepSpeedConfigModel):
    param_1: int = 0
    param_2_old: str = Field(None, deprecated=True, new_param="param_2", new_param_fn=(lambda x: [x]))
    param_2: List[str] = None
    param_3: int = Field(0, alias="param_3_alias")


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


def test_config_base():
    config = SimpleConf(**{"param_1": 42})
    assert config.param_1 == 42


def test_config_base_deprecatedfield():
    config = SimpleConf(**{"param_2_old": "DS"})
    assert config.param_2 == ["DS"]


def test_config_base_aliasfield():
    config = SimpleConf(**{"param_3": 10})
    assert config.param_3 == 10

    config = SimpleConf(**{"param_3_alias": 10})
    assert config.param_3 == 10


@pytest.mark.parametrize("config_dict", [{"param_1": "DS"}, {"param_2": "DS"}, {"param_1_typo": 0}])
def test_config_base_literalfail(config_dict):
    with pytest.raises(ValidationError):
        config = SimpleConf(**config_dict)


def test_config_base_deprecatedfail():
    with pytest.raises(AssertionError):
        config = SimpleConf(**{"param_2": ["DS"], "param_2_old": "DS"})
