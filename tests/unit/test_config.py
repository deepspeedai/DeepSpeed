# A test on its own
import torch
import pytest
from common import distributed_test
import torch.distributed as dist

# A test on its own
import deepspeed
from deepspeed.pt.deepspeed_config import DeepSpeedConfig


def test_cuda():
    assert (torch.cuda.is_available())


def test_check_version():
    assert hasattr(deepspeed, "__git_hash__")
    assert hasattr(deepspeed, "__git_branch__")
    assert hasattr(deepspeed, "__version__")


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
def test_batch_config(num_ranks, batch, micro_batch, gas, success):
    @distributed_test(world_size=2)
    def _test_batch_config(num_ranks, batch, micro_batch, gas, success):
        assert dist.get_world_size() == num_ranks, \
        'The test assumes a world size of f{num_ranks}'

        ds_batch_config = 'tests/unit/ds_batch_config.json'
        ds_config = DeepSpeedConfig(ds_batch_config)

        #test cases when all parameters are provided
        status = _run_batch_config(ds_config,
                                   train_batch=batch,
                                   micro_batch=micro_batch,
                                   gas=gas)
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

    """Run batch config test """
    _test_batch_config(num_ranks, batch, micro_batch, gas, success)
