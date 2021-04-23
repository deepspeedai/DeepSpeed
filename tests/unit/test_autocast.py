import pytest
import torch
import deepspeed
from deepspeed.runtime.zero.linear import LinearModuleForZeroStage3


def _skip_autocast_test():
    try:
        from torch.cuda.amp import custom_fwd, custom_bwd
    except (ImportError, AttributeError) as exp:
        return True

    return False


@pytest.mark.parametrize('half_op', [False, True])
def test_missing_amp_autocast(tmpdir, half_op):
    hidden_dim = 4
    if half_op:
        input = torch.randn(hidden_dim).cuda().half()
        ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).cuda().half()
    else:
        input = torch.randn(hidden_dim).cuda()
        ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).cuda()

    output = ds_linear(input)
    assert output.dtype == ds_linear.weight.dtype


@pytest.mark.parametrize('half_op', [False, True])
def test_disable_autocast_linear(tmpdir, half_op):
    if _skip_autocast_test():
        pytest.skip("amp autocast is not availalbe")

    hidden_dim = 4
    if half_op:
        input = torch.randn(hidden_dim).cuda().half()
        ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).cuda().half()
    else:
        input = torch.randn(hidden_dim).cuda()
        ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).cuda()

    with torch.cuda.amp.autocast(False):
        output = ds_linear(input)
        assert output.dtype == ds_linear.weight.dtype


@pytest.mark.parametrize('half_input, half_weight',
                         [(False,
                           False),
                          (False,
                           True),
                          (True,
                           False),
                          (True,
                           True)])
def test_autocast_linear(tmpdir, half_input, half_weight):
    if _skip_autocast_test():
        pytest.skip("amp autocast is not availalbe")

    hidden_dim = 4
    input = torch.randn(hidden_dim).cuda()
    ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).cuda()

    if half_input:
        input = input.half()

    if half_weight:
        ds_linear = ds_linear.half()

    with torch.cuda.amp.autocast():
        output = ds_linear(input)
        assert output.dtype == torch.half
