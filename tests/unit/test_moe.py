import torch
import deepspeed
import pytest
from .common import distributed_test
from .simple_model import SimplePRMoEModel, args_from_dict, SimpleMoEModel, sequence_dataloader
from .util import required_torch_version

try:
    from apex import amp  # noqa: F401
    _amp_available = True
except ImportError:
    _amp_available = False
amp_available = pytest.mark.skip(_amp_available, reason="apex/amp is not installed")


@pytest.mark.parametrize("ep_size, use_residual",
                         [(2,
                           True),
                          (2,
                           False),
                          (4,
                           True),
                          (4,
                           False)])
def test_moe(tmpdir, ep_size, use_residual):
    if not required_torch_version():
        pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

    config_dict = {
        "train_batch_size": 8,
        "steps_per_print": 1,
        "fp16": {
            "enabled": True
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 16

    @distributed_test(world_size=[4])
    def _test_moe(args, hidden_dim, ep_size, use_residual):
        # E+D -- ep_size = 2
        # E only -- ep_size = 4
        model = SimpleMoEModel(hidden_dim, ep_size=ep_size, use_residual=use_residual)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False)
        #dist_init_required=False -- parameterize to True/False?

        data_loader = sequence_dataloader(model=model,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=model.device)

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_moe(args=args,
              hidden_dim=hidden_dim,
              ep_size=ep_size,
              use_residual=use_residual)


@pytest.mark.parametrize("ep_size, use_residual", [(2, True), (2, False)])
def test_pr_moe(tmpdir, ep_size, use_residual):
    if not required_torch_version():
        pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

    config_dict = {
        "train_batch_size": 8,
        "steps_per_print": 1,
        "fp16": {
            "enabled": True
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 16

    @distributed_test(world_size=[4])
    def _test_moe(args, hidden_dim, ep_size, use_residual):
        # E+D -- ep_size = 2
        # E only -- ep_size = 4

        model = SimplePRMoEModel(hidden_dim, ep_size=ep_size, use_residual=use_residual)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False)

        data_loader = sequence_dataloader(model=model,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=model.device)

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_moe(args=args,
              hidden_dim=hidden_dim,
              ep_size=ep_size,
              use_residual=use_residual)
