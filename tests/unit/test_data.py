from deepspeed.utils import RepeatingLoader
import torch
import pytest
import deepspeed
from .common import distributed_test
from .simple_model import SimpleModel, args_from_dict, random_dataset


def test_repeating_loader():
    loader = [1, 2, 3]
    loader = RepeatingLoader(loader)

    for idx in range(50):
        assert next(loader) == 1
        assert next(loader) == 2
        assert next(loader) == 3


@pytest.mark.parametrize('train_batch_size, drop_last',
                         [(1,
                           True),
                          (4,
                           True),
                          (1,
                           False),
                          (4,
                           False)])
def test_dataloader_drop_last(tmpdir, train_batch_size, drop_last):
    config_dict = {
        "train_batch_size": train_batch_size,
        "dataloader_drop_last": drop_last,
        "steps_per_print": 1
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1])
    def _test_dataloader_drop_last(args, model, hidden_dim):
        optimizer = torch.optim.AdamW(params=model.parameters())
        #TODO: Figure out why this breaks with cuda device
        train_dataset = random_dataset(total_samples=50,
                                       hidden_dim=hidden_dim,
                                       device=torch.device('cpu'),
                                       dtype=torch.float32)
        model, _, training_dataloader, _ = deepspeed.initialize(args=args,
                                                                model=model,
                                                                training_data=train_dataset,
                                                                optimizer=optimizer)
        for n, batch in enumerate(training_dataloader):
            x = batch[0].to(torch.cuda.current_device())
            y = batch[1].to(torch.cuda.current_device())
            loss = model(x, y)
            model.backward(loss)
            model.step()

    _test_dataloader_drop_last(args=args, model=model, hidden_dim=hidden_dim)
