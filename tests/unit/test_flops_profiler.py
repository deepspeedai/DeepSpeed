import torch
import deepspeed
import deepspeed.runtime.utils as ds_utils
from deepspeed.profiling.flops_profiler import FlopsProfiler, get_model_profile
from simple_model import SimpleModel, SimpleOptimizer, random_dataloader, args_from_dict
from common import distributed_test


def test_flops_profiler_in_ds_trainning(tmpdir):
    config_dict = {
        "train_batch_size": 1,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
            }
        },
        "zero_optimization": {
            "stage": 0
        },
        "fp16": {
            "enabled": True,
        },
        "flops_profiler": {
            "enabled": True,
            "step": 1,
            "module_depth": -1,
            "top_modules": 3,
        },
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10
    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[1])
    def _test_flops_profiler_in_ds_trainning(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                            model=model,
                                            model_parameters=model.parameters())

        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.half)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            if n == 3: break
        assert model.flops_profiler.flops == 100
        assert model.flops_profiler.params == 110

    _test_flops_profiler_in_ds_trainning(args, model, hidden_dim)


class LeNet5(torch.nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=6,
                            kernel_size=5,
                            stride=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=6,
                            out_channels=16,
                            kernel_size=5,
                            stride=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=120,
                            kernel_size=5,
                            stride=1),
            torch.nn.Tanh(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=120,
                            out_features=84),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=84,
                            out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return logits, probs


def test_flops_profiler_in_inference():
    mod = LeNet5(10)
    batch_size = 1024
    input = torch.randn(batch_size, 1, 32, 32)
    macs, params = get_model_profile(
        mod,
        tuple(input.shape),
        print_profile=True,
        detailed=True,
        module_depth=-1,
        top_modules=3,
        warm_up=1,
        as_string=True,
        ignore_modules=None,
    )
    print(macs, params)
    assert macs == "439.56 MMACs"
    assert params == "61.71 k"
