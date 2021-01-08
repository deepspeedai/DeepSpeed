import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import pytest

import deepspeed
import deepspeed.runtime.utils as ds_utils


from deepspeed.runtime.pipe.topology import PipeDataParallelTopology, PipeModelDataParallelTopology
PipeTopo = PipeDataParallelTopology
import deepspeed.runtime.pipe.module as PipelineModule
from deepspeed.runtime.pipe.module import LayerSpec

from common import distributed_test


def rel_diff(A, B):
    return abs(A - B) / abs(A)


# All models
from simple_model import args_from_dict


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,
                      64,
                      kernel_size=11,
                      stride=4,
                      padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(64,
                      192,
                      kernel_size=5,
                      padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(192,
                      384,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,
                      256,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,
                      256,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.loss_fn(x, y)


class AlexNetPipe(PipelineModule.PipelineModule):
    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        specs = [
            LayerSpec(nn.Conv2d, 3, 64, kernel_size=11, stride=4, padding=5),
            LayerSpec(nn.ReLU, inplace=True),
            LayerSpec(nn.MaxPool2d, kernel_size=2, stride=2),
            LayerSpec(nn.Conv2d, 64, 192, kernel_size=5, padding=2),
            F.relu,
            LayerSpec(nn.MaxPool2d, kernel_size=2, stride=2),
            LayerSpec(nn.Conv2d, 192, 384, kernel_size=3, padding=1),
            F.relu,
            LayerSpec(nn.Conv2d, 384, 256, kernel_size=3, padding=1),
            F.relu,
            LayerSpec(nn.Conv2d, 256, 256, kernel_size=3, padding=1),
            F.relu,
            LayerSpec(nn.MaxPool2d, kernel_size=2, stride=2),

            lambda x: x.view(x.size(0), -1),
            LayerSpec(nn.Linear, 256, self.num_classes), # classifier
        ]
        super().__init__(layers=specs, loss_fn=nn.CrossEntropyLoss(), **kwargs)


def cifar_trainset(fp16=False):
    import torchvision
    import torchvision.transforms as transforms

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,
                              0.5,
                              0.5),
                             (0.5,
                              0.5,
                              0.5)),
    ]
    if fp16:
        transform_list.append(torchvision.transforms.Lambda(lambda x: x.half()))

    transform = transforms.Compose(transform_list)

    local_rank = torch.cuda.current_device()

    # Only one rank per machine downloads.
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    trainset = torchvision.datasets.CIFAR10(root='/tmp/cifar10-data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    if local_rank == 0:
        dist.barrier()
    return trainset


def train_cifar(model, args, num_steps=400, average_dp_losses=True, fp16=True, seed=123):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        ds_utils.set_random_seed(seed)

        trainset = cifar_trainset(fp16=fp16)
        args.local_rank = dist.get_rank()

        engine, _, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=[p for p in model.parameters()],
            training_data=trainset)

        losses = []
        for step in range(num_steps):
            loss = engine.train_batch()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f'STEP={step} LOSS={loss.item()}')

        if average_dp_losses:
            loss_tensor = torch.tensor(losses).cuda()
            dist.all_reduce(loss_tensor)
            loss_tensor /= dist.get_world_size()
            losses = loss_tensor.tolist()

    return losses


@pytest.mark.parametrize('base_topo,test_topo',
                         [
                             (PipeTopo(num_pp=1,
                                       num_dp=4),
                              PipeTopo(num_pp=2,
                                       num_dp=2)),
                             (PipeTopo(num_pp=1,
                                       num_dp=4),
                              PipeTopo(num_pp=4,
                                       num_dp=1)),
                         ])
def test_pipe_cifar10_seedlayers(base_topo, test_topo, tmpdir):
    config_dict = {
        "train_batch_size": 16,
        "train_micro_batch_size_per_gpu": 4,
        "steps_per_print": 20,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.9,
                          0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },
        "zero_optimization": {
            "stage": 0
        },
        "fp16": {
            "enabled": False
        },
        "pipeline": {
            "seed_layers": True,
            "activation_checkpoint_interval": 1
        }
    }
    args = args_from_dict(tmpdir, config_dict)

    @distributed_test(world_size=4)
    def _helper(base_topo, test_topo, tmpdir, steps=500):
        assert steps >= 100

        base_model = AlexNetPipe(num_classes=10,
                                 topology=base_topo,
                                 seed_layers=config_dict['pipeline']['seed_layers'])
        base_losses = train_cifar(base_model,
                                  args,
                                  num_steps=steps,
                                  fp16=config_dict['fp16']['enabled'])

        test_model = AlexNetPipe(num_classes=10,
                                 topology=test_topo,
                                 seed_layers=config_dict['pipeline']['seed_layers'])
        test_losses = train_cifar(test_model,
                                  args,
                                  num_steps=steps,
                                  fp16=config_dict['fp16']['enabled'])

        abs_diffs = [l0 - l1 for l0, l1 in zip(base_losses, test_losses)]
        rel_diffs = [rel_diff(l0, l1) for l0, l1 in zip(base_losses, test_losses)]
        if dist.get_rank() == 0:
            print(
                f'abs min={min(abs_diffs)} max={max(abs_diffs)} avg={sum(abs_diffs)/len(abs_diffs)}'
            )
            print(
                f'rel min={min(rel_diffs)} max={max(rel_diffs)} avg={sum(rel_diffs)/len(rel_diffs)}'
            )
            print(
                f'first: base={base_losses[0]} test={test_losses[0]} abs={abs_diffs[0]} rel={rel_diffs[0]}'
            )

            for lastX in [1, 10, 100]:
                base_avg = sum(base_losses[-lastX:]) / lastX
                test_avg = sum(test_losses[-lastX:]) / lastX
                print(
                    f'last-{lastX}: base={base_avg} test={test_avg} abs={base_avg - test_avg} rel={rel_diff(base_avg, test_avg)}'
                )

        lastX = 100
        base = base_losses[-lastX:]
        base_avg = sum(base) / len(base)
        test = test_losses[-lastX:]
        test_avg = sum(test) / len(test)
        assert rel_diff(base_avg, test_avg) < 0.03

    _helper(base_topo, test_topo, tmpdir)
