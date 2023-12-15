# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
import deepspeed.comm as dist
import deepspeed.runtime.utils as ds_utils
from deepspeed.runtime.utils import required_torch_version
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.pipe.module import PipelineModule, LayerSpec


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.loss_fn(x, y)


class AlexNetPipe(AlexNet):

    def to_layers(self):
        layers = [*self.features, lambda x: x.view(x.size(0), -1), self.classifier]
        return layers


class AlexNetPipeSpec(PipelineModule):

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
            LayerSpec(nn.Linear, 256, self.num_classes),  # classifier
        ]
        super().__init__(layers=specs, loss_fn=nn.CrossEntropyLoss(), **kwargs)


# Define this here because we cannot pickle local lambda functions
def cast_to_half(x):
    return x.half()


def cifar_trainset(fp16=False):
    torchvision = pytest.importorskip("torchvision", minversion="0.5.0")
    import torchvision.transforms as transforms

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    if fp16:
        transform_list.append(torchvision.transforms.Lambda(cast_to_half))

    transform = transforms.Compose(transform_list)

    local_rank = get_accelerator().current_device()

    # Only one rank per machine downloads.
    dist.barrier()
    if local_rank != 0:
        dist.barrier()

    data_root = os.getenv("TEST_DATA_DIR", "/tmp/")
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(data_root, "cifar10-data"),
                                            train=True,
                                            download=True,
                                            transform=transform)
    if local_rank == 0:
        dist.barrier()
    return trainset


def train_cifar(model, config, num_steps=400, average_dp_losses=True, fp16=True, seed=123):
    if required_torch_version(min_version=2.1):
        fork_kwargs = {"device_type": get_accelerator().device_name()}
    else:
        fork_kwargs = {}
    with get_accelerator().random().fork_rng(devices=[get_accelerator().current_device_name()], **fork_kwargs):
        ds_utils.set_random_seed(seed)

        # disable dropout
        model.eval()

        trainset = cifar_trainset(fp16=fp16)
        config['local_rank'] = dist.get_rank()

        # deepspeed_io defaults to creating a dataloader that uses a
        # multiprocessing pool. Our tests use pools and we cannot nest pools in
        # python. Therefore we're injecting this kwarg to ensure that no pools
        # are used in the dataloader.
        old_method = deepspeed.runtime.engine.DeepSpeedEngine.deepspeed_io

        def new_method(*args, **kwargs):
            kwargs["num_local_io_workers"] = 0
            return old_method(*args, **kwargs)

        deepspeed.runtime.engine.DeepSpeedEngine.deepspeed_io = new_method

        engine, _, _, _ = deepspeed.initialize(config=config,
                                               model=model,
                                               model_parameters=[p for p in model.parameters()],
                                               training_data=trainset)

        losses = []
        for step in range(num_steps):
            loss = engine.train_batch()
            losses.append(loss.item())
            if step % 50 == 0 and dist.get_rank() == 0:
                print(f'STEP={step} LOSS={loss.item()}')

        if average_dp_losses:
            loss_tensor = torch.tensor(losses).to(get_accelerator().device_name())
            dist.all_reduce(loss_tensor)
            loss_tensor /= dist.get_world_size()
            losses = loss_tensor.tolist()

    return losses
