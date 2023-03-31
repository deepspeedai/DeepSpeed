# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from deepspeed.accelerator import get_accelerator

from deepspeed.runtime.data_pipeline.data_sampling.data_sampler import DeepSpeedDataSampler
from deepspeed.runtime.data_pipeline.constants import CURRICULUM_LEARNING, \
    DATA_EFFICIENCY, DATA_SAMPLING_NUM_WORKERS
from deepspeed.runtime.constants import GRADIENT_ACCUMULATION_STEPS, \
    DATA_PARALLEL_GROUP, GLOBAL_RANK


class RepeatingLoader:

    def __init__(self, loader):
        """Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.

        Args:
            loader (iterator): The data loader to repeat.
        """
        self.loader = loader
        self.data_iter = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch


class DeepSpeedDataLoader(object):

    def __init__(self,
                 dataset,
                 batch_size,
                 pin_memory,
                 local_rank,
                 tput_timer,
                 collate_fn=None,
                 num_local_io_workers=None,
                 data_sampler=None,
                 data_parallel_world_size=None,
                 data_parallel_rank=None,
                 dataloader_drop_last=False,
                 deepspeed_dataloader_config={}):
        self.deepspeed_dataloader_config = deepspeed_dataloader_config
        self.tput_timer = tput_timer
        self.batch_size = batch_size
        self.curriculum_learning_enabled = False
        if CURRICULUM_LEARNING in deepspeed_dataloader_config:
            self.curriculum_learning_enabled = deepspeed_dataloader_config[CURRICULUM_LEARNING]

        if self.curriculum_learning_enabled:
            data_sampler = DeepSpeedDataSampler(self.deepspeed_dataloader_config[DATA_EFFICIENCY],
                                                len(dataset),
                                                self.batch_size,
                                                data_parallel_rank,
                                                data_parallel_world_size,
                                                self.deepspeed_dataloader_config[DATA_PARALLEL_GROUP],
                                                self.deepspeed_dataloader_config[GRADIENT_ACCUMULATION_STEPS],
                                                self.deepspeed_dataloader_config[GLOBAL_RANK],
                                                drop_last=dataloader_drop_last)
            device_count = get_accelerator().device_count()
            num_local_io_workers = self.deepspeed_dataloader_config[DATA_SAMPLING_NUM_WORKERS]
        else:
            if local_rank >= 0:
                if data_sampler is None:
                    data_sampler = DistributedSampler(dataset=dataset,
                                                      num_replicas=data_parallel_world_size,
                                                      rank=data_parallel_rank)
                device_count = 1
            else:
                if data_sampler is None:
                    data_sampler = RandomSampler(dataset)
                device_count = get_accelerator().device_count()
                batch_size *= device_count

            if num_local_io_workers is None:
                num_local_io_workers = 2 * device_count

        self.num_local_io_workers = num_local_io_workers
        self.data_sampler = data_sampler
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.device_count = device_count
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.data = None
        self.dataloader_drop_last = dataloader_drop_last
        self.post_process_func = None

        if self.dataloader_drop_last:
            self.len = len(self.data_sampler) // self.batch_size
        else:
            from math import ceil
            self.len = ceil(len(self.data_sampler) / self.batch_size)

    def __iter__(self):
        self._create_dataloader()
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        if self.tput_timer:
            self.tput_timer.start()
        if self.curriculum_learning_enabled:
            data = next(self.data_iterator)
            if self.post_process_func is not None:
                data = self.post_process_func(data, self.data_sampler.state_dict())
            return data
        else:
            return next(self.data)

    def _create_dataloader(self):
        if self.curriculum_learning_enabled:
            if self.collate_fn is None:
                self.dataloader = DataLoader(self.dataset,
                                             pin_memory=self.pin_memory,
                                             batch_sampler=self.data_sampler,
                                             num_workers=self.num_local_io_workers)
            else:
                self.dataloader = DataLoader(self.dataset,
                                             pin_memory=self.pin_memory,
                                             batch_sampler=self.data_sampler,
                                             collate_fn=self.collate_fn,
                                             num_workers=self.num_local_io_workers)
            self.data_iterator = iter(self.dataloader)
            return self.dataloader
        else:
            if self.collate_fn is None:
                self.dataloader = DataLoader(self.dataset,
                                             batch_size=self.batch_size,
                                             pin_memory=self.pin_memory,
                                             sampler=self.data_sampler,
                                             num_workers=self.num_local_io_workers,
                                             drop_last=self.dataloader_drop_last)
            else:
                self.dataloader = DataLoader(self.dataset,
                                             batch_size=self.batch_size,
                                             pin_memory=self.pin_memory,
                                             sampler=self.data_sampler,
                                             collate_fn=self.collate_fn,
                                             num_workers=self.num_local_io_workers,
                                             drop_last=self.dataloader_drop_last)
            self.data = (x for x in self.dataloader)

            return self.dataloader


# DataLoader([(torch.randn(3, 3), torch.tensor(i % 2)) for i in range(10)], batch_size=2))
