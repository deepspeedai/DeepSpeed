'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler


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
                 data_parallel_rank=None):
        self.tput_timer = tput_timer
        self.batch_size = batch_size

        if local_rank >= 0:
            if data_sampler is None:
                data_sampler = DistributedSampler(dataset=dataset,
                                                  num_replicas=data_parallel_world_size,
                                                  rank=data_parallel_rank)
            device_count = 1
        else:
            if data_sampler is None:
                data_sampler = RandomSampler(dataset)
            device_count = torch.cuda.device_count()
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
        self.len = len(self.data_sampler)
        self.data = None

    def __iter__(self):
        self._create_dataloader()
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        if self.tput_timer:
            self.tput_timer.start()
        return next(self.data)

    def _create_dataloader(self):
        if self.collate_fn is None:
            self.dataloader = DataLoader(self.dataset,
                                         batch_size=self.batch_size,
                                         pin_memory=self.pin_memory,
                                         sampler=self.data_sampler,
                                         num_workers=self.num_local_io_workers)
        else:
            self.dataloader = DataLoader(self.dataset,
                                         batch_size=self.batch_size,
                                         pin_memory=self.pin_memory,
                                         sampler=self.data_sampler,
                                         collate_fn=self.collate_fn,
                                         num_workers=self.num_local_io_workers)
        self.data = (x for x in self.dataloader)

        return self.dataloader


# DataLoader([(torch.randn(3, 3), torch.tensor(i % 2)) for i in range(10)], batch_size=2))
