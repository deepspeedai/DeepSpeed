# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import random
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from deepspeed.utils import logger


def batch_by_size(
    seqlens,
    max_tokens_per_batch,
    sample_ids=None,
    min_batch_size=1,
    max_batch_size=None,
    shuffle_seqlens=False,
    order_by_seqlen=False,
    dataloader_num_replicas=1,
    gradient_accumulation_steps=1,
    required_microbatches_of_same_size=False,
    verbose=False,
    seed=0,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain sequences of different lengths.
    Similar to "Attention is all you need", Section 5.1:
    "Sentence pairs were batched together by approximate sequence length. Each training batch
    contained a set of sentence pairs containing approximately X source tokens and X target tokens"

    Arguments:
    - `seqlens`: a list of difficulties (metric values) for every sample in the dataset;
    - `max_tokens_per_batch`: upper cap in total difficulty in a batch;
    - `sample_ids`: user-defined ids of the samples in seqlens. If not provided,
      automatically assigns a sequential order;
    - `min_batch_size`: smallest allowed size of a batch;
    - `min_batch_size`: largest allowed size of a batch;
    - `shuffle_seqlens`: shuffle metric values before packing samples into batches;
    - `order_by_seqlen`: order samples by ascending metric values before packing into batches;
    - `dataloader_num_replicas`: number of dataloaders
    - `gradient_accumulation_steps`: number of gradient accumulation steps;
    - `required_microbatches_of_same_size`: enable if each mini-batch (in a total of `batch_size_multiple`
       micro-batches per batch), should have all micro-batches with the same batch size ie the same
       number of sentences.

    Returns:
    - `microbatch_ids`: list of tuple of batch id and samples ids per microbatch
    - `batch_sizes`: the effective batch size of each batch, used for to compute the scaled LR
    - `batch_max_seqlens`: the max seqlen across all microbatches in a batch
    """

    assert not shuffle_seqlens or not order_by_seqlen, \
        "either sort_seqlens or shuffle_seqlens can be True, not both."

    sample_ids = sample_ids or list(range(len(seqlens)))
    metrics = list(zip(seqlens, sample_ids))

    if shuffle_seqlens:
        metric_random = random.Random(seed)
        metric_random.shuffle(metrics)
    if order_by_seqlen:
        metrics = sorted(metrics)

    # go through metrics and warn user and filter samples that alone exceed the max batch threshold
    long_ids = [idx for val, idx in metrics if val > max_tokens_per_batch]
    if len(long_ids) > 0:
        logger.warning(f"Data indices {long_ids} ignored as metrics exceed {max_tokens_per_batch}.")
        logger.info(f"Original dataset length: {len(metrics)}. New dataset length: {len(long_ids)}")
        metrics = [m for m in metrics if m[1] not in long_ids]

    def is_microbatch_valid(metrics):
        if min_batch_size and len(metrics) < min_batch_size: return False  # insufficient sample count
        if max_batch_size and len(metrics) > max_batch_size: return False  # too many samples
        if sum([m[0] for m in metrics]) > max_tokens_per_batch: return False  # exceeds max
        return True

    # go through all samples and pack then in microbatches of metric sums below the threshold
    # `required_microbatches_of_same_size` means all minibatches in a batch must be of equal size
    num_microbatches_per_batch = dataloader_num_replicas * gradient_accumulation_steps
    equal_size_multiple = num_microbatches_per_batch if required_microbatches_of_same_size else 1
    microbatches = []
    batch_init = 0
    while batch_init < len(metrics):

        # we iterate over possible effective batch sizes (groups of microbatches of same size)
        valid_batch_end = batch_init
        for batch_end in range(batch_init+equal_size_multiple, len(metrics), equal_size_multiple):

            # attempt effective batch
            batch = metrics[batch_init:batch_end]

            # pick interleaved samples for each microbatch to help with load balancing
            # (in the ordered use case), and to replicate what the distributed sampler does.
            mbs = [batch[b::equal_size_multiple] for b in range(equal_size_multiple)]

            # if they are all valid micro-batches, keep them until you find longer mbatches, if any
            is_batch_valid = all([is_microbatch_valid(mb) for mb in mbs])
            if is_batch_valid:
                valid_batch_end = batch_end

        if batch_init == valid_batch_end: break # last batch is not valid (size zero), so we are done
        batch = metrics[batch_init:valid_batch_end]
        mbs = [batch[b::equal_size_multiple] for b in range(equal_size_multiple)]
        batch_init += sum([len(l) for l in mbs])
        microbatches += mbs

    # make sure we give the same number of (micro-)batches to each dataloader by trimming dataset
    microbatches = microbatches[:len(microbatches) - len(microbatches) % num_microbatches_per_batch]

    #compute the effective batch size for each microbatch.
    batch_sizes, microbatch_ids = [], []
    for rank in range(0, len(microbatches), num_microbatches_per_batch):
        batch_id = rank // num_microbatches_per_batch
        mbs = microbatches[rank:rank + num_microbatches_per_batch]
        batch_size = sum([len(mb) for mb in mbs])
        mb_ids = [ [m[1] for m in metrics] for metrics in mbs]
        batch_sizes.append(batch_size)
        microbatch_ids += mb_ids
        n_tokens_in_batch = sum([m[0] for m in mbs[0]])
        assert n_tokens_in_batch <= max_tokens_per_batch
        if verbose:
            print(f"Batch id {batch_id}, size {batch_size}, tokens {n_tokens_in_batch} tokens, samples: {mb_ids}")

    # return the sample ids of each microbatch, and the batch sizes
    assert len(batch_sizes) == len(microbatch_ids) // num_microbatches_per_batch
    return microbatch_ids, batch_sizes


def scale_lr(base_batch_size, batch_size, base_lr=1, method="linear"):
    """ given a reference lr and batch_size, compute the new LR for a given batch size """
    if method == "linear":
        # Linear Scaling Rule: "When the minibatch size is multiplied by k, multiply the learning
        # rate by k" (Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, Goyal et al)
        return base_lr * batch_size / base_batch_size
    if method == "sqrt":
        # Square Root scaling: "when multiplying the batch size by k, multiply the learning rate
        # by √k, to keep the variance in the gradient expectation constant"
        # (A. Krizhevsky. One weird trick for parallelizing convolutional neural networks)
        return base_lr * torch.sqrt(batch_size / base_batch_size)
    elif method == None or method.upper() == "NONE":
        return base_lr
    raise ValueError("Unknown scaling method: {}".format(method))


def dataloader_for_variable_batch_size(dataset,
                                       microbatch_ids,
                                       dataloader_rank,
                                       dataloader_num_replicas,
                                       dataloader_collate_fn=None,
                                       dataloader_num_workers=2,
                                       dataloader_pin_memory=False):

    # equidistantly distribute the microbatches across the replicas in an interleaved fashion.
    sampler = DistributedSampler(
        dataset=microbatch_ids,
        num_replicas=dataloader_num_replicas,
        rank=dataloader_rank,
        shuffle=False,
        drop_last=False,
    )

    # collate function applies wraps user defined collate function to the variable batch data
    def collate_fn_wrapper(list_microbatch_ids, dataset, collate_fn=None):
        assert len(list_microbatch_ids) == 1, "only 1 element should be returned by the sampler."
        microbatch_ids = list_microbatch_ids[0]
        batch = [dataset[idx] for idx in microbatch_ids]
        return collate_fn(batch) if collate_fn else batch

    collate_fn = lambda b: collate_fn_wrapper(b, dataset, dataloader_collate_fn)

    dataloader = DataLoader(
        dataset=microbatch_ids,
        sampler=sampler,
        num_workers=dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=dataloader_pin_memory,
    )

    deepspeed_io_kwargs = dict(
        dataset=microbatch_ids,
        batch_size=1,
        pin_memory=dataloader_pin_memory,
        data_sampler=sampler,
        collate_fn=collate_fn,
        num_local_io_workers=dataloader_num_workers,
    )

    return dataloader, deepspeed_io_kwargs


class StubLRScheduler(LRScheduler):
    """ a stub LR scheduler that does not change the LR, keeps it constant """

    def get_lr(self) -> float:
        return self.base_lrs


def lr_scheduler_for_variable_batch_size(base_batch_size,
                                         batch_sizes,
                                         dataloader,
                                         lr_scaling_method='linear',
                                         optimizer=None,
                                         lr_scheduler_class=None,
                                         **lr_scheduler_kwargs):
    """
    returns a class that provides an LR scheduler that scales learning rate at every
    epoch taking into account the batch size of each epoch.
    If learning rate is constant, ie no LR scheduler, then `optimizer` must be provided.
    Otherwise, the base `LRScheduler` must be provided as  `lr_scheduler_class`.

    Arguments:
    - `base_batch_size`: the batch size that the base LR in the optimizer or scheduler refers to;
    - `lr_scaling_method`: method to use to scale LR - see `scale_lr()`;
    - `batch_sizes`: the effective batch size of each batch in the dataloader;
    - `optimizer` and `lr_scheduler_class`: the base LR scheduler. It not provided,
       will use the constant LRs from the optimizer's param groups instead. If provided,
       the initialization of the scheduler will be done with `lr_scheduler_kwargs`.

    Returns the new LRScheduler
    """

    class VariableBatchSizeLR(lr_scheduler_class or StubLRScheduler):

        def __init__(self, optimizer, **lr_scheduler_kwargs):
            self.batch_sizes = batch_sizes
            self.base_batch_size = base_batch_size
            self.lr_scaling_method = lr_scaling_method
            self.dataloader = dataloader
            self._last_lr = [p['lr'] for p in optimizer.param_groups]
            super().__init__(optimizer=optimizer, **lr_scheduler_kwargs)

        def state_dict(self):
            return {
                'base': super().state_dict(),
                'base_batch_size': self.base_batch_size,
                'lr_scaling_method': self.lr_scaling_method,
                'batch_sizes': self.batch_sizes,
            }

        def load_state_dict(self, state_dict):
            super().load_state_dict(state_dict['base'])
            self.base_batch_size = state_dict['base_batch_size']
            self.lr_scaling_method = state_dict['lr_scaling_method']
            self.batch_sizes = state_dict['batch_sizes']

        def get_lr(self):
            return [group['lr'] for group in self.optimizer.param_groups]

        def step(self, epoch=None):
            # call the base scheduler's step method to get LR for next epoch
            # Note: optimizer.step preecceds lr_scheduler.step(), so the stepping workflow is:
            # init: lr_scheduler.step(0) --> set LR for epoch 0
            # epoch 0: optimizer.step(); lr_scheduler.step(1) --> set LR for epoch 1
            # epoch 1: optimizer.step(); lr_scheduler.step(2) --> set LR for epoch 2

            # reset unscaled LRs (to the original scheduler's one) for the current epoch
            # Note: epoch==0: reset LR scheduler; epoch==None: scale LR for next epoch;
            unscaled_lrs = self.base_lrs if epoch == 0 else self._last_lr
            for group, lr in zip(self.optimizer.param_groups, unscaled_lrs):
                group['lr'] = lr

            super().step(epoch)  # set unscaled lr, _step_count, last_epoch, _last_lr for new epoch

            # scale the learning rate for next epoch for each parameter group.
            batch_size = self.batch_sizes[self.last_epoch % len(self.batch_sizes)]
            for group in self.optimizer.param_groups:
                group['lr'] = scale_lr(self.base_batch_size, batch_size, group['lr'], lr_scaling_method)

            if self.verbose:
                print(f"Batch id {self.last_epoch}, unscaled LR: {unscaled_lrs}, scaled LR: {self.get_lr()}")

    #### main loop: double check arguments and returns correctly-instantiated LR scheduler

    if lr_scheduler_class is None:
        assert optimizer is not None, "optimizer must be provided if lr_scheduler_class is not"
    else:
        assert issubclass(lr_scheduler_class, LRScheduler), "lr_scheduler should be a LRScheduler"

    if optimizer is None:
        assert lr_scheduler_class is not None, "lr_scheduler_class must be provided if optimizer is not"
        optimizer = lr_scheduler_kwargs['optimizer']

    return VariableBatchSizeLR(optimizer=optimizer, **lr_scheduler_kwargs)


def get_dataloader_and_lr_scheduler_for_variable_batch_size(
    dataset,
    dataset_seqlens,
    max_seqlen_per_batch,
    base_batch_size,
    sample_ids=None,
    lr_scaling_method="linear",
    min_batch_size=1,
    max_batch_size=None,
    shuffle_seqlens=False,
    order_by_seqlen=False,
    gradient_accumulation_steps=1,
    dataloader_rank=0,
    dataloader_num_replicas=1,
    dataloader_num_workers=0,
    dataloader_collate_fn=None,
    dataloader_pin_memory=False,
    optimizer=None,
    lr_scheduler_class=None,
    lr_scheduler_kwargs={'verbose': False},
    required_microbatches_of_same_size=False,
    verbose=False,
):

    microbatch_ids, batch_sizes = batch_by_size(
        seqlens=dataset_seqlens,
        max_tokens_per_batch=max_seqlen_per_batch,
        sample_ids=sample_ids,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        shuffle_seqlens=shuffle_seqlens,
        order_by_seqlen=order_by_seqlen,
        dataloader_num_replicas=dataloader_num_replicas,
        gradient_accumulation_steps=gradient_accumulation_steps,
        required_microbatches_of_same_size=required_microbatches_of_same_size,
        verbose=verbose,
    )

    dataloader, deepspeed_io_kwargs = dataloader_for_variable_batch_size(
        dataset=dataset,
        microbatch_ids=microbatch_ids,
        dataloader_rank=dataloader_rank,
        dataloader_num_replicas=dataloader_num_replicas,
        dataloader_collate_fn=dataloader_collate_fn,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
    )

    lr_scheduler = lr_scheduler_for_variable_batch_size(base_batch_size=base_batch_size,
                                                        batch_sizes=batch_sizes,
                                                        lr_scaling_method=lr_scaling_method,
                                                        optimizer=optimizer,
                                                        dataloader=dataloader,
                                                        lr_scheduler_class=lr_scheduler_class,
                                                        **lr_scheduler_kwargs)

    return dataloader, lr_scheduler, deepspeed_io_kwargs
