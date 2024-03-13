# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import random
import torch
import numpy as np
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from deepspeed.utils import logger
from deepspeed.runtime.pipe.engine import PipelineEngine


def batch_by_size(
    seqlens,
    max_tokens_per_batch,
    dataset_filter_ids=None,
    min_batch_size=1,
    max_batch_size=None,
    samples_order="dataloader",
    effective_batch_size=1,
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
    - `dataset_filter_ids`: user-defined indices of samples in the dataset that will be used to
       batch. Remaining indices to be ignored. Default is `None` for all indices.
    - `min_batch_size`: smallest allowed size of a batch;
    - `min_batch_size`: largest allowed size of a batch;
    - `samples_order`: order in which to process samples: "dataloader" (default), "random" or "seqlen" (ascending)
    - `dataloader_num_replicas`: number of dataloaders
    - `effective_batch_size`: effective batch size;
    - `required_microbatches_of_same_size`: enable if each mini-batch (in a total of `batch_size_multiple`
       micro-batches per batch), should have all micro-batches with the same batch size ie the same
       number of sentences.

    Returns:
    - `microbatch_ids`: list of tuple of batch id and samples ids per microbatch
    - `batch_sizes`: the effective batch size of each batch, used for to compute the scaled LR
    - `batch_max_seqlens`: the max seqlen across all microbatches in a batch
    """

    assert samples_order in ["random", "seqlen", "dataloader"]
    if dataset_filter_ids is None:
        metrics = list(zip(seqlens, range(len(seqlens))))  # use all samples
    else:
        metrics = list(zip(np.array(seqlens)[dataset_filter_ids], dataset_filter_ids))

    if samples_order == 'random':
        metric_random = random.Random(seed)
        metric_random.shuffle(metrics)
    if samples_order == 'seqlen':
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
    equal_size_multiple = effective_batch_size if required_microbatches_of_same_size else 1
    microbatches = []
    batch_init = 0
    while batch_init < len(metrics):

        # we iterate over possible effective batch sizes (groups of microbatches of same size)
        valid_batch_end = batch_init
        for batch_end in range(batch_init + equal_size_multiple, len(metrics), equal_size_multiple):

            # attempt effective batch
            batch = metrics[batch_init:batch_end]

            # pick interleaved samples for each microbatch to help with load balancing
            # (in the ordered use case), and to replicate what the distributed sampler does.
            mbs = [batch[b::equal_size_multiple] for b in range(equal_size_multiple)]

            # if they are all valid micro-batches, keep them until you find longer mbatches, if any
            is_batch_valid = all([is_microbatch_valid(mb) for mb in mbs])
            if is_batch_valid:
                valid_batch_end = batch_end

        if batch_init == valid_batch_end: break  # last batch is not valid (size zero), so we are done
        batch = metrics[batch_init:valid_batch_end]
        mbs = [batch[b::equal_size_multiple] for b in range(equal_size_multiple)]
        batch_init += sum([len(l) for l in mbs])
        microbatches += mbs

    # make sure we give the same number of (micro-)batches to each dataloader by trimming dataset
    microbatches = microbatches[:len(microbatches) - len(microbatches) % effective_batch_size]

    #compute the effective batch size for each microbatch.
    batch_sizes, batch_max_seqlens, microbatch_ids = [], [], []
    for rank in range(0, len(microbatches), effective_batch_size):
        batch_id = rank // effective_batch_size
        mbs = microbatches[rank:rank + effective_batch_size]
        batch_size = sum([len(mb) for mb in mbs])
        batch_max_seqlen = max([m[0] for metrics in mbs for m in metrics])
        dataset_filter_ids = [[m[1] for m in metrics] for metrics in mbs]
        batch_and_mb_ids = zip([batch_id] * effective_batch_size, dataset_filter_ids)
        batch_sizes.append(batch_size)
        batch_max_seqlens.append(batch_max_seqlen)
        microbatch_ids += batch_and_mb_ids
        n_tokens_in_batch = sum([m[0] for m in mbs[0]])
        assert n_tokens_in_batch <= max_tokens_per_batch
        if verbose:
            logger.info(
                f"Batch id {batch_id}, size {batch_size}, tokens {n_tokens_in_batch} tokens, samples: {dataset_filter_ids}"
            )

    # return the sample ids of each microbatch, and the batch sizes
    assert len(batch_sizes) == len(microbatch_ids) // effective_batch_size
    return microbatch_ids, batch_sizes, batch_max_seqlens


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


def dataloader_for_variable_batch_size(
    dataset,
    microbatch_ids,
    batch_max_seqlens,
    dataloader_rank=0,
    dataloader_batch_size=1,
    dataloader_num_replicas=1,
    dataloader_collate_fn=None,
    dataloader_num_workers=2,
    dataloader_pin_memory=False,
    required_microbatches_of_same_seqlen=False,
    sample_padding_fn=None,
):

    # equidistantly distribute the microbatches across the replicas in an interleaved fashion.
    sampler = DistributedSampler(
        dataset=microbatch_ids,
        num_replicas=dataloader_num_replicas,
        rank=dataloader_rank,
        shuffle=False,
        drop_last=False,
    )

    # collate function wraps user-defined collate function to the variable batch data
    def collate_fn_wrapper(list_microbatch_ids):
        # each batch is a list of sample ids that fill up to the max tokens per batch
        # we return the collated batch of all dataset samples of all input batches.
        batch = []
        for batch_id, microbatch_ids in list_microbatch_ids:
            batch_data = [dataset[idx] for idx in microbatch_ids]
            if required_microbatches_of_same_seqlen:
                assert sample_padding_fn is not None, \
                    "padding dataloader_padding_fn must be provided if required_microbatches_of_same_seqlen is True"
                pad_len = batch_max_seqlens[batch_id]
                batch_data = [sample_padding_fn(sample, pad_len) for sample in batch_data]
            batch += batch_data
        return dataloader_collate_fn(batch) if dataloader_collate_fn else batch

    dataloader = DataLoader(
        dataset=microbatch_ids,
        batch_size=dataloader_batch_size,
        sampler=sampler,
        num_workers=dataloader_num_workers,
        collate_fn=collate_fn_wrapper,
        pin_memory=dataloader_pin_memory,
    )

    deepspeed_io_kwargs = dict(
        dataset=microbatch_ids,
        batch_size=dataloader_batch_size,
        pin_memory=dataloader_pin_memory,
        data_sampler=sampler,
        collate_fn=collate_fn_wrapper,
        num_local_io_workers=dataloader_num_workers,
    )

    return dataloader, deepspeed_io_kwargs


class VariableBatchSizeLR(LRScheduler):
    """ an LR scheduler that scales the LR of a given scheduler's LR """

    @property
    def optimizer(self):
        return self.base_lr_scheduler.optimizer

    def __init__(self,
                 lr_scheduler,
                 base_batch_size,
                 batch_sizes,
                 dataloader,
                 lr_scaling_method="linear",
                 last_epoch=-1,
                 verbose=False):
        self.batch_sizes = batch_sizes
        self.base_batch_size = base_batch_size
        self.lr_scaling_method = lr_scaling_method
        self.dataloader = dataloader
        self.base_lr_scheduler = lr_scheduler
        # the following exist in LRScheduler but not in DeepSpeed's LRScheduler so we redefine them here
        self.base_lrs = self.base_lr_scheduler.get_lr()
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.step(0)

    def state_dict(self):
        return {
            'base_lr_scheduler': self.base_lr_scheduler.state_dict()
        } | {
            'base_batch_size': self.base_batch_size,
            'lr_scaling_method': self.lr_scaling_method,
            'batch_sizes': self.batch_sizes,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch,
            'verbose': self.verbose,
        }

    def load_state_dict(self, state_dict):
        self.base_lr_scheduler.load_state_dict(state_dict['base_lr_scheduler'])
        self.base_batch_size = state_dict['base_batch_size']
        self.lr_scaling_method = state_dict['lr_scaling_method']
        self.batch_sizes = state_dict['batch_sizes']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']
        self.verbose = state_dict['verbose']

    def get_last_lr(self):
        return self.base_lr_scheduler._last_lr

    def get_lr(self):
        return [group['lr'] for group in self.base_lr_scheduler.optimizer.param_groups]

    def step(self, epoch=None):
        # call the base scheduler's step method to get LR for next epoch
        # Note: optimizer.step precedes lr_scheduler.step(), so the stepping workflow is:
        # init: lr_scheduler.step(0) --> set LR for epoch 0
        # epoch 0: optimizer.step(); lr_scheduler.step(1) --> set LR for epoch 1
        # epoch 1: optimizer.step(); lr_scheduler.step(2) --> set LR for epoch 2

        # reset unscaled LRs (to the original scheduler's one) for the current epoch
        # Note: epoch==0: reset LR scheduler; epoch==None: scale LR for next epoch;
        unscaled_lrs = self.base_lrs if epoch == 0 else self.get_last_lr()
        for group, lr in zip(self.base_lr_scheduler.optimizer.param_groups, unscaled_lrs):
            group['lr'] = lr

        self.base_lr_scheduler.step(epoch)  # set unscaled lr, _step_count, last_epoch, _last_lr for new epoch

        # scale the learning rate for next epoch for each parameter group.
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        batch_size = self.batch_sizes[self.last_epoch % len(self.batch_sizes)]
        for group in self.base_lr_scheduler.optimizer.param_groups:
            group['lr'] = scale_lr(self.base_batch_size, batch_size, group['lr'], self.lr_scaling_method)

        if self.verbose:
            logger.info(f"Batch id {self.last_epoch}, unscaled LRs {unscaled_lrs}, scaled LRs {self.get_lr()}")


def lr_scheduler_for_variable_batch_size(base_batch_size,
                                         batch_sizes,
                                         dataloader,
                                         lr_scheduler_or_optimizer,
                                         lr_scaling_method='linear'):
    """
    returns a class that provides an LR scheduler that scales learning rate at every
    epoch taking into account the batch size of each epoch.
    If learning rate is constant, ie no LR scheduler, then the LR will be taken from the
    constant LR values in the optimizer param groups. Otherwise from the scheduler's LR.

    Arguments:
    - `base_batch_size`: the batch size that the base LR in the optimizer or scheduler refers to;
    - `lr_scaling_method`: method to use to scale LR - see `scale_lr()`;
    - `lr_scheduler_or_optimizer`: one instance of `LRScheduler` or `Optimizer` to be used as base;
    - `batch_sizes`: the effective batch size of each batch in the dataloader;

    Returns the new LRScheduler
    """

    class StubLRScheduler(LRScheduler):
        """ a stub LR scheduler that does not change the LR, keeps it constant """

        def get_lr(self) -> float:
            return self.base_lrs

    if isinstance(lr_scheduler_or_optimizer, Optimizer):
        lr_scheduler = StubLRScheduler(lr_scheduler_or_optimizer)
    elif hasattr(lr_scheduler_or_optimizer, 'optimizer'):  #LRScheduler or DeepSpeed 'object' schedulers
        assert isinstance(lr_scheduler_or_optimizer.optimizer, Optimizer)
        lr_scheduler = lr_scheduler_or_optimizer
    else:
        raise ValueError("Unknown type for lr_scheduler_or_optimizer: {}".format(type(lr_scheduler_or_optimizer)))

    return VariableBatchSizeLR(lr_scheduler=lr_scheduler,
                               base_batch_size=base_batch_size,
                               batch_sizes=batch_sizes,
                               dataloader=dataloader,
                               lr_scaling_method=lr_scaling_method)


def get_dataloader_and_lr_scheduler_for_variable_batch_size_deepspeed(dataset,
                                                                      dataset_seqlens,
                                                                      engine,
                                                                      dataset_filter_ids=None,
                                                                      dataloader_collate_fn=None,
                                                                      sample_padding_fn=None,
                                                                      replace_lr_scheduler=True,
                                                                      replace_dataloader=True):
    """
    a simplified call to get_dataloader_and_lr_scheduler_for_variable_batch_size for the deepspeed runtime.
    See `batch_by_size()` for arguments and documentation.
    """
    batching_config = engine.config['data_efficiency']['dynamic_batching']
    dataloader, lr_scheduler, deepspeed_io_kwargs = get_dataloader_and_lr_scheduler_for_variable_batch_size(
        dataset=dataset,
        dataset_filter_ids=dataset_filter_ids,
        dataset_seqlens=dataset_seqlens,
        effective_batch_size=engine.train_batch_size(),
        max_tokens_per_batch=batching_config["max_tokens_per_batch"],
        lr_scaling_method=batching_config["lr_scaling_method"],
        samples_order=batching_config["samples_order"],
        min_batch_size=batching_config["min_batch_size"],
        max_batch_size=batching_config["max_batch_size"],
        dataloader_batch_size=engine.train_micro_batch_size_per_gpu(),
        dataloader_rank=engine.data_parallel_group.rank(),
        dataloader_num_replicas=engine.data_parallel_group.size(),
        dataloader_num_workers=batching_config["dataloader_num_workers"],
        dataloader_collate_fn=dataloader_collate_fn,
        dataloader_pin_memory=batching_config["dataloader_pin_memory"],
        sample_padding_fn=sample_padding_fn,
        lr_scheduler_or_optimizer=engine.lr_scheduler or engine.optimizer,
        required_microbatches_of_same_size=isinstance(engine, PipelineEngine),
        required_microbatches_of_same_seqlen=isinstance(engine, PipelineEngine),
        verbose=batching_config["verbose"],
    )
    if replace_lr_scheduler:
        engine.lr_scheduler = lr_scheduler
    if replace_dataloader:
        engine.training_dataloader = dataloader
        engine.data_iterator = iter(engine.training_dataloader)
        # engine.deepspeed_io(**deepspeed_io_kwargs)
    return dataloader, lr_scheduler, deepspeed_io_kwargs


def get_dataloader_and_lr_scheduler_for_variable_batch_size(
    dataset,
    dataset_seqlens,
    max_tokens_per_batch,
    effective_batch_size,
    dataset_filter_ids=None,
    lr_scaling_method="linear",
    min_batch_size=1,
    max_batch_size=None,
    samples_order="dataloader",
    dataloader_batch_size=1,
    dataloader_rank=0,
    dataloader_num_replicas=1,
    dataloader_num_workers=0,
    dataloader_collate_fn=None,
    dataloader_pin_memory=False,
    lr_scheduler_or_optimizer=None,
    required_microbatches_of_same_size=False,
    required_microbatches_of_same_seqlen=False,
    sample_padding_fn=None,
    verbose=False,
):
    """ returns a dataloader and LR scheduler for the variable batch size. see `batch_by_size()` for details. """

    # effective_batch_size = train_micro_batch_size_per_gpu * gradient_accumulation_steps * number of dataloaders
    microbatch_ids, batch_sizes, batch_max_seqlens = batch_by_size(
        seqlens=dataset_seqlens,
        max_tokens_per_batch=max_tokens_per_batch,
        dataset_filter_ids=dataset_filter_ids,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        samples_order=samples_order,
        effective_batch_size=effective_batch_size,
        required_microbatches_of_same_size=required_microbatches_of_same_size,
        verbose=verbose,
    )

    dataloader, deepspeed_io_kwargs = dataloader_for_variable_batch_size(
        dataset=dataset,
        microbatch_ids=microbatch_ids,
        batch_max_seqlens=batch_max_seqlens,
        dataloader_rank=dataloader_rank,
        dataloader_num_replicas=dataloader_num_replicas,
        dataloader_batch_size=dataloader_batch_size,
        dataloader_collate_fn=dataloader_collate_fn,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        required_microbatches_of_same_seqlen=required_microbatches_of_same_seqlen,
        sample_padding_fn=sample_padding_fn,
    )

    lr_scheduler = lr_scheduler_for_variable_batch_size(base_batch_size=effective_batch_size,
                                                        batch_sizes=batch_sizes,
                                                        lr_scaling_method=lr_scaling_method,
                                                        lr_scheduler_or_optimizer=lr_scheduler_or_optimizer,
                                                        dataloader=dataloader)

    return dataloader, lr_scheduler, deepspeed_io_kwargs
