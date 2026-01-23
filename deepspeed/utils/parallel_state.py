# SPDX-License-Identifier: Apache-2.0
# Copyright (c) DeepSpeed Team

# DeepSpeed Team

# The file has been adapted from https://github.com/NVIDIA/Megatron-LM and retains the following license from the original file

# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Refactored Model and data parallel groups with class-based design."""

import logging
from datetime import timedelta
from typing import Callable, List, Optional

import numpy as np
import torch

from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
from deepspeed.utils.torch import required_torch_version


logger = logging.getLogger(__name__)

try:
    import einops
    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False


def is_torch_min_version(version: str, check_equality: bool = True) -> bool:
    """Check if PyTorch version meets minimum requirement.

    Args:
        version: Version string to check (e.g., "2.4.0")
        check_equality: If True, also check for equality

    Returns:
        True if version requirement is met
    """
    try:
        from packaging.version import Version as PkgVersion
        torch_version = PkgVersion(torch.__version__)
        required_version = PkgVersion(version)
        if check_equality:
            return torch_version >= required_version
        return torch_version > required_version
    except Exception:
        return False


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context=None):
        """Returns a sub-tensor from the buffer for the given shape."""
        from functools import reduce
        import operator

        required_len = reduce(operator.mul, tensor_shape, 1)
        if (self.buffer.get((name, dtype), None) is None or self.buffer[(name, dtype)].numel() < required_len):
            from contextlib import nullcontext
            mem_alloc_context = mem_alloc_context if mem_alloc_context else nullcontext
            with mem_alloc_context():
                self.buffer[(name, dtype)] = torch.empty(
                    required_len,
                    dtype=dtype,
                    device=get_accelerator().current_device(),
                    requires_grad=False,
                )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def generate_masked_orthogonal_rank_groups(world_size: int, parallel_size: List[int],
                                           mask: List[bool]) -> List[List[int]]:
    r"""Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size
        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].
        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size
    """

    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        """Solve: index = sum(idx[i] * stride[i])"""
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        assert (sum([x * y for x, y in zip(idx, stride[:-1])]) == index), f"idx {index} with shape {shape} mismatch"
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride) +
                inner_product(decomposed_group_idx, unmasked_stride))
        ranks.append(rank)
    return ranks


class RankGenerator:
    """A class for generating rank groups for different modes of parallelism."""

    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, sp: int, order: str, rank_offset: int = 0) -> None:
        assert (ep == 1 or cp == 1), "Both EP and CP > 1 is not allowed in one rank generator."

        # Check SP compatibility: SP cannot be used with TP, PP, or EP
        if sp > 1:
            if tp > 1:
                raise RuntimeError(f"Sequence Parallel (SP) cannot be used together with Tensor Parallel (TP). "
                                   f"SP size: {sp}, TP size: {tp}. "
                                   "Please set tp=1 when using SP.")
            if pp > 1:
                raise RuntimeError(f"Sequence Parallel (SP) cannot be used together with Pipeline Parallel (PP). "
                                   f"SP size: {sp}, PP size: {pp}. "
                                   "Please set pp=1 when using SP.")
            if ep > 1:
                raise RuntimeError(f"Sequence Parallel (SP) cannot be used together with Expert Parallel (EP). "
                                   f"SP size: {sp}, EP size: {ep}. "
                                   "Please set ep=1 when using SP.")

        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.sp = sp
        self.rank_offset = rank_offset
        self.world_size = tp * dp * pp * cp * ep * sp

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
            "sp": self.sp,
        }
        self.order = order
        order = order.lower()

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't"
                                   f"specified the order ({self.order}).")
            elif name not in order:
                order = order + "-" + name

        self.order = order
        self.ordered_size = []

        for token in order.split("-"):
            self.ordered_size.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str):
        """Create a mask for the specified tokens based on the given order."""
        ordered_token = order.split("-")
        token_list = token.split("-")
        mask = [False] * len(ordered_token)
        for t in token_list:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token):
        """Get rank group by input token.

        Args:
            token (str): Specify the ranks type (e.g., 'tp-dp')
        """
        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, self.ordered_size, mask)
        if self.rank_offset > 0:
            for rank_group in ranks:
                for i in range(len(rank_group)):
                    rank_group[i] += self.rank_offset
        return ranks


class ParallelState:
    """Encapsulates all parallel state and operations.

    This class replaces the global variables and functions from the original
    parallel_state.py, providing a cleaner, more maintainable interface.
    """

    def __init__(self):
        # Process groups
        self.tensor_model_parallel_group = None
        self.pipeline_model_parallel_group = None
        self.model_parallel_group = None
        self.embedding_group = None
        self.position_embedding_group = None
        self.data_parallel_group = None
        self.data_parallel_group_gloo = None
        self.tensor_and_data_parallel_group = None
        self.context_parallel_group = None
        self.tensor_and_context_parallel_group = None
        self.tensor_and_data_parallel_group_with_cp = None
        self.data_parallel_group_with_cp = None
        self.data_parallel_group_with_cp_gloo = None

        # Sequence parallel groups
        self.sequence_parallel_group = None
        self.sequence_and_data_parallel_group = None

        # Expert-related groups
        self.expert_model_parallel_group = None
        self.expert_tensor_parallel_group = None
        self.expert_tensor_and_model_parallel_group = None
        self.expert_tensor_model_pipeline_parallel_group = None
        self.expert_data_parallel_group = None
        self.expert_data_parallel_group_gloo = None
        self.intra_partial_expert_data_parallel_group = None
        self.intra_partial_expert_data_parallel_group_gloo = None
        self.inter_partial_expert_data_parallel_group = None

        # All-to-All groups for ZeRO++ quantized gradients
        self.all_to_all_groups = {}
        self.all_to_all_initialized = False

        # Global ranks lists
        self.embedding_global_ranks = None
        self.position_embedding_global_ranks = None
        self.pipeline_global_ranks = None
        self.data_parallel_global_ranks = None
        self.tensor_model_parallel_global_ranks = None
        self.model_parallel_global_ranks = None
        self.context_parallel_global_ranks = None
        self.data_parallel_global_ranks_with_cp = None
        self.hierarchical_context_parallel_groups = None

        # Parallel state values
        self.virtual_pipeline_model_parallel_rank = None
        self.virtual_pipeline_model_parallel_world_size = None
        self.mpu_tensor_model_parallel_world_size = None
        self.mpu_pipeline_model_parallel_world_size = None
        self.mpu_data_parallel_world_size = None
        self.mpu_data_parallel_rank = None
        self.mpu_tensor_model_parallel_rank = None
        self.mpu_pipeline_model_parallel_rank = None

        # Expert parallel state values
        self.mpu_expert_model_parallel_world_size = None
        self.mpu_expert_model_parallel_rank = None
        self.mpu_expert_tensor_parallel_world_size = None
        self.mpu_expert_tensor_parallel_rank = None

        # Other
        self.global_memory_buffer = None
        self.global_process_group_list = None
        self.intra_partial_data_parallel_group_with_cp = None
        self.intra_partial_data_parallel_group_with_cp_gloo = None
        self.intra_distributed_optimizer_instance_group = None

        # Rank generators
        self.decoder_rank_generator = None
        self.expert_decoder_rank_generator = None

    def _get_pg_options(self, pg_name: str, pg_comm_cfgs: dict):
        """Get the options for a specific process group."""
        # TODO: construct process group options from json config
        #
        # As of PyTorch 2.9, the only backend that supports pg options is nccl,
        # and a nccl-specific class, namely ProcessGroupNCCL.Options, is
        # required to construct the options.
        #
        # To enable configuring such options in DeepSpeed, we need to define the
        # interface for users to specify them and also figure out whether we
        # want to export ProcessGroupNCCL.Options in deepspeed.comm or allow
        # using torch distributed for this specific case in check-torchdist.py.
        # Those are left as future work.
        return None

    def _create_group(
        self,
        ranks,
        timeout=None,
        backend=None,
        pg_options=None,
        use_local_synchronization=False,
        group_desc=None,
    ):
        """Creates a ProcessGroup."""
        if backend is not None and backend != "nccl":
            logger.warning(f"{backend} backend is not supported for new_group. Using torch.distributed directly.")
            return None

        # TODO: Currently using deepspeed.comm.new_group() which only supports 'ranks' parameter.
        # The following parameters are commented out and will be enabled once DeepSpeed's
        # comm interface supports them:
        # - timeout: Timeout for process group operations
        # - backend: Communication backend (e.g., 'nccl', 'gloo')
        # - pg_options: Process group options
        # - use_local_synchronization: Enable local synchronization
        # - group_desc: Group description for debugging (requires PyTorch >= 2.4)
        kwargs = {
            "ranks": ranks,
            # "timeout": timeout,
            # "backend": backend,
            # "pg_options": pg_options,
            # "use_local_synchronization": use_local_synchronization,
            # "group_desc": group_desc,
        }

        group = dist.new_group(**kwargs)
        if self.global_process_group_list is None:
            self.global_process_group_list = [None]
        if dist.get_rank() in ranks:
            self.global_process_group_list.append(group)
        return group

    def _create_hierarchical_groups(
        self,
        rank,
        ranks,
        hierarchical_group_sizes,
        create_gloo_process_groups=False,
        pg_options=None,
        timeout=None,
        group_desc=None,
    ):
        """Create hierarchical groups for a set of ranks."""
        if not HAVE_EINOPS:
            raise ImportError("einops is not installed. Please install it with `pip install einops`.")

        hierarchical_groups = []
        hierarchical_groups_gloo = []
        if not isinstance(pg_options, list):
            pg_options = [pg_options] * len(hierarchical_group_sizes)

        for level in range(len(hierarchical_group_sizes)):
            rearranged_ranks = einops.rearrange(
                np.array(ranks),
                "(l s u) -> (l u) s",
                u=int(np.prod(hierarchical_group_sizes[:level])),
                s=hierarchical_group_sizes[level],
                l=int(np.prod(hierarchical_group_sizes[level + 1:])),
            ).tolist()
            for sub_ranks in rearranged_ranks:
                sub_group = self._create_group(
                    sub_ranks,
                    timeout=timeout,
                    pg_options=pg_options[level],
                    group_desc=f"HIERARCHICAL_{group_desc}_L{level}",
                )
                if create_gloo_process_groups:
                    sub_group_gloo = self._create_group(
                        sub_ranks,
                        timeout=timeout,
                        backend="gloo",
                        pg_options=pg_options[level],
                        group_desc=f"HIERARCHICAL_{group_desc}_GLOO_L{level}",
                    )
                else:
                    sub_group_gloo = None
                if rank in sub_ranks:
                    hierarchical_groups.append(sub_group)
                    hierarchical_groups_gloo.append(sub_group_gloo)

        assert rank not in ranks or len(hierarchical_groups) == len(hierarchical_group_sizes)
        assert rank not in ranks or len(hierarchical_groups_gloo) == len(hierarchical_group_sizes)
        return hierarchical_groups, hierarchical_groups_gloo

    def initialize_model_parallel(
        self,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_comm_backend: Optional[str] = None,
        context_parallel_size: int = 1,
        sequence_parallel_size: int = 1,
        hierarchical_context_parallel_sizes: Optional[List[int]] = None,
        expert_model_parallel_size: int = 1,
        num_distributed_optimizer_instances: int = 1,
        expert_tensor_parallel_size: Optional[int] = None,
        distributed_timeout_minutes: int = 30,
        order: str = "tp-ep-dp-pp",
        get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
        get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
        create_gloo_process_groups: bool = False,
    ) -> None:
        """Initialize model data parallel groups.

        This is the main initialization method that sets up all parallel groups.
        """

        def default_embedding_ranks(pp_ranks):
            """Return the default ranks that constitute the stages on which the word embeddings live."""
            if len(pp_ranks) == 1:
                return [pp_ranks[0]]
            else:
                return [pp_ranks[0], pp_ranks[-1]]

        def default_position_embedding_ranks(pp_ranks):
            """Return the default ranks that constitute the stages on which the position embeddings live."""
            return [pp_ranks[0]]

        if get_embedding_ranks is None:
            get_embedding_ranks = default_embedding_ranks
        if get_position_embedding_ranks is None:
            get_position_embedding_ranks = default_position_embedding_ranks

        # Get world size and rank
        assert dist.is_initialized()
        world_size: int = dist.get_world_size()
        rank = dist.get_rank()

        model_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size * sequence_parallel_size
        if world_size % model_size != 0:
            raise RuntimeError(f"world_size ({world_size}) is not divisible by {model_size}")

        data_parallel_size: int = world_size // model_size

        if virtual_pipeline_model_parallel_size is not None:
            if not pipeline_model_parallel_size > 1:
                raise RuntimeError("pipeline-model-parallel size should be greater than 1 with interleaved schedule")
            self.virtual_pipeline_model_parallel_rank = 0
            self.virtual_pipeline_model_parallel_world_size = virtual_pipeline_model_parallel_size

        # TODO: Collect process group options from configs
        #
        # Check _get_pg_options for details.
        pg_comm_cfgs = {}

        # Create rank generators
        self.decoder_rank_generator = RankGenerator(
            tp=tensor_model_parallel_size,
            ep=1,
            dp=data_parallel_size,
            pp=pipeline_model_parallel_size,
            cp=context_parallel_size,
            order=order,
            rank_offset=0,
            sp=sequence_parallel_size,
        )

        # Build expert rank generator
        if expert_tensor_parallel_size is None:
            expert_tensor_parallel_size = tensor_model_parallel_size
        expert_tensor_model_pipeline_parallel_size = (expert_tensor_parallel_size * expert_model_parallel_size *
                                                      pipeline_model_parallel_size)
        expert_data_parallel_size = world_size // expert_tensor_model_pipeline_parallel_size
        if world_size % expert_tensor_model_pipeline_parallel_size != 0:
            raise RuntimeError(
                f"world_size ({world_size}) is not divisible by expert_tensor_model_pipeline_parallel size ({expert_tensor_model_pipeline_parallel_size})"
            )

        self.expert_decoder_rank_generator = RankGenerator(
            tp=expert_tensor_parallel_size,
            ep=expert_model_parallel_size,
            dp=expert_data_parallel_size,
            pp=pipeline_model_parallel_size,
            cp=1,
            order=order,
            rank_offset=0,
            sp=1,
        )

        timeout = timedelta(minutes=distributed_timeout_minutes)

        # Build data-parallel groups with context parallel
        assert self.data_parallel_group is None, "data parallel group is already initialized"
        assert (data_parallel_size * context_parallel_size) % num_distributed_optimizer_instances == 0, (
            "Data parallel size should be divisible by partial DistOpt shard factor")
        intra_partial_data_parallel_size = (data_parallel_size *
                                            context_parallel_size) // num_distributed_optimizer_instances

        for ranks_with_cp in self.decoder_rank_generator.get_ranks('dp-cp'):
            group_with_cp = self._create_group(
                ranks_with_cp,
                timeout=timeout,
                pg_options=self._get_pg_options("dp_cp", pg_comm_cfgs),
                group_desc="DATA_PARALLEL_GROUP_WITH_CP",
            )
            if create_gloo_process_groups:
                group_with_cp_gloo = self._create_group(
                    ranks_with_cp,
                    timeout=timeout,
                    backend="gloo",
                    group_desc="DATA_PARALLEL_GROUP_WITH_CP_GLOO",
                )
            else:
                group_with_cp_gloo = None
            if rank in ranks_with_cp:
                self.data_parallel_group_with_cp = group_with_cp
                self.data_parallel_group_with_cp_gloo = group_with_cp_gloo
                self.data_parallel_global_ranks_with_cp = ranks_with_cp

            if num_distributed_optimizer_instances > 1:
                for i in range(num_distributed_optimizer_instances):
                    intra_partial_dp_ranks_with_cp = ranks_with_cp[(
                        i * intra_partial_data_parallel_size):((i + 1) * intra_partial_data_parallel_size)]
                    intra_partial_dp_group_with_cp = self._create_group(
                        intra_partial_dp_ranks_with_cp,
                        timeout=timeout,
                        pg_options=self._get_pg_options("intra_dp_cp", pg_comm_cfgs),
                        group_desc="INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP",
                    )
                    if create_gloo_process_groups:
                        intra_partial_dp_group_with_cp_gloo = self._create_group(
                            intra_partial_dp_ranks_with_cp,
                            timeout=timeout,
                            backend="gloo",
                            group_desc="INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO",
                        )
                    else:
                        intra_partial_dp_group_with_cp_gloo = None
                    if rank in intra_partial_dp_ranks_with_cp:
                        self.intra_partial_data_parallel_group_with_cp = intra_partial_dp_group_with_cp
                        self.intra_partial_data_parallel_group_with_cp_gloo = (intra_partial_dp_group_with_cp_gloo)
            else:
                self.intra_partial_data_parallel_group_with_cp = self.data_parallel_group_with_cp
                self.intra_partial_data_parallel_group_with_cp_gloo = self.data_parallel_group_with_cp_gloo

        # Build data-parallel groups
        for ranks in self.decoder_rank_generator.get_ranks('dp'):
            group = self._create_group(
                ranks,
                timeout=timeout,
                pg_options=self._get_pg_options("dp", pg_comm_cfgs),
                group_desc="DATA_PARALLEL_GROUP",
            )
            if create_gloo_process_groups:
                group_gloo = self._create_group(ranks,
                                                timeout=timeout,
                                                backend="gloo",
                                                group_desc="DATA_PARALLEL_GROUP_GLOO")
            else:
                group_gloo = None
            if rank in ranks:
                self.data_parallel_group = group
                self.data_parallel_group_gloo = group_gloo
                self.data_parallel_global_ranks = ranks

        # Build context-parallel groups
        assert self.context_parallel_group is None, 'context parallel group is already initialized'
        for ranks in self.decoder_rank_generator.get_ranks('cp'):
            group = self._create_group(
                ranks,
                timeout=timeout,
                pg_options=self._get_pg_options("cp", pg_comm_cfgs),
                group_desc="CONTEXT_PARALLEL_GROUP",
            )
            if rank in ranks:
                self.context_parallel_group = group
                self.context_parallel_global_ranks = ranks
            if hierarchical_context_parallel_sizes:
                assert np.prod(hierarchical_context_parallel_sizes) == context_parallel_size
                hierarchical_groups, _ = self._create_hierarchical_groups(
                    rank,
                    ranks,
                    hierarchical_context_parallel_sizes,
                    create_gloo_process_groups=False,
                    pg_options=self._get_pg_options("hcp", pg_comm_cfgs),
                    timeout=timeout,
                    group_desc="CONTEXT_PARALLEL_GROUP",
                )
                if rank in ranks:
                    self.hierarchical_context_parallel_groups = hierarchical_groups

        # Build model-parallel groups
        assert self.model_parallel_group is None, 'model parallel group is already initialized'
        for ranks in self.decoder_rank_generator.get_ranks('tp-pp'):
            group = self._create_group(
                ranks,
                timeout=timeout,
                pg_options=self._get_pg_options("mp", pg_comm_cfgs),
                group_desc="MODEL_PARALLEL_GROUP",
            )
            if rank in ranks:
                self.model_parallel_group = group
                self.model_parallel_global_ranks = ranks

        # Build tensor model-parallel groups
        assert self.tensor_model_parallel_group is None, 'tensor model parallel group is already initialized'
        for ranks in self.decoder_rank_generator.get_ranks('tp'):
            group = self._create_group(
                ranks,
                timeout=timeout,
                pg_options=self._get_pg_options("tp", pg_comm_cfgs),
                group_desc="TENSOR_MODEL_PARALLEL_GROUP",
            )
            if rank in ranks:
                self.tensor_model_parallel_group = group
                self.tensor_model_parallel_global_ranks = ranks

        # Build pipeline model-parallel groups and embedding groups
        assert self.pipeline_model_parallel_group is None, "pipeline model parallel group is already initialized"
        assert self.embedding_group is None, "embedding group is already initialized"
        assert self.position_embedding_group is None, "position embedding group is already initialized"

        for ranks in self.decoder_rank_generator.get_ranks('pp'):
            group = self._create_group(
                ranks,
                timeout=timeout,
                backend=pipeline_model_parallel_comm_backend,
                pg_options=(None if pipeline_model_parallel_comm_backend == "ucc" else self._get_pg_options(
                    "pp", pg_comm_cfgs)),
                group_desc="PIPELINE_MODEL_PARALLEL_GROUP",
            )
            assert (
                pipeline_model_parallel_comm_backend == None or pipeline_model_parallel_comm_backend == "nccl"
                or pipeline_model_parallel_comm_backend == "ucc"
            ), f'"{pipeline_model_parallel_comm_backend}" backend for PP communication is currently not supported'

            if rank in ranks:
                if self.pipeline_model_parallel_group is None:
                    self.pipeline_model_parallel_group = group
                    self.pipeline_global_ranks = ranks
                elif isinstance(self.pipeline_global_ranks[0], list):
                    if not isinstance(self.pipeline_model_parallel_group, list):
                        self.pipeline_model_parallel_group = [self.pipeline_model_parallel_group]
                    self.pipeline_model_parallel_group.append(group)
                    self.pipeline_global_ranks.append(ranks)
                else:
                    self.pipeline_model_parallel_group = [self.pipeline_model_parallel_group, group]
                    self.pipeline_global_ranks = [self.pipeline_global_ranks, ranks]

            embedding_ranks = get_embedding_ranks(ranks)
            group = self._create_group(
                embedding_ranks,
                timeout=timeout,
                pg_options=self._get_pg_options("embd", pg_comm_cfgs),
                group_desc="EMBEDDING_GROUP",
            )
            if rank in embedding_ranks:
                self.embedding_group = group
                self.embedding_global_ranks = embedding_ranks

            position_embedding_ranks = get_position_embedding_ranks(ranks)
            group = self._create_group(
                position_embedding_ranks,
                timeout=timeout,
                pg_options=self._get_pg_options("pos_embd", pg_comm_cfgs),
                group_desc="POSITION_EMBEDDING_GROUP",
            )
            if rank in position_embedding_ranks:
                self.position_embedding_group = group
                self.position_embedding_global_ranks = position_embedding_ranks

        # Build tensor + data parallel groups
        assert self.tensor_and_data_parallel_group is None, 'Tensor + data parallel group is already initialized'
        for ranks in self.decoder_rank_generator.get_ranks('tp-dp-cp'):
            group = self._create_group(
                ranks,
                timeout=timeout,
                pg_options=self._get_pg_options("tp_dp_cp", pg_comm_cfgs),
                group_desc="TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP",
            )
            if rank in ranks:
                self.tensor_and_data_parallel_group_with_cp = group
        for ranks in self.decoder_rank_generator.get_ranks('tp-dp'):
            group = self._create_group(
                ranks,
                timeout=timeout,
                pg_options=self._get_pg_options("tp_dp", pg_comm_cfgs),
                group_desc="TENSOR_AND_DATA_PARALLEL_GROUP",
            )
            if rank in ranks:
                self.tensor_and_data_parallel_group = group

        assert self.tensor_and_context_parallel_group is None, 'Tensor + context parallel group is already initialized'
        for ranks in self.decoder_rank_generator.get_ranks('tp-cp'):
            group = self._create_group(
                ranks,
                timeout=timeout,
                pg_options=self._get_pg_options("tp_cp", pg_comm_cfgs),
                group_desc="TENSOR_AND_CONTEXT_PARALLEL_GROUP",
            )
            if rank in ranks:
                self.tensor_and_context_parallel_group = group

        # Build expert-related parallel groups
        assert self.expert_model_parallel_group is None, 'Expert parallel group is already initialized'
        for ranks in self.expert_decoder_rank_generator.get_ranks('ep'):
            group = self._create_group(
                ranks,
                pg_options=self._get_pg_options("ep", pg_comm_cfgs),
                group_desc="EXPERT_MODEL_PARALLEL_GROUP",
            )
            if rank in ranks:
                self.expert_model_parallel_group = group

        assert self.expert_tensor_parallel_group is None, 'Expert tensor model parallel group is already initialized'
        for ranks in self.expert_decoder_rank_generator.get_ranks('tp'):
            group = self._create_group(
                ranks,
                timeout=timeout,
                pg_options=self._get_pg_options("ep_tp", pg_comm_cfgs),
                group_desc="EXPERT_TENSOR_PARALLEL_GROUP",
            )
            if rank in ranks:
                self.expert_tensor_parallel_group = group

        assert self.expert_tensor_and_model_parallel_group is None, 'Expert tensor + model parallel group is already initialized'
        for ranks in self.expert_decoder_rank_generator.get_ranks('tp-ep'):
            group = self._create_group(
                ranks,
                timeout=timeout,
                pg_options=self._get_pg_options("tp_ep_mp", pg_comm_cfgs),
                group_desc="EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
            )
            if rank in ranks:
                self.expert_tensor_and_model_parallel_group = group

        assert self.expert_tensor_model_pipeline_parallel_group is None, 'The expert_tensor_model_pipeline parallel group is already initialized'
        for ranks in self.expert_decoder_rank_generator.get_ranks('tp-ep-pp'):
            group = self._create_group(
                ranks,
                timeout=timeout,
                pg_options=self._get_pg_options("tp_ep_pp", pg_comm_cfgs),
                group_desc="EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP",
            )
            if rank in ranks:
                self.expert_tensor_model_pipeline_parallel_group = group

        assert self.expert_data_parallel_group is None, "Expert data group is already initialized"
        assert self.expert_data_parallel_group_gloo is None, "Expert data group-gloo is already initialized"
        assert self.intra_partial_expert_data_parallel_group is None, "Intra partial expert data group is already initialized"
        assert self.intra_partial_expert_data_parallel_group_gloo is None, "Intra partial expert data group-gloo is already initialized"
        assert self.inter_partial_expert_data_parallel_group is None, "Inter partial expert data group is already initialized"

        assert (expert_data_parallel_size % num_distributed_optimizer_instances == 0
                ), "Expert data parallel size should be divisible by partial DistOpt shard factor"
        intra_partial_expert_data_parallel_size = (expert_data_parallel_size // num_distributed_optimizer_instances)

        for ranks in self.expert_decoder_rank_generator.get_ranks('dp'):
            group = self._create_group(
                ranks,
                timeout=timeout,
                pg_options=self._get_pg_options("ep_dp", pg_comm_cfgs),
                group_desc="EXPERT_DATA_PARALLEL_GROUP",
            )
            if create_gloo_process_groups:
                group_gloo = self._create_group(ranks, backend="gloo", group_desc="EXPERT_DATA_PARALLEL_GROUP_GLOO")
            else:
                group_gloo = None
            if rank in ranks:
                self.expert_data_parallel_group = group
                self.expert_data_parallel_group_gloo = group_gloo

            if num_distributed_optimizer_instances > 1:
                hierarchical_groups, hierarchical_groups_gloo = self._create_hierarchical_groups(
                    rank,
                    ranks,
                    [intra_partial_expert_data_parallel_size, num_distributed_optimizer_instances],
                    create_gloo_process_groups=create_gloo_process_groups,
                    pg_options=[
                        self._get_pg_options("intra_ep_dp", pg_comm_cfgs),
                        self._get_pg_options("inter_ep_dp", pg_comm_cfgs),
                    ],
                    timeout=timeout,
                    group_desc="EXPERT_DATA_PARALLEL_GROUP",
                )
                if rank in ranks:
                    self.intra_partial_expert_data_parallel_group = hierarchical_groups[0]
                    self.intra_partial_expert_data_parallel_group_gloo = hierarchical_groups_gloo[0]
                    self.inter_partial_expert_data_parallel_group = hierarchical_groups[1]
            else:
                self.intra_partial_expert_data_parallel_group = self.expert_data_parallel_group
                self.intra_partial_expert_data_parallel_group_gloo = self.expert_data_parallel_group_gloo

        # Build intra distributed optimizer instance group
        assert self.intra_distributed_optimizer_instance_group is None, "Intra distributed optimizer instance group is already initialized"
        model_parallel_group_id = 0
        intra_dist_opt_ranks = []
        for ranks in self.expert_decoder_rank_generator.get_ranks('tp-ep-pp'):
            model_parallel_group_id += 1
            intra_dist_opt_ranks.extend(ranks)
            if model_parallel_group_id % intra_partial_expert_data_parallel_size == 0:
                intra_dist_opt_instance_group = self._create_group(
                    intra_dist_opt_ranks,
                    timeout=timeout,
                    pg_options=self._get_pg_options("intra_dist_opt_instance", pg_comm_cfgs),
                    group_desc="INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP",
                )
                if rank in intra_dist_opt_ranks:
                    self.intra_distributed_optimizer_instance_group = intra_dist_opt_instance_group
                intra_dist_opt_ranks = []

        # Build sequence parallel groups
        if sequence_parallel_size > 1:
            assert self.sequence_parallel_group is None, "sequence parallel group is already initialized"
            assert self.sequence_and_data_parallel_group is None, "sequence and data parallel group is already initialized"

            if world_size < sequence_parallel_size:
                raise RuntimeError(
                    f"world_size ({world_size}) is less than sequence_parallel_size ({sequence_parallel_size})")

            if world_size % sequence_parallel_size != 0:
                raise RuntimeError(
                    f"world_size ({world_size}) is not divisible by sequence_parallel_size ({sequence_parallel_size})")

            # SP groups use consecutive ranks
            # Number of SP groups = data_parallel_size (each DP rank has its own SP group)
            num_sequence_parallel_groups = data_parallel_size
            sequence_and_data_parallel_size = world_size
            num_sequence_and_data_parallel_groups = 1

            # Build the sequence parallel groups using consecutive ranks
            # SP uses consecutive rank grouping, not orthogonal grouping like TP/PP/CP
            for i in range(num_sequence_parallel_groups):
                ranks = list(range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size))
                group = self._create_group(
                    ranks,
                    timeout=timeout,
                    pg_options=self._get_pg_options("sp", pg_comm_cfgs),
                    group_desc="SEQUENCE_PARALLEL_GROUP",
                )
                if rank in ranks:
                    self.sequence_parallel_group = group

            # Build the sequence and data parallel groups
            for i in range(num_sequence_and_data_parallel_groups):
                ranks = list(range(i * sequence_and_data_parallel_size, (i + 1) * sequence_and_data_parallel_size))
                group = self._create_group(
                    ranks,
                    timeout=timeout,
                    pg_options=self._get_pg_options("sp_dp", pg_comm_cfgs),
                    group_desc="SEQUENCE_AND_DATA_PARALLEL_GROUP",
                )
                if rank in ranks:
                    self.sequence_and_data_parallel_group = group

        # Initialize global memory buffer
        self._set_global_memory_buffer()

    def _set_global_memory_buffer(self):
        """Initialize global buffer."""
        assert self.global_memory_buffer is None, "global memory buffer is already initialized"
        self.global_memory_buffer = GlobalMemoryBuffer()

    # Getter methods for process groups
    def get_model_parallel_group(self, check_initialized=True):
        """Get the model-parallel group the caller rank belongs to."""
        if check_initialized:
            assert self.model_parallel_group is not None, "model parallel group is not initialized"
        return self.model_parallel_group

    def get_tensor_model_parallel_group(self, check_initialized=True):
        """Get the tensor-model-parallel group the caller rank belongs to."""
        if check_initialized:
            assert self.tensor_model_parallel_group is not None, "tensor model parallel group is not initialized"
        return self.tensor_model_parallel_group

    def get_pipeline_model_parallel_group(self, check_initialized=True):
        """Get the pipeline-model-parallel group the caller rank belongs to."""
        if check_initialized:
            assert self.pipeline_model_parallel_group is not None, "pipeline_model parallel group is not initialized"
        return self.pipeline_model_parallel_group

    def get_data_parallel_group(self, with_context_parallel=False, partial_data_parallel=False):
        """Get the data-parallel group the caller rank belongs to."""
        if with_context_parallel:
            if partial_data_parallel:
                assert self.intra_partial_data_parallel_group_with_cp is not None, "Intra partial data parallel group is not initialized"
                return self.intra_partial_data_parallel_group_with_cp
            assert self.data_parallel_group_with_cp is not None, "data parallel group with context parallel combined is not initialized"
            return self.data_parallel_group_with_cp
        else:
            assert self.data_parallel_group is not None, "data parallel group is not initialized"
            assert partial_data_parallel == False, "Partial DP for Optimizer needs to include CP"
            return self.data_parallel_group

    def get_context_parallel_group(self, check_initialized=True):
        """Get the context-parallel group the caller rank belongs to."""
        if check_initialized:
            assert self.context_parallel_group is not None, "context parallel group is not initialized"
        return self.context_parallel_group

    def get_sequence_parallel_group(self, check_initialized=True):
        """Get the sequence-parallel group the caller rank belongs to."""
        if check_initialized:
            assert self.sequence_parallel_group is not None, "sequence parallel group is not initialized"
        return self.sequence_parallel_group

    def get_sequence_and_data_parallel_group(self, check_initialized=True):
        """Get the sequence and data parallel group the caller rank belongs to."""
        if check_initialized:
            assert self.sequence_and_data_parallel_group is not None, "sequence and data parallel group is not initialized"
        return self.sequence_and_data_parallel_group

    def get_embedding_group(self, check_initialized=True):
        """Get the embedding group the caller rank belongs to."""
        if check_initialized:
            assert self.embedding_group is not None, "embedding group is not initialized"
        return self.embedding_group

    def get_tensor_and_data_parallel_group(self, check_initialized=True, with_context_parallel=False):
        """Get the tensor- and data-parallel group the caller rank belongs to."""
        if with_context_parallel:
            if check_initialized:
                assert self.tensor_and_data_parallel_group_with_cp is not None, 'tensor and data parallel group is not initialized'
            return self.tensor_and_data_parallel_group_with_cp
        else:
            if check_initialized:
                assert self.tensor_and_data_parallel_group is not None, 'tensor and data parallel group is not initialized'
            return self.tensor_and_data_parallel_group

    def get_tensor_and_context_parallel_group(self, check_initialized=True):
        """Get the tensor- and context-parallel group the caller rank belongs to."""
        if check_initialized:
            assert self.tensor_and_context_parallel_group is not None, "tensor and context parallel group is not initialized"
        return self.tensor_and_context_parallel_group

    # Getter methods for world sizes and ranks
    def get_tensor_model_parallel_world_size(self):
        """Return world size for the tensor-model-parallel group."""
        if self.mpu_tensor_model_parallel_world_size is not None:
            return self.mpu_tensor_model_parallel_world_size
        return self.get_tensor_model_parallel_group().size()

    def get_pipeline_model_parallel_world_size(self):
        """Return world size for the pipeline-model-parallel group."""
        if self.mpu_pipeline_model_parallel_world_size is not None:
            return self.mpu_pipeline_model_parallel_world_size
        return self.get_pipeline_model_parallel_group().size()

    def get_tensor_model_parallel_rank(self):
        """Return caller's rank for the tensor-model-parallel group."""
        if self.mpu_tensor_model_parallel_rank is not None:
            return self.mpu_tensor_model_parallel_rank
        return self.get_tensor_model_parallel_group().rank()

    def get_pipeline_model_parallel_rank(self):
        """Return caller's rank for the pipeline-model-parallel group."""
        if self.mpu_pipeline_model_parallel_rank is not None:
            return self.mpu_pipeline_model_parallel_rank
        return dist.get_rank(group=self.get_pipeline_model_parallel_group())

    def get_data_parallel_world_size(self, with_context_parallel=False, partial_data_parallel=False):
        """Return world size for the data parallel group."""
        if self.mpu_data_parallel_world_size is not None:
            return self.mpu_data_parallel_world_size
        if dist.is_available() and dist.is_initialized():
            return self.get_data_parallel_group(with_context_parallel=with_context_parallel,
                                                partial_data_parallel=partial_data_parallel).size()
        else:
            return 0

    def get_data_parallel_rank(self, with_context_parallel=False, partial_data_parallel=False):
        """Return caller's rank in the data-parallel group."""
        if self.mpu_data_parallel_rank is not None:
            return self.mpu_data_parallel_rank
        if dist.is_available() and dist.is_initialized():
            return self.get_data_parallel_group(with_context_parallel=with_context_parallel,
                                                partial_data_parallel=partial_data_parallel).rank()
        else:
            return 0

    def get_context_parallel_world_size(self):
        """Return world size for the context parallel group."""
        if dist.is_available() and dist.is_initialized():
            return self.get_context_parallel_group().size()
        else:
            return 0

    def get_context_parallel_rank(self):
        """Return caller's rank in the context-parallel group."""
        if dist.is_available() and dist.is_initialized():
            return self.get_context_parallel_group().rank()
        else:
            return 0

    def get_sequence_parallel_world_size(self):
        """Return world size for the sequence parallel group."""
        if dist.is_available() and dist.is_initialized():
            if self.sequence_parallel_group is not None:
                return self.get_sequence_parallel_group().size()
        return 1

    def get_sequence_parallel_rank(self):
        """Return caller's rank in the sequence-parallel group."""
        if dist.is_available() and dist.is_initialized():
            if self.sequence_parallel_group is not None:
                return self.get_sequence_parallel_group().rank()
        return 0

    def get_sequence_and_data_parallel_world_size(self):
        """Return world size for the sequence and data parallel group."""
        if dist.is_available() and dist.is_initialized():
            if self.sequence_and_data_parallel_group is not None:
                return self.get_sequence_and_data_parallel_group().size()
        return 0

    def get_sequence_and_data_parallel_rank(self):
        """Return caller's rank in the sequence and data parallel group."""
        if dist.is_available() and dist.is_initialized():
            if self.sequence_and_data_parallel_group is not None:
                return self.get_sequence_and_data_parallel_group().rank()
        return 0

    def is_initialized(self):
        """Check if parallel state has been initialized"""
        return self.data_parallel_group is not None

    def initialize_all_to_all_groups(self):
        """Initialize All-to-All groups for quantized gradient communication.

        Creates local and global All-to-All groups based on node topology:
        - Local groups: intra-node communication (NVLink/NVSwitch)
        - Global groups: inter-node communication (cross-node)

        Used by ZeRO++ when zero_quantized_gradients is enabled.

        Returns:
            Dictionary of All-to-All groups
        """
        if self.all_to_all_initialized:
            return self.all_to_all_groups

        assert dist.is_initialized(), 'dist is not initialized'

        device_per_node = get_accelerator().device_count()
        world_size = dist.get_world_size()
        num_nodes = world_size // device_per_node

        if num_nodes == 0 and world_size > 0:
            # Single incomplete node
            assert world_size >= 1, 'num_gpus must >=1, cannot initialize All-To-All'
            ranks = list(range(world_size))
            self.all_to_all_groups['local_0'] = self._create_group(ranks)

        elif num_nodes == 1:
            # Exactly one node
            assert world_size == device_per_node, 'num_gpus not equal to device per node, cannot initialize All-To-All'
            ranks = list(range(device_per_node))
            self.all_to_all_groups['local_0'] = self._create_group(ranks)

        else:
            # Multiple nodes: create both local and global groups
            assert world_size > device_per_node, 'num_nodes<2 cannot initialize All-To-All'

            # Local groups (intra-node)
            for node_id in range(num_nodes):
                local_ranks = [j + device_per_node * node_id for j in range(device_per_node)]
                self.all_to_all_groups[f"local_{node_id}"] = self._create_group(local_ranks)

            # Global groups (inter-node)
            for device_id in range(device_per_node):
                global_ranks = [device_id + j * device_per_node for j in range(num_nodes)]
                self.all_to_all_groups[f"global_{device_id}"] = self._create_group(global_ranks)

        self.all_to_all_initialized = True
        return self.all_to_all_groups

    def get_all_to_all_groups(self):
        """Get All-to-All groups dictionary.

        Initializes the groups if not already initialized.

        Returns:
            Dictionary of All-to-All groups
        """
        if not self.all_to_all_initialized:
            self.initialize_all_to_all_groups()
        return self.all_to_all_groups

    def get_global_memory_buffer(self):
        """Return the global GlobalMemoryBuffer object"""
        assert self.global_memory_buffer is not None, "global memory buffer is not initialized"
        return self.global_memory_buffer

    # Expert-related getter methods
    def get_expert_model_parallel_group(self, check_initialized=True):
        """Get the expert-model-parallel group the caller rank belongs to."""
        if check_initialized:
            assert self.expert_model_parallel_group is not None, "expert model parallel group is not initialized"
        return self.expert_model_parallel_group

    def get_expert_model_parallel_world_size(self):
        """Return world size for the expert-model-parallel group."""
        if self.mpu_expert_model_parallel_world_size is not None:
            return self.mpu_expert_model_parallel_world_size
        if dist.is_available() and dist.is_initialized():
            return self.get_expert_model_parallel_group().size()
        else:
            return 0

    def get_expert_model_parallel_rank(self):
        """Return caller's rank in the expert-model-parallel group."""
        if self.mpu_expert_model_parallel_rank is not None:
            return self.mpu_expert_model_parallel_rank
        if dist.is_available() and dist.is_initialized():
            return self.get_expert_model_parallel_group().rank()
        else:
            return 0

    def get_expert_tensor_parallel_group(self, check_initialized=True):
        """Get the expert-tensor-parallel group the caller rank belongs to."""
        if check_initialized:
            assert self.expert_tensor_parallel_group is not None, "Expert tensor parallel group is not initialized"
        return self.expert_tensor_parallel_group

    def get_expert_tensor_parallel_world_size(self):
        """Return world size for the expert tensor parallel group."""
        if self.mpu_expert_tensor_parallel_world_size is not None:
            return self.mpu_expert_tensor_parallel_world_size
        if not self.expert_tensor_parallel_group:
            return self.mpu_tensor_model_parallel_world_size
        else:
            return self.get_expert_tensor_parallel_group().size()

    def get_expert_tensor_parallel_rank(self):
        """Return my rank for the expert tensor parallel group."""
        if self.mpu_expert_tensor_parallel_rank is not None:
            return self.mpu_expert_tensor_parallel_rank
        if not self.expert_tensor_parallel_group:
            return self.mpu_tensor_model_parallel_rank
        else:
            return self.get_expert_tensor_parallel_group().rank()

    def get_expert_data_parallel_group(self, check_initialized=True, partial_expert_data_parallel=False):
        """Get expert data parallel group."""
        if partial_expert_data_parallel:
            if check_initialized:
                assert self.intra_partial_expert_data_parallel_group is not None, "Intra partial expert data parallel group is not initialized"
            return self.intra_partial_expert_data_parallel_group
        else:
            if check_initialized:
                assert self.expert_data_parallel_group is not None, "Expert data parallel group is not initialized"
            return self.expert_data_parallel_group

    def get_expert_data_parallel_rank(self, partial_expert_data_parallel=False):
        """Return caller's rank in the expert data parallel group."""
        if dist.is_available() and dist.is_initialized():
            return self.get_expert_data_parallel_group(
                partial_expert_data_parallel=partial_expert_data_parallel).rank()
        else:
            return 0

    def get_expert_data_parallel_world_size(self, partial_expert_data_parallel=False):
        """Return world size for the expert data parallel group."""
        if dist.is_available() and dist.is_initialized():
            return self.get_expert_data_parallel_group(
                partial_expert_data_parallel=partial_expert_data_parallel).size()
        else:
            return 0


# Convenience function to create a singleton instance
_parallel_state_instance = None


def get_parallel_state() -> ParallelState:
    """Get or create the global ParallelState instance."""
    global _parallel_state_instance
    if _parallel_state_instance is None:
        _parallel_state_instance = ParallelState()
    return _parallel_state_instance
