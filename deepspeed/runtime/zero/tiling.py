import torch
import deepspeed
from deepspeed.runtime.utils import partition_uniform as partition


def split_tensor_along_last_dim(tensor, partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension. Adapted from Megatron-LM.

    Arguments:
        tensor: input tensor.
        partitions: list of partition sizes to supply to torch.split
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    # Split.
    tensor_list = torch.split(tensor, partitions, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class TiledLinear(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 in_splits=1,
                 out_splits=1,
                 input_is_already_split=False,
                 combine_out_splits=True,
                 linear_cls=torch.nn.Linear,
                 init_linear=None,
                 **kwargs):
        """A replacement for ``torch.nn.Linear`` that works with ZeRO-3 to reduce
        memory requirements via tiling.

        TiledLinear breaks the input and output dimensions of a linear layer
        into tiles that are processed in sequence. This class enables huge
        linear layers when combined with ZeRO-3 because inactive tiles can be
        partitioned and offloaded.

        .. note::
            We recommend using as few tiles as necessary. Tiling
            significantly reduces memory usage, but can reduce throughput
            for inexpensive layers. This due to the smaller kernels having
            less parallelism and lower arithmetic intensity, while
            introducing more frequent synchronization and communication.

        Args:
            in_features (int): See ``torch.nn.Linear``
            out_features (int): See ``torch.nn.Linear``
            bias (bool, optional): See ``torch.nn.Linear``
            in_splits (int, optional): The number of tiles along the input dimension. Defaults to 1.
            out_splits (int, optional): The number of tiles along the output dimension. Defaults to 1.
            input_is_already_split (bool, optional): If set to ``True``, assume that the ``input_`` in
                to ``forward()`` is already split into ``in_splits`` chunks. Defaults to ``False``.
            combine_out_splits (bool, optional): If set to ``False``, do not combine the ``out_splits`` outputs
                into a single tensor. Defaults to ``True``.
            linear_cls (class, optional): The underlying class to build individual tiles.
                Defaults to ``torch.nn.Linear``.
            init_linear (``torch.nn.Linear``, optional): If set, copy the parameters of
                ``init_linear``. Useful for debugging. Defaults to ``None``.
            kwargs (dict, optional): additional keyword arguments to provide to ``linear_cls()``.

        Raises:
            RuntimeError: ``in_splits`` must be within the range [1, in_features).
            RuntimeError: ``out_splits`` must be within the range of [1, out_features).
        """

        super().__init__()

        if (in_splits < 1) or (in_splits > in_features):
            raise RuntimeError('in splits must be in range [1, in_features].')
        if (out_splits < 1) or (out_splits > out_features):
            raise RuntimeError('out splits must be in range [1, out_features].')

        # global, not necessarily local
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.out_splits = out_splits
        self.in_splits = in_splits
        self.input_is_already_split = input_is_already_split
        self.combine_out_splits = combine_out_splits

        # Build partition-lists. These are CSR-style splits [0, part0, part1, ..., features]
        # For example, row_parts[p] gives the start of partition p and row_parts[p+1]
        # is the exclusive end.
        self.in_parts = partition(num_items=in_features, num_parts=in_splits)
        self.out_parts = partition(num_items=out_features, num_parts=out_splits)

        assert len(self.out_parts) == out_splits + 1
        assert len(self.in_parts) == in_splits + 1
        assert self.out_parts[0] == 0
        assert self.out_parts[out_splits] == out_features
        assert self.in_parts[in_splits] == in_features

        self.linears = torch.nn.ModuleList()
        for out_id in range(out_splits):
            self.linears.append(torch.nn.ModuleList())

            local_out_dim = self.out_parts[out_id + 1] - self.out_parts[out_id]

            for in_id in range(in_splits):
                #if input_size is split, we only need one bias
                local_bias = bias if in_id == (in_splits - 1) else False

                local_in_dim = self.in_parts[in_id + 1] - self.in_parts[in_id]
                local = linear_cls(local_in_dim,
                                   local_out_dim,
                                   bias=local_bias,
                                   **kwargs)
                self.linears[out_id].append(local)

        # Optionally initialize with a known tensor
        if init_linear is not None:
            self.copy_params_from(init_linear)

    def forward(self, input_):
        if self.in_splits > 1 and not self.input_is_already_split:
            split_sizes = [
                self.in_parts[p + 1] - self.in_parts[p] for p in range(self.in_splits)
            ]
            inputs = self._split_global_input(input_, split_sizes)
        elif self.in_splits > 1:
            inputs = input_
            assert len(inputs) == self.in_splits, f"Col splits {self.in_splits} does not match input splits {len(inputs)}"
        else:
            # no splits
            inputs = [input_]

        outputs = [None] * self.out_splits
        for out_id in range(self.out_splits):
            for in_id in range(self.in_splits):
                local_output = self.linears[out_id][in_id](inputs[in_id])

                outputs[out_id] = self._reduce_local_output(in_id=in_id,
                                                            out_id=out_id,
                                                            current_out=outputs[out_id],
                                                            new_out=local_output)

        if self.combine_out_splits:
            return self._combine_output_splits(outputs)

        return outputs

    def _split_global_input(self, input, split_sizes):
        """Partition an input tensor along the last dimension, aligned with given splits.

        Subclasses should override this method to account for new input types.

        Args:
            input (List[Tensor]): The tensor to partition along the last dimension.
            split_sizes (List[int]): The size of each partition.

        Returns:
            List[Any]: A list of the chunks of ``input``.
        """
        return split_tensor_along_last_dim(input, split_sizes)

    def _reduce_local_output(self, in_id, out_id, current_out, new_out):
        """Reduce (sum) a new local result into the existing local results.

        Subclasses should override this method.

        For a given ``out_id``, this method is called ``in_id-1`` times. The first input
        split is a simple assignment.

        Args:
            in_id (int): The input split that produced ``new_out``.
            out_id (int): The output split that produced ``new_out``.
            current_out (Any): The reduced form of all previous ``out_id`` results.
            new_out (Any): The local result from forward (``in_id``, ``out_id``)e

        Returns:
            Any: The combined result of ``current_out`` and ``new_out``.
        """

        if current_out is None:
            #this clone is necessary to preserve auto grad
            #there is some issue with inplace update for outputs that are views
            return new_out.clone()
        else:
            return current_out + new_out

    def _combine_output_splits(self, outputs):
        """Join the splits of the output into a single result.

        Args:
            outputs (List[Any]): The reduced outputs for each output split.

        Returns:
            Any: The combined outputs.
        """
        assert len(outputs) == self.out_splits
        return torch.cat(outputs, dim=-1)

    @torch.no_grad()
    def copy_params_from(self, other):
        """Copy the weight and bias data from ``other``.

        This is especially useful for reproducible initialization and testing.

        Equivalent to:

        .. code-block:: python

            with torch.no_grad():
                self.weight.copy_(other.weight)
                if self.bias is not None:
                    self.bias.copy_(other.bias)

        .. note::
            If ZeRO-3 is enabled, this is a collective operation and the
            updated parameters of data-parallel rank 0 will be visible on all
            ranks. See :class:`deepspeed.zero.GatheredParameters` for more
            information.


        Args:
            other (``torch.nn.Linear``): the linear layer to copy from.
        """
        assert hasattr(other, 'weight')
        assert other.weight.size() == (self.out_features, self.in_features)
        if self.use_bias:
            assert hasattr(other, 'bias')
            assert other.bias is not None
            assert other.bias.size() == (self.out_features, )
        else:
            assert other.bias is None

        for row in range(self.out_splits):
            rstart = self.out_parts[row]
            rstop = self.out_parts[row + 1]

            for col in range(self.in_splits):
                cstart = self.in_parts[col]
                cstop = self.in_parts[col + 1]

                local = self.linears[row][col]
                global_weight = other.weight[rstart:rstop, cstart:cstop]
                with deepspeed.zero.GatheredParameters(local.weight, modifier_rank=0):
                    local.weight.copy_(global_weight)

            if local.bias is not None:
                with deepspeed.zero.GatheredParameters(local.bias, modifier_rank=0):
                    local.bias.data.copy_(other.bias[rstart:rstop].data)


class TiledLinearReturnBias(TiledLinear):
    """Wrapper for a Linear class that returns its own bias parameter, such as
    used by Megatron-LM.
    """
    def _reduce_local_output(self, in_id, out_id, current_out, new_out):
        """Reduces output tensors, but not the returned bias. """
        if current_out is not None:
            old_tensor, old_bias = current_out
        else:
            old_tensor, old_bias = None, None

        assert isinstance(new_out, tuple)
        assert len(new_out) == 2

        tensor, bias = new_out
        assert tensor is not None

        tensor = super()._reduce_local_output(in_id=in_id,
                                              out_id=out_id,
                                              current_out=old_tensor,
                                              new_out=tensor)

        if bias is None:
            bias = old_bias

        return tensor, bias

    def _combine_output_splits(self, outputs):
        # stack output tensors
        tensors = [o[0] for o in outputs]
        tensor = super()._combine_output_splits(tensors)

        # stack biases if applicable
        biases = [o[1] for o in outputs if o[1] is not None]
        if len(biases) > 0:
            bias = super()._combine_output_splits(biases)
        else:
            bias = None

        return tensor, bias
