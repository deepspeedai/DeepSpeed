MiniMax-M3 (``minimax_m3_vl``)
==============================

``deepspeed.models.minimax_m3_vl`` trains MiniMax-M3's sparse attention layers with memory and
compute that grow linearly with the sequence length. The transformers
``MiniMaxM3VLAttention`` expands the indexer's block selection into a dense
``[batch, heads, seq, seq]`` mask (32 GiB at 16,384 tokens for MiniMax-M3's 64 heads) and runs dense
attention under it. The replacement reads only the selected key blocks (16 blocks of 128 keys per
query and key-value head) and takes the softmax over them, with one of two implementations
(``kernel``):

* ``selected_attention``: Triton kernels adapted from flash-linear-attention's native sparse attention.
  The query heads that share a key-value head are computed together, nothing is copied per query, and
  the key and value gradients of a block are summed in float32 over the queries that selected it. On
  one H200 at MiniMax-M3's shapes (16,384 queries over 65,536 keys) the forward takes 12.9 ms and the
  backward 32.5 ms, against 84.7 ms and 355.6 ms for the PyTorch path, with the same error against a
  float32 reference.
* ``block_sparse_attention``: plain PyTorch, one chunk of query rows at a time; every query gathers its
  selected keys and values, and the backward recomputes the probabilities from the saved
  log-sum-exp and adds the key and value gradients in float32.

``kernel="auto"`` (the default) takes the Triton kernels for bf16 or fp16 on an accelerator that runs
Triton when the shapes fit them (query heads per key-value head, head dim and block size each a power of
two of at least 16), and the PyTorch path otherwise. The replacement keeps the parameters and attribute
names of the module it replaces, so checkpoints load unchanged; dense attention layers are left as they
are.

The indexer picks the blocks from float32 scores of every query against every earlier key, as in
transformers (also under an engine's bf16 ``torch_autocast``). With ``kernel="auto"`` or ``"triton"`` and
bf16 or fp16 inputs on an accelerator that runs Triton, ``index_block_max`` computes each query's
per-block maximum score in one kernel instead of materializing the float32 scores (2 GiB per 2,048
queries at 65,536 keys): the products are exact and the sums float32, so only near-ties between blocks
can be ordered differently. On one H200, for 16,384 queries over 65,536 keys, one layer's selection
takes 2.8-5.7 ms instead of 60.5 ms.

.. code-block:: python

    import deepspeed
    from deepspeed.models.minimax_m3_vl import replace_attention

    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config)  # also works under deepspeed.zero.Init
    replace_attention(model)
    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=model.parameters())

**Limits:**

* The replacement runs only without a key-value cache and without an attention mask. Otherwise the
  stock forward runs unchanged: decoding works, and a padded batch or the eager attention
  implementation falls back to the dense mask. Train with ``attn_implementation="sdpa"`` and
  unpadded sequences.
* The indexer must have one head per key-value head (MiniMax-M3: 4 and 4).
* The indexer is not trained, as in transformers: its output is integer block ids.
* Attention dropout must be 0.

**Splitting each sequence over GPUs (context parallel).** ``set_context_parallel_group(group)`` makes
every replaced module split each sequence over the ranks of ``group``: rank ``r`` of a group of ``n``
ranks holds tokens ``[r * S / n, (r + 1) * S / n)`` of the group's sequence (batch size 1) and calls
the model with those tokens and their global ``position_ids``. Queries stay on their rank; the keys and
values and the indexer's key are gathered over the group, and their gradients are reduce-scattered
back. The dense attention layers need the same, so call ``replace_attention(model, dense=True)``. The
rest of the model works per token and needs no change; the data loader must give the ranks of a group
the same sequence.

.. code-block:: python

    import deepspeed.comm as dist
    from deepspeed.models.minimax_m3_vl import replace_attention, set_context_parallel_group

    replace_attention(model, dense=True)
    # groups of 4 consecutive ranks share one sequence
    for first in range(0, dist.get_world_size(), 4):
        group = dist.new_group(list(range(first, first + 4)))
        if first <= dist.get_rank() < first + 4:
            set_context_parallel_group(group)

.. autofunction:: deepspeed.models.minimax_m3_vl.replace_attention

.. autofunction:: deepspeed.models.minimax_m3_vl.set_context_parallel_group

.. autofunction:: deepspeed.models.minimax_m3_vl.selected_attention

.. autofunction:: deepspeed.models.minimax_m3_vl.block_sparse_attention

.. autofunction:: deepspeed.models.minimax_m3_vl.index_block_max

.. autoclass:: deepspeed.models.minimax_m3_vl.DeepSpeedMiniMaxM3VLAttention

.. autoclass:: deepspeed.models.minimax_m3_vl.DeepSpeedMiniMaxM3VLDenseAttention
