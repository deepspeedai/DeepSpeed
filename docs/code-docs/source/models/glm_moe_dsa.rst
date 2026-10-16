GLM-5.2 (``glm_moe_dsa``)
=========================

``deepspeed.models.glm_moe_dsa`` trains GLM-5.2's DeepSeek Sparse Attention (DSA) with TileLang
kernels. The transformers ``GlmMoeDsaAttention`` builds a dense ``[batch, index_heads, seq, seq]``
float32 indexer score tensor (128 GiB at 32,768 tokens for GLM-5.2's 32 index heads) and a dense
attention mask. The replacement attends only to the 2,048 keys the indexer selects for each query,
in the absorbed multi-head latent attention form, so its memory grows linearly with the sequence
length. It keeps the parameters and attribute names of the module it replaces, so checkpoints load
unchanged.

.. code-block:: python

    import deepspeed
    from deepspeed.models.glm_moe_dsa import replace_attention

    model = AutoModelForCausalLM.from_config(config)  # also works under deepspeed.zero.Init
    replace_attention(model)
    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=model.parameters())

**Requirements:** TileLang (``pip install tilelang``), a CUDA device, and bf16 training.

**Limits:**

* Training only: no key-value cache. Every sequence in a batch must be a full causal sequence: the
  module raises on a padding mask and on packed sequences (position ids that restart).
* ``output_attentions=True`` raises: the sparse kernel never forms the attention weights.
* The indexer is not trained, as in transformers: its top-k selection has no gradient.
* Attention dropout must be 0.
* Replace every DSA layer of a model or none: a layer that reuses the previous layer's selection
  must receive it in the same encoding.
* Tensor parallelism inside the module is not supported yet.

**Splitting each sequence over GPUs (context parallel).** At long sequence lengths one sequence may
not fit on one GPU. ``set_context_parallel_group(group)`` makes every replaced module split each
sequence over the ranks of ``group``: rank ``r`` of a group of ``n`` ranks holds tokens
``[r * S / n, (r + 1) * S / n)`` of the group's sequence (batch size 1) and calls the model with those
tokens, their global ``position_ids`` and, for the loss, the next token of the whole sequence as the
label of its last token. Queries stay on their rank; the latent keys (``kv_lora_rank +
qk_rope_head_dim`` values per token) and the indexer keys are gathered over the group, and their
gradients are reduce-scattered back. The rest of the model works per token and needs no change; the
data loader must give the ranks of a group the same sequence.

.. code-block:: python

    import deepspeed.comm as dist
    from deepspeed.models.glm_moe_dsa import set_context_parallel_group

    # pairs of consecutive ranks share one sequence
    for first in range(0, dist.get_world_size(), 2):
        group = dist.new_group([first, first + 1])
        if dist.get_rank() in (first, first + 1):
            set_context_parallel_group(group)

.. autofunction:: deepspeed.models.glm_moe_dsa.replace_attention

.. autofunction:: deepspeed.models.glm_moe_dsa.set_context_parallel_group

.. autoclass:: deepspeed.models.glm_moe_dsa.DeepSpeedGlmMoeDsaAttention
