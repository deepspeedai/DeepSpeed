MiniMax-M3 (``minimax_m3_vl``)
==============================

``deepspeed.models.minimax_m3_vl`` trains MiniMax-M3's sparse attention layers with memory and
compute that grow linearly with the sequence length, in plain PyTorch. The transformers
``MiniMaxM3VLAttention`` expands the indexer's block selection into a dense
``[batch, heads, seq, seq]`` mask (32 GiB at 16,384 tokens for MiniMax-M3's 64 heads) and runs dense
attention under it. The replacement gathers only the selected key blocks (16 blocks of 128 keys per
query and key-value head) and takes the softmax over them. Its backward pass recomputes the
probabilities from the saved log-sum-exp and adds the key and value gradients in float32. It keeps
the parameters and attribute names of the module it replaces, so checkpoints load unchanged; dense
attention layers are left as they are.

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
* It is not a fused kernel: every query gathers its selected keys and values, one chunk of query
  rows at a time.

.. autofunction:: deepspeed.models.minimax_m3_vl.replace_attention

.. autofunction:: deepspeed.models.minimax_m3_vl.block_sparse_attention

.. autoclass:: deepspeed.models.minimax_m3_vl.DeepSpeedMiniMaxM3VLAttention
