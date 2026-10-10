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

* Training only: no key-value cache. The attention mask is not read, so every sequence in a batch
  must be a full causal sequence (no padding, no packing).
* The indexer is not trained, as in transformers: its top-k selection has no gradient.
* Attention dropout must be 0.
* Replace every DSA layer of a model or none: a layer that reuses the previous layer's selection
  must receive it in the same encoding.
* Tensor parallelism inside the module is not supported yet.

.. autofunction:: deepspeed.models.glm_moe_dsa.replace_attention

.. autoclass:: deepspeed.models.glm_moe_dsa.DeepSpeedGlmMoeDsaAttention
