Transformer Kernels
===================

The transformer kernel API in DeepSpeed can be used to create BERT transformer layer for
more efficient pre-training and fine-tuning, it includes the transformer layer configurations and
transformer layer module initialization.

Here we present the transformer kernel API.
Please see the `BERT pre-training tutorial <https://www.deepspeed.ai/tutorials/bert-pretraining/>`_ for usage details.

DeepSpeed Transformer Config
----------------------------
.. autoclass:: deepspeed.DeepSpeedTransformerConfig

DeepSpeed Transformer Layer
----------------------------
.. autoclass:: deepspeed.DeepSpeedTransformerLayer
