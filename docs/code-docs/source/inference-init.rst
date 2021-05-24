Inference Setup
-----------------------
The entrypoint for inference with DeepSpeed is ``deepspeed.init_inference()``.

Example usage:

.. code-block:: python

    engine = deepspeed.init_inference(model=net)

.. autofunction:: deepspeed.init_inference
