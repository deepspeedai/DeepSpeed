Initializing DeepSpeed
======================

The entrypoint for all training with DeepSpeed is ``deepspeed.initialize()``.

Example usage:

.. code-block:: python

    model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                         model=net,
                                                         model_parameters=net.parameters())

.. autofunction:: deepspeed.initialize
