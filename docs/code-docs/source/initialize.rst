Training Setup
==============

.. _deepspeed-args:

Argument Parsing
----------------
DeepSpeed uses the `argparse <https://docs.python.org/3/library/argparse.html>`_ library to
supply commandline configuration to the DeepSpeed runtime. Use ``deepspeed.add_config_arguments()``
to add DeepSpeed's builtin arguments to your application's parser.

.. code-block:: python

    parser = argparse.ArgumentParser(description='My training script.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

.. autofunction:: deepspeed.add_config_arguments


.. _deepspeed-init:

Training Initialization
-----------------------
The entrypoint for all training with DeepSpeed is ``deepspeed.initialize()``. Will initialize distributed backend if it is not initialized already.

Example usage:

.. code-block:: python

    model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                         model=net,
                                                         model_parameters=net.parameters())

.. autofunction:: deepspeed.initialize

Distributed Initialization
-----------------------
Optional distributed backend initialization separate from ``deepspeed.initialize()``. Useful in scenarios where the user wants to use torch distributed calls before calling ``deepspeed.initialize()``, such as when using model parallelism, pipeline parallelism, or certain data loader scenarios.

.. autofunction:: deepspeed.init_distributed
