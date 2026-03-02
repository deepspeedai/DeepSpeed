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
--------------------------
Optional distributed backend initialization separate from ``deepspeed.initialize()``. Useful in scenarios where the user wants to use torch distributed calls before calling ``deepspeed.initialize()``, such as when using model parallelism, pipeline parallelism, or certain data loader scenarios.

.. autofunction:: deepspeed.init_distributed


.. _parallel-state-init:

Parallel State Initialization
-----------------------------
DeepSpeed provides a built-in ``ParallelState`` class for Megatron-style process group management
covering tensor, pipeline, data, sequence, context, and expert parallelism.

Use ``initialize_parallel_state_from_config`` to create and initialize a ``ParallelState`` from
a DeepSpeed config dictionary (or ``DeepSpeedConfig`` object). The returned instance implements
the ``mpu`` interface and can be passed directly to ``deepspeed.initialize(mpu=...)``.

Example usage:

.. code-block:: python

    from deepspeed.utils import parallel_state_deepspeed as ps

    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "tensor_parallel": {"autotp_size": 4},
    }

    # Initialize and use as mpu
    parallel_state = ps.initialize_parallel_state_from_config(config_dict)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=config_dict,
        mpu=parallel_state,
    )

.. autofunction:: deepspeed.utils.parallel_state_deepspeed.initialize_parallel_state_from_config

.. autoclass:: deepspeed.utils.parallel_state.ParallelState
   :members: initialize_model_parallel, is_initialized, get_tensor_model_parallel_group, get_data_parallel_group, get_pipeline_model_parallel_group, get_sequence_parallel_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank, get_data_parallel_world_size, get_data_parallel_rank, get_pipeline_model_parallel_world_size, get_pipeline_model_parallel_rank
   :noindex:
