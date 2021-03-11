ZeRO-3 Offload
##############

The Zero Redundancy Optimizer (ZeRO) removes the memory redundancies across
data-parallel processes by partitioning the three model states (optimizer
states, gradients, and parameters) across data-parallel processes instead of
replicating them. By doing this, it boosts memory efficiency compared to
classic data-parallelism while retaining its computational granularity and
communication efficiency.

ZeRO-Offload further increases memory efficiency by offloading the
optimizer's states and computations to the CPU. The model parameters can also
be offloaded for even more memory savings!

For more information on our algorithms, please see our papers on `ZeRO
<https://arxiv.org/abs/1910.02054>`_ and `ZeRO-Offload
<https://arxiv.org/abs/2101.06840>`_.

Getting Started
---------------

If you are new to DeepSpeed, check out our `Getting Started <https://www.deepspeed.ai/getting-started/>`_ page.

Once you are training with DeepSpeed, enabling ZeRO-3 Offload is as simple as enabling it
in your DeepSpeed configuration! Below are a few examples of ZeRO-3 configurations. Please see
our `config guide <https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training>`_
for a complete list of options for configuration and performance tuning.

.. note::
        ZeRO-3 Offload works best with our heavily optimized
        :class:`deepspeed.ops.adam.DeepSpeedCPUAdam` optimizer. We recommend using
        our `optimizer config <https://www.deepspeed.ai/docs/config-json/#optimizer-parameters>`_
        to instruct :meth:`deepspeed.initialize` to build the optimizer for you.


Example ZeRO-3 Offload Configurations
=====================================

#. Use ZeRO to partition the optimizer states (stage 1), gradients (stage 2),
   and parameters (stage 3).

    .. code-block:: python
        :emphasize-lines: 3

        {
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": true
            },
            "fp16": {
                "enabled": true
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                "lr": 0.001,
                "betas": [
                    0.8,
                    0.999
                ],
                "eps": 1e-8,
                "weight_decay": 3e-7
                }
            },
            ...
        }


#. Additionally offload the optimizer states and computations to the CPU.

    .. code-block:: python
        :emphasize-lines:  4

        {
            "zero_optimization": {
                "stage": 3,
                "cpu_offload": true,
                "overlap_comm": true
            },
            ...
        }


#. Save even more memory by offloading parameters to the CPU memory.

    .. code-block:: python
        :emphasize-lines:  5

        {
            "zero_optimization": {
                "stage": 3,
                "cpu_offload": true,
                "cpu_offload_params": true,
                "overlap_comm": true
            },
            ...
        }



Assumptions
===========

DeepSpeed automatically coordinates the collection (*i.e.,* all-gather),
partitioning (*i.e.,* scatter), and offloading of parameters at the
granularity of (sub)module ``forward()`` methods. The backward pass is
handled similarly. This strategy has two underlying assumptions:

#. The forward and backward passes of submodules must individually fit in device memory.

#. A module's parameters are only accessed within its own ``__init__`` and ``forward()`` methods.
   Otherwise, DeepSpeed must be instructed to collect and re-partition the parameter.
   See :ref:`external-parameters` for manually coordinating parameters.


Constructing Massive Models
---------------------------

ZeRO-3 enables massive models whose parameters exceed the size of individual
nodes in a system. For the typical case of training without model parallelism,
you can simply allocate your model in our context:

.. code-block:: python

    with deepspeed.zero.Init():
        model = MyLargeModel()



.. autoclass:: deepspeed.zero.Init
    :members:


.. _external-parameters:

Manual Parameter Coordination
-----------------------------

Most models require no modification to be trained with ZeRO-3. However, in
some cases one may need to access model weights outside of the training loop,
or to share weights across submodules during training. DeepSpeed has
several mechanisms to coordinate partitioned weights for ZeRO-3.


Gathering Parameters
====================

DeepSpeed provides mechanisms for collecting (or *gathering*) a partitioned parameter.

Some models partitioned with :class:`deepspeed.zero.Init` may need to access
a module’s weights outside of the class constructor or its ``forward()``
method. We refer to these weights as **external parameters**, since these
parameters are accessed outside of the module that created them. To do so, use
:class:`deepspeed.zero.GatheredParameters` or :meth:`deepspeed.zero.register_external_parameter`.

.. autoclass:: deepspeed.zero.GatheredParameters
    :members:


Registering External Parameters
===============================

Consider the following pattern common in language models such as GPT:

.. code-block:: python

    class LanguageModel(torch.nn.Module):
        ...
        def forward(self, inputs):
            embeds = self.embeddings(inputs)
            ...
            logits = compute_logits(output, self.embeddings.weight)
            ...


The tensor ``embeddings.weight`` is used in both ``embeddings.forward()`` and
``compute_logits()``. We call ``embeddings.weight`` an *external* parameter
because it is used in the training loop outside of its owning module's
forward pass. DeepSpeed will coordinate external parameters if they are
registered prior to the first forward pass.

.. autofunction:: deepspeed.zero.register_external_parameter

.. autofunction:: deepspeed.zero.unregister_external_parameter
