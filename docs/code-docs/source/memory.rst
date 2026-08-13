Memory Requirements
-----------------------


API To Estimate Memory Usage
============================

ZeRO2:

.. autofunction:: deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs_all_live

.. autofunction:: deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs_all_cold

Examples:

Let's try a 3B model with just 1 node with 8 gpus, using live model:

.. code-block:: bash

    python -c 'from transformers import AutoModel; \
    from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live; \
    model = AutoModel.from_pretrained("t5-3b"); \
    estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)'

    Estimated memory needed for params, optim states and gradients for a:
    HW: Setup with 1 node, 8 GPUs per node.
    SW: Model with 2851M total params.
      per CPU  |  per GPU |   Options
      127.48GB |   5.31GB | offload_optimizer=cpu
      127.48GB |  15.93GB | offload_optimizer=none

Now, without the actual model, which requires us to know ``total_params`` and
``largest_layer_params``, but we got those from the run above, so future estimators are now much
faster as we don't need to load the model.

.. code-block:: bash

    python -c 'from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_cold; \
    estimate_zero2_model_states_mem_needs_all_cold(total_params=2851e6, num_gpus_per_node=8, num_nodes=1)'

    Estimated memory needed for params, optim states and gradients for a:
    HW: Setup with 1 node, 8 GPUs per node.
    SW: Model with 2851M total params.
      per CPU  |  per GPU |   Options
      127.45GB |   5.31GB | offload_optimizer=cpu
      127.45GB |  15.93GB | offload_optimizer=none

There is a slight difference due to rounding - the actual live model has a few more params


ZeRO3:

.. autofunction:: deepspeed.runtime.zero.stage3.estimate_zero3_model_states_mem_needs_all_live

.. autofunction:: deepspeed.runtime.zero.stage3.estimate_zero3_model_states_mem_needs_all_cold

Examples:

Let's try a 3B model with just 1 node with 8 gpus, using live model:

.. code-block:: bash

    python -c 'from transformers import AutoModel; \
    from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
    model = AutoModel.from_pretrained("t5-3b"); \
    estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)'

    Estimated memory needed for params, optim states and gradients for a:
    HW: Setup with 1 node, 8 GPUs per node.
    SW: Model with 2851M total params, 32M largest layer params.
      per CPU  |  per GPU |   Options
       71.71GB |   0.12GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
      127.48GB |   0.12GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
       63.74GB |   0.79GB | offload_param=none, offload_optimizer=cpu , zero_init=1
      127.48GB |   0.79GB | offload_param=none, offload_optimizer=cpu , zero_init=0
        1.47GB |   6.10GB | offload_param=none, offload_optimizer=none, zero_init=1
      127.48GB |   6.10GB | offload_param=none, offload_optimizer=none, zero_init=0

Now, without the actual model, which requires us to know ``total_params`` and
``largest_layer_params``, but we got those from the run above, so future estimators are now much
faster as we don't need to load the model.

.. code-block:: bash

    python -c 'from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_cold; \
    estimate_zero3_model_states_mem_needs_all_cold(total_params=2851e6, largest_layer_params=32e6, num_gpus_per_node=8, num_nodes=1)'

    Estimated memory needed for params, optim states and gradients for a:
    HW: Setup with 1 node, 8 GPUs per node.
    SW: Model with 2851M total params, 32M largest layer params.
      per CPU  |  per GPU |   Options
       71.69GB |   0.12GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
      127.45GB |   0.12GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
       63.72GB |   0.78GB | offload_param=none, offload_optimizer=cpu , zero_init=1
      127.45GB |   0.78GB | offload_param=none, offload_optimizer=cpu , zero_init=0
        1.43GB |   6.09GB | offload_param=none, offload_optimizer=none, zero_init=1
      127.45GB |   6.09GB | offload_param=none, offload_optimizer=none, zero_init=0

There is a slight difference due to rounding - the actual live model has a few more params



Discussion
==========

Let's look in detail how the memory estimator API calculates these numbers and also discuss some additional numbers that aren't covered by the API.

In the following discussion:

- ``params`` - total number of model params, which can be calculated as:

.. code-block:: python

    print(sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()))

Some models already include the number of params in the model name, e.g. t5-11b (11B params), gpt-neo-1.3B (1.3B params), etc.

Also if the model weights are stored in ``fp32`` the other quick way to calculate the size of the model is to simply divide the size of the ``state_dict`` file by 4 (fp32 == 4 bytes). For example, you can see that `t5-11b's pytorch_model.bin <https://huggingface.co/t5-11b/tree/main>`__ is 42.1GB in size, so if we divide it by 4, we can immediately tell it's an 11B model.

The following calculations show how much memory is required by model params, gradients and optimizer states. In addition to those you will need enough memory to fit activation calculations and any temporary memory for intermediate calculations, which for long sequences could be very significant (e.g. could take the same amount of memory as params+grads+optim_states combined).

The optimizer states assume that ``Adam`` is used, where 4 bytes per parameter are used by momentum and another 4 by variance (8 in total).

Gradients at ``fp32`` take 4 bytes, and parameters take 2 bytes at ``fp16`` and 4 bytes at ``fp32``.

**GPU RAM**

The big question is how big of a model you can fit on the hardware you have? Or rather what size of a GPU RAM do you need to fit the desired model.


* ZeRO-2:

   - ``"offload_optimizer": {"device": "cpu"}``: 2 * params

   Example: a 40GB GPU can fit ~11B param model (regardless of how many GPUs are used). Here the model is loaded in ``fp16`` so just the model weights take about 22GB and the remaining 18GB are used by other components. You can barely fit a very small batch size in this scenario.

   - ``"offload_optimizer": {"device": "none"}``: 4 * params + 16 * params/ (total number of gpus)

* ZeRO-3:

``largest_layer_memory = 4*largest_layer_params`` - GPU memory needed to gather the largest layer on a single GPU. 2 bytes fp16 params are gathered and 2 bytes fp16 grads are computed (total 4x). The optimizer states and fp32 parameters are updated in partitioned form and copied to fp16 params in partitioned form. This happens during the optimizer step. After that the fp16 params are sufficient.

   - case 1: ``"offload_param": {"device": "none"}, "offload_optimizer": {"device": "none"}`` - largest_layer_memory + 18 * params / total number of gpus across all nodes
   - case 2: ``"offload_param": {"device": "cpu"}, "offload_optimizer": {"device": "cpu"}``- largest_layer_memory. The main limit here is general RAM.
   - case 3: ``"offload_param": {"device": "none"}, "offload_optimizer": {"device": "cpu"}``- largest_layer_memory + 2 * params / total number of gpus across all nodes

     Example:

.. code-block:: python

    from transformers import AutoModel
    model = AutoModel.from_pretrained("t5-large")

    # shared params calculated only ones
    total_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

    largest_layer_params = 0
    for m in model.modules():
        # assuming no shared params within a single layer
        layer_params = sum(p.numel() for p in m.parameters(recurse=False))
        largest_layer_params = max(largest_layer_params, layer_params)

    largest_layer_memory = (4*largest_layer_params)

    total_gpus = 4

    case1 = largest_layer_memory + int(18*total_params/total_gpus)
    case2 = largest_layer_memory
    case3 = largest_layer_memory + int(2*total_params/total_gpus)

    print(f"total params:         {total_params/1e6:6.2f}M")
    print(f"largest layer params: {largest_layer_params/1e6:6.2f}M")
    print(f"largest layer memory: {largest_layer_memory>>20:6}MB")
    print(f"case1 gpu memory: {(case1)>>20:6}MB")
    print(f"case2 gpu memory: {(case2)>>20:6}MB")
    print(f"case3 gpu memory: {(case3)>>20:6}MB")

    total params:         737.67M
    largest layer params:  32.90M
    largest layer memory:    125MB
    case1 gpu memory:   3291MB
    case2 gpu memory:    125MB
    case3 gpu memory:    477MB


**General RAM**:

One of the key features of ZeRO is its CPU offload which can dramatically extend the total memory pool accessible to the project by using general RAM. One can easily expand their general RAM by 10x times, at a significantly lower cost than what it'd take to have the same GPU RAM. And often, it's not even possible to buy GPUs with a lot of RAM (112GB GPU anybody?) since they simply don't yet exist.

In the following calculations we will use:

- ``additional_buffer_factor=1.5`` as an additional buffer factor to be conservative
- ``n_gpus`` the number of GPUs on a single node (machine)
- ``total_gpus`` the total number of GPUs across all nodes
- ``params`` - total number of model params (see above for how to get this number)

* ZeRO-2:

   - ``"offload_optimizer": {"device": "none"}``:

      params * 4 * n_gpus * additional_buffer_factor - this is the memory needed only at the beginning to initialize the model on CPU memory

   - ``"offload_optimizer": {"device": "cpu"}``:

      params * max(4 * n_gpus, 16) * additional_buffer_factor

   Example: xxx

* ZeRO-3:

   gpus_factor = n_gpus / total_gpus

   - case 1: ``"offload_param": {"device": "none"}, "offload_optimizer": {"device": "none"}``:

      Without ``zero.Init``:

          params * 4 * n_gpus * additional_buffer_factor

          this is the memory needed only at the beginning to initialize the model on CPU memory. Once the model is transferred to GPUs this memory is freed.

      With ``zero.Init``:

          largest_layer_params * 4 * n_gpus * additional_buffer_factor

          assuming Pytorch is deallocating the memory once the tensors are moved to the GPU by ZeRO.Init

   - case 2: ``"offload_param": {"device": "cpu"}, "offload_optimizer": {"device": "cpu"}``:

      Without ``zero.Init``:

          params * max(4 * n_gpus, 18 * gpus_factor) * additional_buffer_factor

      With ``zero.Init``:

          params * 18 * gpus_factor * additional_buffer_factor

   - case 3: ``"offload_param": {"device": "none"}, "offload_optimizer": {"device": "cpu"}``:

      Without ``zero.Init``:

          params * max(4 * n_gpus, 16 * gpus_factor) * additional_buffer_factor

      With ``zero.Init``:

          params * 16 * gpus_factor * additional_buffer_factor


Here is a breakdown for the 16 and 18 multipliers (b = bytes):

4 (in ``4*n_gpus``):

- when pytorch creates a model it creates it in fp32 by default (4 bytes)

16:

- 16b for fp32: 4b params, 4b grads, 4b momentum and 4b variance per parameter

18:

- 16b for fp32: 4b params, 4b grads, 4b momentum and 4b variance per parameter
- +2b for fp16 params

Note about gradients: While gradients are stored in fp16 (2 bytes), during the weight update, all of them are converted into fp32 before doing the weight updates since the weight updates are done at almost the entire model granularity (param_group granularity) in FusedAdam Optimizer in DeepSpeed. So after that conversion we would need the 4 bytes per gradient for nearly the entire set of weights.


**Pinned Memory**

Pinned general RAM is included in normal general RAM allocations (i.e. this is not extra memory allocations but simply shows how much of the general RAM is pinned). Pinning is controlled by the ``pin_memory`` field of ``offload_optimizer`` / ``offload_param`` (both default to ``true``); set to ``false`` on hosts with tight memlock limits (``ulimit -l``). See :ref:`host-memory-pinning` for how host memory is page-locked (accelerator / torch pinning and the DeepSpeed ``pin_memory`` op).

* ZeRO-1/2: controlled by ``offload_optimizer.pin_memory``

* ZeRO-3

With pinning enabled there are 2 sub-cases:

1. ``offload_param`` enabled (``device: cpu``):

   - 6 * params (2b for fp16 params + 4b for fp32 gradients)
   - if ``gradient_accumulation_steps > 1`` an additional 2b for fp16 gradients are pinned

2. ``offload_param`` not enabled:

   - 4b for fp32 gradients


**Activation Memory**

XXX: For Transformers is probably around (2* seq * attn_heads + 16 * hidden_size) * sequence * batch/gpu

This needs to be completed.


.. _host-memory-pinning:

Host Memory Pinning
-----------------------

DeepSpeed page-locks (pins) host memory so DMA engines can move data efficiently
between CPU RAM and accelerators or NVMe. Host tensors can be pinned either with
PyTorch / the accelerator hook, or with the standalone DeepSpeed ``pin_memory``
operator (``posix_memalign`` + ``mlock``).

Accelerator API (torch path)
============================

.. code-block:: python

    from deepspeed.accelerator import get_accelerator

    pinned = get_accelerator().pin_memory(tensor)
    assert get_accelerator().is_pinned(pinned)

On most accelerators this routes to ``tensor.pin_memory()``. The CPU accelerator
historically treats this as a no-op (torch cannot pin a CPU tensor to a device).

DeepSpeed ``pin_memory`` op
==========================

Allocate and free page-locked host tensors through ``PinMemoryBuilder`` /
``pin_handle``:

.. code-block:: python

    import torch
    from deepspeed.ops.op_builder import PinMemoryBuilder

    pin = PinMemoryBuilder().load().pin_handle()
    t = pin.new_cpu_locked_tensor(1024**3, torch.empty(0, dtype=torch.uint8))
    assert pin.is_pinned(t)
    pin.free_cpu_locked_tensor(t)

``pin_handle`` exposes:

* ``new_cpu_locked_tensor(num_elem, example_tensor)`` — allocate a page-locked CPU tensor
* ``free_cpu_locked_tensor(tensor)`` — unlock and free the allocation
* ``is_pinned(tensor)`` — ``True`` for torch-pinned buffers and for tensors whose
  storage falls inside a DeepSpeed-managed locked range (slices/views included)

The same three methods remain available on DeepNVMe I/O handles as thin wrappers.
Buffers allocated via ``pin_handle`` and via an I/O handle share one process-wide
pin manager, so DeepNVMe bounce-buffer skipping stays consistent across both paths.

.. list-table:: Differences between torch pinning and the DeepSpeed ``pin_memory`` op
   :header-rows: 1
   :widths: 22 39 39

   * - Aspect
     - Torch / accelerator ``pin_memory``
     - DeepSpeed ``pin_memory`` op
   * - Allocator
     - ``torch.Tensor.pin_memory()`` (device-specific accelerator hook)
     - ``posix_memalign`` + ``mlock`` via ``PinMemoryBuilder`` / ``pin_handle``
   * - Build dependency
     - None beyond the active accelerator / PyTorch
     - DeepSpeed **pin_memory** op must build and load
   * - Visible to DeepNVMe
     - Yes (``is_pinned`` / bounce-buffer skip on I/O handles)
     - Yes (same process-wide pin manager)
   * - ``is_pinned``
     - Torch pinned status (``tensor.is_pinned()``)
     - Torch-pinned **or** DeepSpeed-managed locked range (slices/views included)
   * - Explicit free
     - Freed when the tensor is garbage-collected (PyTorch)
     - ``free_cpu_locked_tensor`` releases the mlocked pages immediately
   * - CPU accelerator
     - Historical no-op (torch cannot pin CPU tensors)
     - Real ``mlock`` pins on CPU-only hosts

Requirements
============

* The DeepSpeed **pin_memory** op must build and load successfully.
* The process must be allowed to lock the requested host memory (see
  ``ulimit -l`` / ``memlock`` limits). Large ZeRO CPU-offload or DeepNVMe
  footprints can otherwise hit the memlock ceiling.

See also the DeepNVMe tutorial for end-to-end examples of pinning host and device
tensors for NVMe I/O.
