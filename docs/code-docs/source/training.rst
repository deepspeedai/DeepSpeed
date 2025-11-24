Training API
############

:func:`deepspeed.initialize` returns a *training engine* in its first argument
of type :class:`DeepSpeedEngine`. This engine is used to progress training:

.. code-block:: python

    for step, batch in enumerate(data_loader):
        #forward() method
        loss = model_engine(batch)

        #runs backpropagation
        model_engine.backward(loss)

        #weight update
        model_engine.step()

Note that ``model_engine.backward()`` accepts only a scalar loss tensor produced by a forward pass.
Starting from v0.18.3, DeepSpeed also supports direct calls to ``tensor.backward()``. You can now call
``loss.backward()`` or ``tensor.backward(out_grad)`` when your PyTorch version supports the necessary APIs.
If your PyTorch version does not support these APIs, a direct call to ``tensor.backward()`` will raise an error.

Forward Propagation
-------------------
.. autofunction:: deepspeed.DeepSpeedEngine.forward

Backward Propagation
--------------------
.. autofunction:: deepspeed.DeepSpeedEngine.backward

Loss Scaling for Manual Backward Passes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: deepspeed.DeepSpeedEngine.scale

When using mixed precision training (fp16, bf16, or torch.autocast), DeepSpeed applies loss scaling
to prevent gradient underflow. If you prefer to call ``loss.backward()`` directly instead of
``engine.backward(loss)``, you must use ``engine.scale(loss)`` to apply the appropriate loss scaler:

.. code-block:: python

    # Option 1: Use engine.backward() (recommended)
    loss = model_engine(batch)
    model_engine.backward(loss)

    # Option 2: Manual backward with scaling
    loss = model_engine(batch)
    scaled_loss = model_engine.scale(loss)
    scaled_loss.backward()

Both approaches produce identical gradients. The ``scale()`` method automatically applies the
appropriate scaler based on your configuration (ZeRO optimizer scaler, torch.autocast GradScaler, etc.).

Optimizer Step
--------------
.. autofunction:: deepspeed.DeepSpeedEngine.step

Gradient Accumulation
---------------------
.. autofunction:: deepspeed.DeepSpeedEngine.is_gradient_accumulation_boundary


Mixed Precision Training
-------------------------
DeepSpeed supports mixed precision training using either native or PyTorch mechanisms. The desired mixed precision mode can be selected through the configuration dict.
Mixed precision training can used with ZeRO (i.e., stages > 0) and without ZeRO (i.e., stage=0).


Native Mixed Precision
======================================================
DeepSpeed provides native support for
`fp16 <https://www.deepspeed.ai/docs/config-json/#fp16-training-options>`_ and `bf16 <https://www.deepspeed.ai/docs/config-json/#bfloat16-training-options>`_ mixed precsion training.


PyTorch Automatic Mixed Precision (AMP)
======================================================
DeepSpeed provides torch-compatible automatic mixed precision (AMP) training via
`torch.autocast <https://docs.pytorch.org/docs/stable/amp.html>`_ functionality.  The following snippet illustrates how to enable Torch AMP.

    .. code-block:: python

        {
            "torch_autocast": {
                "enabled": true,
                "dtype": "bfloat16",
                "lower_precision_safe_modules": ["torch.nn.Linear", "torch.nn.Conv2d"]
            },
            ...
        }

Each configuration works as follows:

* ``enabled``: Enable ``torch.autocast`` when set to ``True``. You don't need to call ``torch.autocast`` in your code. The grad scaler is also applied in the DeepSpeed optimizer.
* ``dtype``: Lower precision dtype passed to ``torch.autocast``. Gradients for all-reduce (reduce-scatter) and parameters for all-gather (only for ZeRO3) of ``lower_precision_safe_modules`` are also downcasted to this ``dtype``.
* ``lower_precision_safe_modules``: The list of modules that will be downcasted for all-reduce (reduce-scatter) and all-gather (ZeRO3). The precision for PyTorch operators in forward/backward follows ``torch.autocast``'s policy, not this list. If you don't set this item, DeepSpeed uses the default list: ``[torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]``.

Manual Backward with torch.autocast
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using ``torch.autocast`` with manual backward passes (``loss.backward()`` instead of ``engine.backward()``),
you must use ``engine.scale(loss)`` to apply the gradient scaler:

.. code-block:: python

    # Training loop with torch.autocast and manual backward
    for batch in data_loader:
        loss = model_engine(batch)

        # Apply loss scaling before manual backward
        scaled_loss = model_engine.scale(loss)
        scaled_loss.backward()

        model_engine.step()

The ``scale()`` method ensures that the ``torch.amp.GradScaler`` is properly applied when ``torch.autocast``
is enabled with fp16. For bf16 or when no mixed precision is used, ``scale()`` returns the loss unchanged.

If you call ``loss.backward()`` directly without using ``engine.scale()`` or ``engine.backward()``, DeepSpeed
will raise a ``RuntimeError`` to prevent training with unscaled gradients, which can lead to incorrect results
or gradient underflow.

.. autofunction:: deepspeed.runtime.torch_autocast.init_autocast_params
.. autofunction:: deepspeed.runtime.torch_autocast.is_autocast_initialized
.. autofunction:: deepspeed.runtime.torch_autocast.get_default_autocast_lower_precision_modules
.. autofunction:: deepspeed.runtime.torch_autocast.has_autocast_dtype


Configuring ZeRO Leaf Modules
-----------------------------

ZeRO-3 relies on module execution order to gather partitioned parameters.
When models select submodules dynamically (for example, MoE routers), different data-parallel ranks may gather different sets of parameters, which can cause the all-gather collective to deadlock.
To avoid this problem, you can designate the parent of dynamically activated submodules (e.g., MoE experts) as a "leaf" module.
When a module is marked as a leaf, ZeRO gathers all of its descendants immediately and stops inserting hooks beneath it.

Programmatic API
================

Use :func:`deepspeed.utils.set_z3_leaf_modules` to flag modules by class, class
name, or both. Optionally combine with
:func:`deepspeed.utils.set_z3_leaf_modules_by_name` to target specific entries
from ``model.named_modules()`` or
:func:`deepspeed.utils.set_z3_leaf_modules_by_suffix` to match suffixes of those
names.

.. code-block:: python

    from deepspeed.utils import (
        set_z3_leaf_modules,
        set_z3_leaf_modules_by_name,
        set_z3_leaf_modules_by_suffix,
    )

    # Match by class or subclass
    set_z3_leaf_modules(model, [CustomMoEBlock])

    # Match by fully qualified class name
    set_z3_leaf_modules(model, ["my_package.layers.CustomMoEBlock"])

    # Match by module name returned from model.named_modules()
    set_z3_leaf_modules_by_name(model, ["transformer.layers.0.experts"])

    # Match by suffix of names returned from model.named_modules()
    set_z3_leaf_modules_by_suffix(model, ["experts"])

Configuration in DeepSpeed config
=================================

The same behavior can be controlled from the DeepSpeed config. Add a
``leaf_module`` block to ``zero_optimization`` specifying either classes,
module names, or name suffixes (or any combination). While the example below shows three different ways (``classes``, ``names``, and ``name_suffixes``) to specify modules as leaf modules, typically you will use just one of these.

.. code-block:: json

    {
      "train_micro_batch_size_per_gpu": 1,
      "zero_optimization": {
        "stage": 3,
        "leaf_module": {
          "classes": ["my_package.layers.CustomMoEBlock"],
          "names": ["transformer.layers.0.experts"],
          "name_suffixes": ["experts"]
        }
      }
    }

``names`` must match exactly what ``model.named_modules()`` produces. The
``name_suffixes`` field compares each suffix against the end of those same
module paths, making it convenient to apply a rule across repeated structures.
Entries in ``classes`` may be either bare class names (for example,
``MixtralSparseMoeBlock``) or fully qualified dotted paths; both forms are
accepted.

You can mix and match the API and configuration approaches; all referenced
modules are flagged before ZeRO installs its hooks.

By default DeepSpeed marks several Hugging Face MoE blocks—including Mixtral and Qwen MoE sparse blocks so that they behave well with ZeRO3. The default class list currently contains:

* ``transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock``
* ``transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock``
* ``transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock``


Model Saving
------------
.. autofunction:: deepspeed.DeepSpeedEngine.save_16bit_model


Additionally when a DeepSpeed checkpoint is created, a script ``zero_to_fp32.py`` is added there which can be used to reconstruct fp32 master weights into a single pytorch ``state_dict`` file.


Training Multiple Models
------------------------
DeepSpeed supports training multiple models, which is a useful feature in `scenarios <https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed_multiple_model>`_ such as knowledge distillation and post-training RLHF.
The core approach is to create individual DeepSpeedEngines for each model.


Training Independent Models
===========================

The following code snippet illustrates independently training multiple models on the same dataset.

.. code-block:: python

    model_engines = [engine for engine, _, _, _ in [deepspeed.initialize(m, ...,) for m in models]]
    for batch in data_loader:
       losses = [engine(batch) for engine in model_engines]
       for engine, loss in zip(model_engines, losses):
          engine.backward(loss)

The above is similar to typical DeepSpeed usage except for the creation of multiple DeepSpeedEngines (one for each model).


Jointly Training Models With Shared Loss
========================================

The following code snippet illustrates jointly training multiple models on a shared loss value.

.. code-block:: python

    model_engines = [engine for engine, _, _, _ in [deepspeed.initialize(m, ...,) for m in models]]
    for batch in data_loader:
        losses = [engine(batch[0], batch[1]) for engine in model_engines]
        loss = sum(l / (i + 1) for i, l in enumerate(losses))
        loss.backward()

        for engine in model_engines:
            engine.step()

        for engine in model_engines:
            engine.optimizer.zero_grad()

Besides the use of multiple DeepSpeedEngines, the above differs from typical usage in two key ways:

#. The **backward** call is made using the common loss value rather on individual model engines.

You can call ``loss.backward()`` once for the shared loss.

**Note:** Previously, you had to call ``_backward_epilogue`` on each model engine after ``loss.backward()``. However, starting from v0.18.3, DeepSpeed automatically handles this internally, so you no longer need to call ``_backward_epilogue`` manually.
