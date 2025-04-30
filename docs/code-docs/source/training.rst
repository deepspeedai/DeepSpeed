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

Forward Propagation
-------------------
.. autofunction:: deepspeed.DeepSpeedEngine.forward

Backward Propagation
--------------------
.. autofunction:: deepspeed.DeepSpeedEngine.backward

Optimizer Step
--------------
.. autofunction:: deepspeed.DeepSpeedEngine.step

Gradient Accumulation
---------------------
.. autofunction:: deepspeed.DeepSpeedEngine.is_gradient_accumulation_boundary


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
            engine._backward_epilogue()

        for engine in model_engines:
            engine.step()

        for engine in model_engines:
            engine.optimizer.zero_grad()

Besides the use of multiple DeepSpeedEngines, the above differs from typical usage in two key ways:

#. The **backward** call is made using the common loss value rather on individual model engines.

#. **_backward_epilogue** is called on model engine, after the **loss.backward()**.
