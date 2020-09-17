Training API
============

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
