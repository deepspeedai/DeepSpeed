Training API
============

:func:`deepspeed.initialize` returns a *model engine* in its first argument
of type ``DeepSpeedLight``. This engine is used to progress training:

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
.. autofunction:: deepspeed.DeepSpeedLight.forward

Backward Propagation
--------------------
.. autofunction:: deepspeed.DeepSpeedLight.backward

Optimizer Step
--------------
.. autofunction:: deepspeed.DeepSpeedLight.step
